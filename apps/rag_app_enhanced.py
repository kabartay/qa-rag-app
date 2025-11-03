"""
Enhanced RAG with Pinecone as Vector DB and Cohere for re-ranking.
Handles large documents and multi-document collections.
Added Redis caching to reduce cost.
"""

import hashlib
import json
import os
import random
import time
from typing import Any, cast

import anthropic
import cohere
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from apps.monitoring import get_monitoring_status, log_cache_event, traced

# Load environment variables
load_dotenv()


class EnhancedRAG:
    """RAG with vector search and re-ranking"""

    def __init__(self, anthropic_key: str, pinecone_key: str, cohere_key: str) -> None:
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        self.cohere_client = cohere.Client(cohere_key)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_key)
        self.index_name: str = "rag-documents"
        self.embedding_dimension: int = 1024

        # Create or connect to index
        self._setup_index()

        # Explicit type annotations
        self.chunks: list[dict[str, Any]] = []
        self.document_metadata: dict[str, Any] = {}

    def _setup_index(self) -> None:
        """Create or connect to Pinecone index"""
        try:
            existing_indexes = self.pc.list_indexes()

            if self.index_name not in [idx["name"] for idx in existing_indexes]:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                time.sleep(1)

            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            st.error(f"Pinecone setup error: {str(e)}")
            self.index = None

    def _chunk_text_smart(
        self, text: str, chunk_size: int = 800, overlap: int = 100
    ) -> list[dict[str, Any]]:
        """Smart chunking with paragraph preservation and overlap."""
        chunks: list[dict[str, Any]] = []
        paragraphs = text.split("\n\n")
        current_chunk: list[str] = []
        current_size = 0
        chunk_id = 0

        for para in paragraphs:
            para_words = para.split()
            para_size = len(para_words)

            if current_size + para_size > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "word_count": len(current_chunk),
                        "char_count": len(chunk_text),
                    }
                )

                overlap_words = (
                    current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                )
                current_chunk = overlap_words + para_words
                current_size = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk.extend(para_words)
                current_size += para_size

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "word_count": len(current_chunk),
                    "char_count": len(chunk_text),
                }
            )

        return chunks

    def _get_embeddings_cohere(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using Cohere, or fallback if unavailable."""
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-multilingual-v3.0",
                input_type="search_document",
                embedding_types=["float"],
            )

            # Cast because Cohere's return type is Any
            embeddings = cast(list[list[float]], response.embeddings.float)
            return embeddings

        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            embeddings = []
            for text in texts:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                random.seed(text_hash)
                embedding = [random.random() for _ in range(self.embedding_dimension)]
                embeddings.append(embedding)
            return embeddings

    @traced
    def load_document(self, text: str, doc_name: str = "document") -> bool:
        """Load document into vector database"""
        self.chunks = self._chunk_text_smart(text)
        st.info(f"Created {len(self.chunks)} chunks")

        if not self.index:
            st.error("Pinecone index not available")
            return False

        chunk_texts = [chunk["text"] for chunk in self.chunks]

        with st.spinner("Generating embeddings with Cohere..."):
            embeddings = self._get_embeddings_cohere(chunk_texts)

        vectors: list[dict[str, Any]] = []
        for i, (chunk, embedding) in enumerate(zip(self.chunks, embeddings, strict=False)):
            vectors.append(
                {
                    "id": f"{doc_name}_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk["text"],
                        "chunk_id": chunk["id"],
                        "doc_name": doc_name,
                        "word_count": chunk["word_count"],
                    },
                }
            )

        batch_size = 100
        with st.spinner("Uploading to Pinecone..."):
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch)

        st.success(f"Loaded {len(vectors)} chunks to Pinecone")
        return True

    @traced
    def retrieve_and_rerank(
        self, query: str, top_k: int = 20, rerank_top_n: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve from Pinecone and rerank with Cohere."""
        query_embedding = self._get_embeddings_cohere([query])[0]
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        if not results.matches:
            return []

        candidates: list[dict[str, Any]] = []
        for match in results.matches:
            candidates.append(
                {
                    "text": match.metadata["text"],
                    "score": match.score,
                    "chunk_id": match.metadata.get("chunk_id"),
                    "doc_name": match.metadata.get("doc_name"),
                }
            )

        try:
            rerank_response = self.cohere_client.rerank(
                query=query,
                documents=[c["text"] for c in candidates],
                top_n=rerank_top_n,
                model="rerank-multilingual-v3.0",
            )

            reranked: list[dict[str, Any]] = []
            for result in rerank_response.results:
                original = candidates[result.index]
                reranked.append(
                    {
                        "text": original["text"],
                        "relevance_score": result.relevance_score,
                        "vector_score": original["score"],
                        "chunk_id": original["chunk_id"],
                        "doc_name": original["doc_name"],
                    }
                )
            return reranked

        except Exception as e:
            st.warning(f"Rerank failed: {str(e)}, using vector search only")
            return candidates[:rerank_top_n]

    @traced
    def answer_question(self, question: str) -> dict[str, Any]:
        """Answer question using enhanced RAG pipeline"""
        relevant_chunks = self.retrieve_and_rerank(question, top_k=20, rerank_top_n=5)

        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the document.",
                "context_used": "",
                "chunks_used": 0,
                "scores": [],
            }

        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(
                f"[Chunk {i+1}] (Relevance: {chunk.get('relevance_score', chunk.get('vector_score', 0)):.3f})\n"
                f"{chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are an expert assistant analyzing technical documents.

Retrieved Context (ranked by relevance):
{context}

User Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain sufficient information, state this clearly
- Cite which chunks you used (e.g., "According to Chunk 1...")
- Maintain accuracy and avoid hallucinations

Answer:"""

        try:
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            answer = message.content[0].text

            return {
                "answer": answer,
                "context_used": context,
                "chunks_used": len(relevant_chunks),
                "scores": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "relevance": c.get("relevance_score", c.get("vector_score")),
                    }
                    for c in relevant_chunks
                ],
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context_used": context,
                "chunks_used": len(relevant_chunks),
                "scores": [],
            }


class RedisCache:
    """Redis cache to reduce cost."""

    def __init__(self, redis_url: str | None = None, ttl: int = 3600, enabled: bool = True) -> None:
        self.ttl = ttl
        self.enabled = False
        self.redis_client: "redis.Redis" | None = None

        # Check environment variable for cache control
        use_cache = os.getenv("USE_CACHE", "true").lower() in ("true", "1", "yes")
        if not enabled or not use_cache or not redis_url:
            return

        try:
            import redis

            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            self.redis_client.ping()
            self.enabled = True
        except ImportError:
            st.warning("Redis package not installed.")
        except Exception as e:
            st.warning(f"Redis connection failed: {e}")

    def get(self, question: str, doc_id: str = "default") -> dict[str, Any] | None:
        if not self.enabled or not self.redis_client:
            return None
        try:
            key = f"rag:{hashlib.md5(f'{doc_id}:{question}'.encode()).hexdigest()}"
            cached = self.redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            st.warning(f"Cache get error: {e}")
            return None

    def set(self, question: str, response: dict[str, Any], doc_id: str = "default") -> bool:
        if not self.enabled or not self.redis_client:
            return False
        try:
            key = f"rag:{hashlib.md5(f'{doc_id}:{question}'.encode()).hexdigest()}"
            self.redis_client.setex(key, self.ttl, json.dumps(response))
            return True
        except Exception as e:
            st.warning(f"Cache set error: {e}")
            return False

    def clear(self, pattern: str = "rag:*") -> int:
        if not self.enabled or not self.redis_client:
            return 0
        try:
            keys = list(self.redis_client.scan_iter(pattern))
            count: int = int(self.redis_client.delete(*keys)) if keys else 0
            return count
        except Exception as e:
            st.warning(f"Cache clear error: {e}")
            return 0

    def stats(self) -> dict[str, Any]:
        if not self.enabled or not self.redis_client:
            return {"enabled": False, "message": "Cache disabled (Redis not available)"}
        try:
            info = self.redis_client.info("stats")
            keys_count = len(list(self.redis_client.scan_iter("rag:*")))
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            return {
                "enabled": True,
                "total_keys": keys_count,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / max(hits + misses, 1) * 100,
            }
        except Exception as e:
            st.warning(f"Cache stats error: {e}")
            return {"enabled": False, "error": "Stats unavailable"}


def render_cache_sidebar(cache: "RedisCache") -> None:
    """Render cache status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("Cache Status")

    if not cache.enabled:
        st.sidebar.info("Cache disabled")
        st.sidebar.caption("Set REDIS_URL in .env to enable")
        st.sidebar.caption("Or set USE_CACHE=false to explicitly disable")
        return

    stats = cache.stats()

    if stats.get("enabled"):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Cached Items", stats.get("total_keys", 0))
            st.metric("Cache Hits", stats.get("hits", 0))
        with col2:
            st.metric("Hit Rate", f"{stats.get('hit_rate', 0):.1f}%")
            st.metric("Cache Misses", stats.get("misses", 0))

        if stats.get("hits", 0) > 0:
            savings = stats["hits"] * 0.025
            st.sidebar.success(f"Saved ~${savings:.2f}")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.sidebar.button("Clear Cache"):
                count = cache.clear()
                st.sidebar.success(f"Cleared {count} items")
                st.rerun()
        with col2:
            if st.sidebar.button("Refresh"):
                st.rerun()

        st.sidebar.success("Cache Active")
    else:
        st.sidebar.error("Cache Error")
        st.sidebar.caption(stats.get("error", "Unknown error"))


def main() -> None:
    load_dotenv(override=True)  # Ensure .env is loaded before cache
    st.set_page_config(page_title="Enhanced RAG System", page_icon="🔸", layout="wide")

    st.title("Enhanced RAG-based Question-Answering System")
    st.markdown("**RAG with Pinecone Vector DB + Cohere Re-ranking**")

    # Initialize cache
    if "cache" not in st.session_state:
        use_cache = os.getenv("USE_CACHE", "true").lower() in ("true", "1", "yes")
        redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")

        # Detect Codespaces / Docker environment
        if os.path.exists("/.dockerenv"):
            # Codespaces usually runs Redis on localhost
            if "CODESPACES" in os.environ:
                redis_url = "redis://127.0.0.1:6379"
            else:
                redis_url = "redis://host.docker.internal:6379"

        # Debug
        st.sidebar.info(f"Using Redis at: {redis_url}")

        st.session_state.cache = RedisCache(redis_url=redis_url, ttl=3600, enabled=use_cache)

        # Show cache in the sidebar (only for demo)
        render_cache_sidebar(st.session_state.cache)

        # Monitoring status (only for demo)
        st.sidebar.markdown("### Monitoring Status")
        status = get_monitoring_status()

        if status["langsmith_enabled"]:
            st.sidebar.success(f"LangSmith ({status['project']})")
        else:
            st.sidebar.warning("LangSmith disabled")

        if status["prometheus_enabled"]:
            st.sidebar.info(f"Prometheus active on port {status['prometheus_port']}")
        else:
            st.sidebar.warning("Prometheus not running")

    # Sidebar configuration
    with st.sidebar:
        st.header("API Keys")

        api_keys = {
            "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "Pinecone": bool(os.getenv("PINECONE_API_KEY")),
            "Cohere": bool(os.getenv("COHERE_API_KEY")),
        }

        if all(api_keys.values()):
            st.success("All API keys loaded from environment")
        elif any(api_keys.values()):
            loaded = [k for k, v in api_keys.items() if v]
            st.info(f"Loaded from env: {', '.join(loaded)}")
        else:
            st.warning("No API keys found in environment")

        anthropic_key = st.text_input(
            "Anthropic API Key" + (" ✓" if api_keys["Anthropic"] else ""),
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help=(
                "Loaded from ANTHROPIC_API_KEY env var"
                if api_keys["Anthropic"]
                else "Enter your API key"
            ),
        )
        pinecone_key = st.text_input(
            "Pinecone API Key" + (" ✓" if api_keys["Pinecone"] else ""),
            value=os.getenv("PINECONE_API_KEY", ""),
            type="password",
            help=(
                "Loaded from PINECONE_API_KEY env var"
                if api_keys["Pinecone"]
                else "Enter your API key"
            ),
        )
        cohere_key = st.text_input(
            "Cohere API Key" + (" ✓" if api_keys["Cohere"] else ""),
            value=os.getenv("COHERE_API_KEY", ""),
            type="password",
            help=(
                "Loaded from COHERE_API_KEY env var" if api_keys["Cohere"] else "Enter your API key"
            ),
        )

        st.markdown("---")
        st.header("RAG Configuration")

        _retrieval_top_k = st.slider("Vector search top-K", 10, 50, 20)
        _rerank_top_n = st.slider("Rerank top-N", 3, 10, 5)

        st.markdown("---")
        st.header("System Architecture")
        st.markdown(
            """
        **Pipeline:**
        1. Chunk documents (800 words, 100 overlap)
        2. Generate embeddings (Cohere)
        3. Store in Pinecone (cosine similarity)
        4. Query: Vector search (top-20)
        5. Cohere re-rank (top-5)
        6. Claude generates answer
        """
        )

        st.markdown("---")
        st.header("Benefits:")
        st.markdown("Handles large documents (1000+ pages)")
        st.markdown("Multi-document search")
        st.markdown("Better precision with re-ranking")
        st.markdown("Scalable to millions of chunks")
        st.markdown("Uses Cohere for embeddings + reranking")

    # Initialize RAG
    if "enhanced_rag" not in st.session_state:
        st.session_state.enhanced_rag = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check API keys
    all_keys_provided = all([anthropic_key, pinecone_key, cohere_key])

    # Document loading
    st.header("1 Load Document")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])

    document_text = ""

    with tab1:
        st.markdown("**Upload a PDF file**")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type=["pdf"], key="enhanced_pdf_upload"
        )

        if uploaded_file:
            try:
                from io import BytesIO

                import PyPDF2

                with st.spinner("Extracting text from PDF..."):
                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                    document_text = ""
                    for page in pdf_reader.pages:
                        document_text += page.extract_text() + "\n\n"

                    st.success(
                        f"Extracted {len(document_text)} characters from PDF ({len(pdf_reader.pages)} pages)"
                    )

                    # Show preview
                    with st.expander("Preview extracted text"):
                        st.text(
                            document_text[:1000] + "..."
                            if len(document_text) > 1000
                            else document_text
                        )
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                st.info("Try the 'Paste Text' tab instead")

    with tab2:
        st.markdown("**Paste document text directly**")
        pasted_text = st.text_area(
            "Document text",
            height=300,
            help="Paste the text content of your document",
            key="enhanced_text_paste",
        )
        if pasted_text:
            document_text = pasted_text
            st.info(f"{len(document_text)} characters entered")

    col1, col2 = st.columns([2, 1])

    with col1:
        doc_name = st.text_input("Document name", "doc_1")

    with col2:
        st.markdown("**Scalability**")
        st.metric("Max document size", "Unlimited")
        st.metric("Chunk size", "800 words")
        st.metric("Overlap", "100 words")

        if document_text:
            word_count = len(document_text.split())
            st.metric("Current doc", f"{word_count:,} words")

    if st.button(
        "Load to Pinecone", type="primary", disabled=not all_keys_provided or not document_text
    ):
        if not all_keys_provided:
            st.error("Please provide all API keys in the sidebar")
        elif not document_text:
            st.error("Please provide a document (upload PDF or paste text)")
        else:
            # Initialize RAG system
            with st.spinner("Initializing RAG system..."):
                st.session_state.enhanced_rag = EnhancedRAG(anthropic_key, pinecone_key, cohere_key)

            # Load document
            success = st.session_state.enhanced_rag.load_document(document_text, doc_name)

            if success:
                st.balloons()

    # Q&A Interface
    st.header("2 Ask Questions")

    if st.session_state.enhanced_rag is None:
        st.info("Load a document first")
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if "metadata" in message:
                    with st.expander("🔍 Retrieval Details"):
                        meta = message["metadata"]
                        st.write(f"**Chunks used:** {meta['chunks_used']}")

                        if meta.get("scores"):
                            st.write("**Relevance scores:**")
                            for score in meta["scores"][:3]:
                                st.write(f"  - Chunk {score['chunk_id']}: {score['relevance']:.3f}")

                        st.text_area(
                            "Context",
                            meta.get("context", "")[:1000] + "...",
                            height=150,
                            key=f"context_meta_{int(time.time() * 1000)}",  # to avoid same label text and no unique key
                        )

        # Chat input
        if question := st.chat_input("Ask anything about your documents..."):

            cache = st.session_state.cache

            # Check if cached
            cached = cache.get(question, doc_name)
            log_cache_event(hit=bool(cached))  # monitoring
            if cached:
                result = cached
                st.toast("Cached result used!", icon="⚡")
            else:
                with st.spinner("Searching and analyzing..."):
                    result = st.session_state.enhanced_rag.answer_question(question)
                    cache.set(question, result, doc_name)

            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            # Show assistant answer (either cached or fresh)
            with st.chat_message("assistant"):
                st.markdown(result["answer"])

                with st.expander("Retrieval Details"):
                    st.write(f"**Chunks used:** {result['chunks_used']}")

                    if result.get("scores"):
                        st.write("**Top relevance scores:**")
                        for score in result["scores"][:3]:
                            st.write(f"  - Chunk {score['chunk_id']}: {score['relevance']:.3f}")

                    st.text_area(
                        "Retrieved Context",
                        result["context_used"][:1000] + "...",
                        height=150,
                        key=f"context_{len(st.session_state.messages)}",
                    )

            # Save to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": {
                        "chunks_used": result["chunks_used"],
                        "scores": result.get("scores", []),
                        "context": result["context_used"],
                    },
                }
            )

        # Clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Statistics
    if st.session_state.enhanced_rag:
        st.header("System Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Chunks", len(st.session_state.enhanced_rag.chunks))

        with col2:
            st.metric(
                "Questions Asked",
                len([m for m in st.session_state.messages if m["role"] == "user"]),
            )

        with col3:
            avg_chunks = sum(
                m.get("metadata", {}).get("chunks_used", 0)
                for m in st.session_state.messages
                if m["role"] == "assistant"
            )
            avg_chunks = avg_chunks / max(
                len([m for m in st.session_state.messages if m["role"] == "assistant"]), 1
            )
            st.metric("Avg Chunks/Answer", f"{avg_chunks:.1f}")

        with col4:
            st.metric("Vector DB", "Pinecone")


if __name__ == "__main__":
    main()
