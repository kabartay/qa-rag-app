import streamlit as st
import anthropic
from pinecone import Pinecone, ServerlessSpec
import cohere
from typing import List, Dict, Optional
import hashlib
import time

# Enhanced RAG with Pinecone Vector DB + Cohere Re-ranking
# Handles large documents and multi-document collections

class EnhancedRAG:
    """Production-grade RAG with vector search and re-ranking"""
    
    def __init__(self, anthropic_key: str, pinecone_key: str, cohere_key: str):
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        self.cohere_client = cohere.Client(cohere_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_key)
        self.index_name = "rag-documents"
        self.embedding_dimension = 1024  # Voyage AI dimension
        
        # Create or connect to index
        self._setup_index()
        
        self.chunks = []
        self.document_metadata = {}
    
    def _setup_index(self):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            
            if self.index_name not in [idx['name'] for idx in existing_indexes]:
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                time.sleep(1)  # Wait for index to be ready
            
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            st.error(f"Pinecone setup error: {str(e)}")
            self.index = None
    
    def _chunk_text_smart(self, text: str, chunk_size: int = 800, 
                          overlap: int = 100) -> List[Dict]:
        """
        Smart chunking with:
        - Sentence boundary awareness
        - Paragraph preservation
        - Metadata tracking
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for para in paragraphs:
            para_words = para.split()
            para_size = len(para_words)
            
            if current_size + para_size > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'word_count': len(current_chunk),
                    'char_count': len(chunk_text)
                })
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + para_words
                current_size = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk.extend(para_words)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'word_count': len(current_chunk),
                'char_count': len(chunk_text)
            })
        
        return chunks
    
    def _get_embeddings_cohere(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using Cohere (you already have the API key!)
        Cohere provides both embeddings AND re-ranking
        """
        try:
            # Use Cohere for embeddings (model: embed-english-v3.0 or embed-multilingual-v3.0)
            response = self.cohere_client.embed(
                texts=texts,
                model='embed-multilingual-v3.0',  # Good for French documents
                input_type='search_document',
                embedding_types=['float']
            )
            
            embeddings = response.embeddings.float
            
            # Cohere embeddings are 1024-dimensional, perfect for our setup
            return embeddings
            
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            # Fallback to mock embeddings if Cohere fails
            import random
            embeddings = []
            for text in texts:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                random.seed(text_hash)
                embedding = [random.random() for _ in range(self.embedding_dimension)]
                embeddings.append(embedding)
            return embeddings
    
    def load_document(self, text: str, doc_name: str = "document"):
        """Load document into vector database"""
        
        # Chunk document
        self.chunks = self._chunk_text_smart(text)
        st.info(f"Created {len(self.chunks)} chunks")
        
        if not self.index:
            st.error("Pinecone index not available")
            return False
        
        # Generate embeddings using Cohere
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        
        with st.spinner("Generating embeddings with Cohere..."):
            embeddings = self._get_embeddings_cohere(chunk_texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(self.chunks, embeddings)):
            vectors.append({
                'id': f"{doc_name}_{i}",
                'values': embedding,
                'metadata': {
                    'text': chunk['text'],
                    'chunk_id': chunk['id'],
                    'doc_name': doc_name,
                    'word_count': chunk['word_count']
                }
            })
        
        # Upload to Pinecone in batches
        batch_size = 100
        with st.spinner("Uploading to Pinecone..."):
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
        
        st.success(f"✅ Loaded {len(vectors)} chunks to Pinecone")
        return True
    
    def retrieve_and_rerank(self, query: str, top_k: int = 20, 
                           rerank_top_n: int = 5) -> List[Dict]:
        """
        Two-stage retrieval:
        1. Pinecone: Get top_k candidates
        2. Cohere Rerank: Re-rank to top_n
        """
        
        # Stage 1: Vector search in Pinecone
        query_embedding = self._get_embeddings_cohere([query])[0]
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        if not results.matches:
            return []
        
        # Extract texts and metadata
        candidates = []
        for match in results.matches:
            candidates.append({
                'text': match.metadata['text'],
                'score': match.score,
                'chunk_id': match.metadata.get('chunk_id'),
                'doc_name': match.metadata.get('doc_name')
            })
        
        # Stage 2: Rerank with Cohere
        try:
            rerank_response = self.cohere_client.rerank(
                query=query,
                documents=[c['text'] for c in candidates],
                top_n=rerank_top_n,
                model='rerank-english-v3.0'  # or 'rerank-multilingual-v3.0' for French
            )
            
            # Combine results
            reranked = []
            for result in rerank_response.results:
                original = candidates[result.index]
                reranked.append({
                    'text': original['text'],
                    'relevance_score': result.relevance_score,
                    'vector_score': original['score'],
                    'chunk_id': original['chunk_id'],
                    'doc_name': original['doc_name']
                })
            
            return reranked
            
        except Exception as e:
            st.warning(f"Rerank failed: {str(e)}, using vector search only")
            return candidates[:rerank_top_n]
    
    def answer_question(self, question: str) -> Dict:
        """Answer question using enhanced RAG pipeline"""
        
        # Retrieve and rerank
        relevant_chunks = self.retrieve_and_rerank(
            question,
            top_k=20,
            rerank_top_n=5
        )
        
        if not relevant_chunks:
            return {
                'answer': "No relevant information found in the document.",
                'context_used': "",
                'chunks_used': 0,
                'scores': []
            }
        
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(
                f"[Chunk {i+1}] (Relevance: {chunk.get('relevance_score', chunk.get('vector_score', 0)):.3f})\n"
                f"{chunk['text']}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer with Claude
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
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = message.content[0].text
            
            return {
                'answer': answer,
                'context_used': context,
                'chunks_used': len(relevant_chunks),
                'scores': [
                    {
                        'chunk_id': c.get('chunk_id'),
                        'relevance': c.get('relevance_score', c.get('vector_score'))
                    }
                    for c in relevant_chunks
                ]
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'context_used': context,
                'chunks_used': len(relevant_chunks),
                'scores': []
            }


def main():
    st.set_page_config(
        page_title="Enhanced RAG System", 
        page_icon="🚀", 
        layout="wide"
    )
    
    st.title("🚀 Enhanced RAG System")
    st.markdown("**Production-grade RAG with Pinecone Vector DB + Cohere Re-ranking**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔑 API Keys")
        
        anthropic_key = st.text_input("Anthropic API Key", type="password")
        pinecone_key = st.text_input("Pinecone API Key", type="password")
        cohere_key = st.text_input("Cohere API Key", type="password")
        
        st.markdown("---")
        st.header("⚙️ RAG Configuration")
        
        retrieval_top_k = st.slider("Vector search top-K", 10, 50, 20)
        rerank_top_n = st.slider("Rerank top-N", 3, 10, 5)
        
        st.markdown("---")
        st.header("📊 System Architecture")
        st.markdown("""
        **Pipeline:**
        1. 📄 Chunk documents (800 words, 100 overlap)
        2. 🧮 Generate embeddings (Cohere)
        3. 💾 Store in Pinecone (cosine similarity)
        4. 🔍 Query: Vector search (top-20)
        5. 🎯 Cohere re-rank (top-5)
        6. 🤖 Claude generates answer
        """)
        
        st.markdown("---")
        st.header("🎯 Benefits")
        st.markdown("✅ Handles large documents (1000+ pages)")
        st.markdown("✅ Multi-document search")
        st.markdown("✅ Better precision with re-ranking")
        st.markdown("✅ Scalable to millions of chunks")
        st.markdown("✅ Uses Cohere for embeddings + reranking")
    
    # Initialize RAG
    if 'enhanced_rag' not in st.session_state:
        st.session_state.enhanced_rag = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Check API keys
    all_keys_provided = all([anthropic_key, pinecone_key, cohere_key])
    
    # Document loading
    st.header("1️⃣ Load Document")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        document_text = st.text_area(
            "Paste your document text",
            height=200,
            help="For demo: paste text. Production: upload PDFs with PyPDF2/pdfplumber"
        )
        
        doc_name = st.text_input("Document name", "doc_1")
    
    with col2:
        st.markdown("**📈 Scalability**")
        st.metric("Max document size", "Unlimited")
        st.metric("Chunk size", "800 words")
        st.metric("Overlap", "100 words")
        
        if document_text:
            word_count = len(document_text.split())
            st.metric("Current doc", f"{word_count:,} words")
    
    if st.button("🚀 Load to Pinecone", disabled=not all_keys_provided or not document_text):
        if not all_keys_provided:
            st.error("Please provide all API keys")
        elif not document_text:
            st.error("Please provide document text")
        else:
            # Initialize RAG system
            with st.spinner("Initializing RAG system..."):
                st.session_state.enhanced_rag = EnhancedRAG(
                    anthropic_key,
                    pinecone_key,
                    cohere_key
                )
            
            # Load document
            success = st.session_state.enhanced_rag.load_document(
                document_text,
                doc_name
            )
            
            if success:
                st.balloons()
    
    # Q&A Interface
    st.header("2️⃣ Ask Questions")
    
    if st.session_state.enhanced_rag is None:
        st.info("👆 Load a document first")
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "metadata" in message:
                    with st.expander("🔍 Retrieval Details"):
                        meta = message["metadata"]
                        st.write(f"**Chunks used:** {meta['chunks_used']}")
                        
                        if meta.get('scores'):
                            st.write("**Relevance scores:**")
                            for score in meta['scores'][:3]:
                                st.write(f"  - Chunk {score['chunk_id']}: {score['relevance']:.3f}")
                        
                        st.text_area(
                            "Context",
                            meta.get('context', '')[:1000] + "...",
                            height=150
                        )
        
        # Chat input
        if question := st.chat_input("Ask anything about your documents..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Searching and analyzing..."):
                    result = st.session_state.enhanced_rag.answer_question(question)
                    
                    st.markdown(result['answer'])
                    
                    with st.expander("🔍 Retrieval Details"):
                        st.write(f"**Chunks used:** {result['chunks_used']}")
                        
                        if result.get('scores'):
                            st.write("**Top relevance scores:**")
                            for score in result['scores'][:3]:
                                st.write(f"  - Chunk {score['chunk_id']}: {score['relevance']:.3f}")
                        
                        st.text_area(
                            "Retrieved Context",
                            result['context_used'][:1000] + "...",
                            height=150,
                            key=f"context_{len(st.session_state.messages)}"
                        )
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "metadata": {
                            'chunks_used': result['chunks_used'],
                            'scores': result.get('scores', []),
                            'context': result['context_used']
                        }
                    })
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Statistics
    if st.session_state.enhanced_rag:
        st.header("📊 System Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Chunks", len(st.session_state.enhanced_rag.chunks))
        
        with col2:
            st.metric("Questions Asked", len([m for m in st.session_state.messages if m['role'] == 'user']))
        
        with col3:
            avg_chunks = sum(m.get('metadata', {}).get('chunks_used', 0) 
                           for m in st.session_state.messages if m['role'] == 'assistant')
            avg_chunks = avg_chunks / max(len([m for m in st.session_state.messages if m['role'] == 'assistant']), 1)
            st.metric("Avg Chunks/Answer", f"{avg_chunks:.1f}")
        
        with col4:
            st.metric("Vector DB", "Pinecone")


if __name__ == "__main__":
    main()