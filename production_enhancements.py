"""
Production-ready components for Enhanced RAG
"""

import PyPDF2
import pdfplumber
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
import hashlib


# ============================================
# 1. PRODUCTION-GRADE PDF PROCESSING
# ============================================

class PDFProcessor:
    """Extract text from PDFs with table and metadata support"""
    
    @staticmethod
    def extract_text_pypdf2(pdf_path: str) -> str:
        """Basic PDF extraction - fast but limited"""
        text = []
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text.append(page.extract_text())
        return '\n\n'.join(text)
    
    @staticmethod
    def extract_with_structure(pdf_path: str) -> Dict:
        """
        Advanced extraction with pdfplumber
        Preserves tables, headers, and page boundaries
        """
        result = {
            'text': [],
            'tables': [],
            'metadata': {},
            'pages': []
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            # Extract metadata
            result['metadata'] = {
                'num_pages': len(pdf.pages),
                'info': pdf.metadata
            }
            
            for i, page in enumerate(pdf.pages):
                page_data = {
                    'page_num': i + 1,
                    'text': page.extract_text() or '',
                    'tables': []
                }
                
                # Extract tables separately
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        page_data['tables'].append(table)
                        result['tables'].append({
                            'page': i + 1,
                            'data': table
                        })
                
                result['pages'].append(page_data)
                result['text'].append(page_data['text'])
        
        result['full_text'] = '\n\n'.join(result['text'])
        return result
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)   # Fix punctuation spacing
        
        return text.strip()


# ============================================
# 2. REAL VOYAGE AI EMBEDDINGS
# ============================================

class VoyageEmbeddings:
    """
    Production implementation with Voyage AI
    (Replace the mock embeddings in enhanced RAG)
    """
    
    def __init__(self, api_key: str):
        """
        Install: pip install voyageai
        Sign up: https://www.voyageai.com
        """
        try:
            import voyageai
            self.client = voyageai.Client(api_key=api_key)
            self.model = "voyage-2"  # or "voyage-large-2" for better quality
        except ImportError:
            raise ImportError("Install voyageai: pip install voyageai")
    
    def embed_documents(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """
        Embed documents in batches
        Cost: ~$0.10 per 1M tokens
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Voyage AI handles batching automatically
            result = self.client.embed(
                batch,
                model=self.model,
                input_type="document"  # Optimize for documents
            )
            
            all_embeddings.extend(result.embeddings)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed query - uses different optimization
        """
        result = self.client.embed(
            [query],
            model=self.model,
            input_type="query"  # Optimize for queries
        )
        return result.embeddings[0]


# Alternative: OpenAI Embeddings
class OpenAIEmbeddings:
    """Alternative to Voyage AI"""
    
    def __init__(self, api_key: str):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "text-embedding-3-large"  # 3072 dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return response.data[0].embedding


# ============================================
# 3. SMART CHUNKING STRATEGIES
# ============================================

@dataclass
class Chunk:
    """Structured chunk with metadata"""
    id: str
    text: str
    start_char: int
    end_char: int
    page_num: Optional[int] = None
    section_title: Optional[str] = None
    chunk_type: str = "text"  # text, table, header
    word_count: int = 0


class SmartChunker:
    """Advanced chunking with multiple strategies"""
    
    @staticmethod
    def semantic_chunking(text: str, max_chunk_size: int = 800) -> List[Chunk]:
        """
        Chunk by semantic boundaries (paragraphs, sections)
        Better than fixed-size for coherent retrieval
        """
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        start_char = 0
        
        for para in paragraphs:
            para_words = para.split()
            para_size = len(para_words)
            
            if current_size + para_size > max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk = Chunk(
                    id=hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                    text=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    word_count=current_size
                )
                chunks.append(chunk)
                
                # Start new chunk
                start_char = start_char + len(chunk_text)
                current_chunk = para_words
                current_size = para_size
            else:
                current_chunk.extend(para_words)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                id=hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                text=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                word_count=current_size
            ))
        
        return chunks
    
    @staticmethod
    def hierarchical_chunking(text: str) -> List[Chunk]:
        """
        Create chunks at multiple levels:
        - Section headers
        - Paragraphs under sections
        - Sentences within paragraphs
        
        Useful for: Technical documents with clear structure
        """
        chunks = []
        
        # Detect section headers (e.g., "1. Introduction", "1.1 Background")
        section_pattern = r'^[\d\.]+\s+[A-Z]'
        
        lines = text.split('\n')
        current_section = None
        current_text = []
        
        for line in lines:
            if re.match(section_pattern, line):
                # Save previous section
                if current_text:
                    chunks.append(Chunk(
                        id=f"section_{len(chunks)}",
                        text=' '.join(current_text),
                        start_char=0,
                        end_char=0,
                        section_title=current_section,
                        word_count=len(current_text)
                    ))
                
                # Start new section
                current_section = line.strip()
                current_text = [line]
            else:
                current_text.append(line)
        
        # Add final section
        if current_text:
            chunks.append(Chunk(
                id=f"section_{len(chunks)}",
                text=' '.join(current_text),
                start_char=0,
                end_char=0,
                section_title=current_section,
                word_count=len(current_text)
            ))
        
        return chunks


# ============================================
# 4. CACHING LAYER (REDIS)
# ============================================

class ResponseCache:
    """Cache RAG responses to reduce API costs"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Install: pip install redis
        Run Redis: docker run -p 6379:6379 redis
        """
        try:
            import redis
            self.redis = redis.from_url(redis_url)
            self.ttl = 3600 * 24 * 7  # 7 days
        except ImportError:
            raise ImportError("Install redis: pip install redis")
    
    def _make_key(self, question: str, doc_id: str) -> str:
        """Create cache key from question and document"""
        content = f"{doc_id}:{question}".encode()
        return f"rag:{hashlib.md5(content).hexdigest()}"
    
    def get(self, question: str, doc_id: str) -> Optional[Dict]:
        """Get cached response"""
        key = self._make_key(question, doc_id)
        cached = self.redis.get(key)
        
        if cached:
            import json
            return json.loads(cached)
        return None
    
    def set(self, question: str, doc_id: str, response: Dict):
        """Cache response"""
        import json
        key = self._make_key(question, doc_id)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(response)
        )
    
    def clear(self, pattern: str = "rag:*"):
        """Clear cache"""
        for key in self.redis.scan_iter(pattern):
            self.redis.delete(key)


# ============================================
# 5. MONITORING & LOGGING
# ============================================

class RAGMonitor:
    """Track RAG performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'total_latency': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
    
    def log_query(self, query: str, latency: float, cached: bool, 
                  chunks_used: int, cost: float):
        """Log query metrics"""
        self.metrics['total_queries'] += 1
        self.metrics['total_latency'] += latency
        
        if cached:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
        
        # In production, send to monitoring service:
        # - Datadog
        # - Prometheus
        # - Langsmith
        # - Custom dashboard
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        total = self.metrics['total_queries']
        if total == 0:
            return {}
        
        return {
            'total_queries': total,
            'avg_latency': self.metrics['total_latency'] / total,
            'cache_hit_rate': self.metrics['cache_hits'] / total,
            'error_rate': self.metrics['errors'] / total
        }


# ============================================
# 6. USAGE EXAMPLE
# ============================================

def production_rag_example():
    """
    Complete production RAG setup
    """
    
    # 1. Process PDF
    pdf_processor = PDFProcessor()
    doc_data = pdf_processor.extract_with_structure("document.pdf")
    text = pdf_processor.clean_text(doc_data['full_text'])
    
    # 2. Smart chunking
    chunker = SmartChunker()
    chunks = chunker.semantic_chunking(text, max_chunk_size=800)
    
    # 3. Generate embeddings (replace mock in enhanced RAG)
    embedder = VoyageEmbeddings(api_key="your_key")
    embeddings = embedder.embed_documents([c.text for c in chunks])
    
    # 4. Upload to Pinecone (same as before)
    # pinecone_index.upsert(...)
    
    # 5. Set up caching
    cache = ResponseCache()
    
    # 6. Query with cache
    question = "What are the energy requirements?"
    cached_response = cache.get(question, doc_id="doc_1")
    
    if cached_response:
        print("Cache hit!")
        return cached_response
    else:
        # Run full RAG pipeline
        # response = rag_system.answer(question)
        # cache.set(question, "doc_1", response)
        pass
    
    # 7. Monitor
    monitor = RAGMonitor()
    monitor.log_query(
        query=question,
        latency=2.5,
        cached=False,
        chunks_used=5,
        cost=0.02
    )
    
    print(monitor.get_stats())


# ============================================
# 7. INTEGRATION WITH ENHANCED RAG
# ============================================

"""
To integrate into your Enhanced RAG app:

1. Replace mock embeddings in EnhancedRAG.__init__():
   
   self.embedder = VoyageEmbeddings(voyage_api_key)

2. Replace _get_embeddings_voyage() method:
   
   def _get_embeddings_voyage(self, texts: List[str]):
       return self.embedder.embed_documents(texts)

3. Add PDF processing to load_document():
   
   if file_path.endswith('.pdf'):
       processor = PDFProcessor()
       doc_data = processor.extract_with_structure(file_path)
       text = doc_data['full_text']
   
4. Use smart chunking:
   
   chunker = SmartChunker()
   self.chunks = chunker.semantic_chunking(text)

5. Add caching to answer_question():
   
   cached = self.cache.get(question, doc_id)
   if cached:
       return cached
   
   # ... normal RAG pipeline ...
   
   self.cache.set(question, doc_id, result)

6. Add monitoring:
   
   self.monitor.log_query(...)
"""


if __name__ == "__main__":
    print("Production-Ready RAG Components")
    print("=" * 60)
    print("\n Available enhancements:")
    print("  1. PDF Processing (PyPDF2 + pdfplumber)")
    print("  2. Voyage AI Embeddings (production-grade)")
    print("  3. Smart Chunking (semantic + hierarchical)")
    print("  4. Redis Caching (reduce costs)")
    print("  5. Monitoring & Logging")
    print("\n💡 Copy these components into your Enhanced RAG app!")
    print("=" * 60)