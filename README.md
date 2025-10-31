# 🚀 RAG Evaluation System

Production-ready RAG system with automated evaluation using Pinecone and Cohere.

## ⚡ Quick Start

```bash
# 1. Install uv (ultra-fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup project
make setup

# 3. Install dependencies
make install

# 4. Add API keys to .env
nano .env

# 5. Run application
make run-simple
```

## 📦 What's Included

- **Simple RAG**: Fast prototyping (<50 pages)
- **Enhanced RAG**: Production scale (Pinecone + Cohere)
- **Q&A Generator**: Auto-create evaluation datasets
- **Evaluator**: 4-metric quality assessment

## 🎯 Commands

```bash
make help              # Show all commands
make install           # Install dependencies
make run-enhanced      # Run production RAG
make test              # Run tests
make format            # Format code
```

## 🔑 API Keys

Get free API keys:
- Anthropic: https://console.anthropic.com
- Pinecone: https://www.pinecone.io
- Cohere: https://dashboard.cohere.com

## 📊 Architecture

```
Document → Chunks → Embeddings → Pinecone
                                     ↓
Query → Vector Search → Cohere Rerank → Claude → Answer
```

## 💰 Costs

Simple RAG: $15 per 1K queries
Enhanced RAG: $25 per 1K queries

## 📄 License

MIT
