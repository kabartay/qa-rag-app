FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv using pip (more reliable in Docker)
RUN pip install uv

# Copy application files
COPY pyproject.toml Makefile ./
COPY apps/ apps/
COPY production_enhancements.py ./

# Install dependencies with uv
# Install dependencies with uv
RUN uv pip install --system \
    streamlit \
    anthropic \
    cohere \
    pinecone \
    voyageai \
    PyPDF2 \
    pdfplumber \
    python-dotenv \
    pandas \
    numpy \
    redis

# Expose ports
EXPOSE 8501 8502 8503 8504

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Activate venv and run
CMD ["/bin/bash", "-c", "source .venv/bin/activate && streamlit run apps/rag_app.py --server.port=8501"]
