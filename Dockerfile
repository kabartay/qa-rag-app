# ---------- Base ----------
FROM python:3.11-slim

# ---------- Environment ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_PROMETHEUS=true \
    PROMETHEUS_PORT=9100 \
    PROMETHEUS_BIND_ADDR=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working directory ----------
WORKDIR /app

# ---------- Copy project files ----------
COPY pyproject.toml Makefile README.md ./
COPY apps/ apps/

# ---------- Install uv and dependencies ----------
RUN pip install --no-cache-dir uv

# Using uv to install all deps (you can replace with "uv pip install -e '.[all]'" if pyproject.toml has extras)
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
    redis \
    langsmith

# ---------- Ports ----------
EXPOSE 8501 9100

# ---------- Health check ----------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ---------- Entrypoint ----------
# No venv needed inside container (uv installed system-wide)
# CMD ["streamlit", "run", "apps/rag_app_enhanced.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["bash", "-c", "streamlit run ${APP_PATH:-apps/rag_app_enhanced.py} --server.port=${PORT:-8501} --server.address=0.0.0.0"]
