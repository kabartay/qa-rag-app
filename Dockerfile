FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

COPY pyproject.toml Makefile ./
COPY apps/ apps/
COPY src/ src/

RUN uv pip install --system -e ".[all]"

EXPOSE 8501 8502 8503 8504

CMD ["make", "run-simple"]
