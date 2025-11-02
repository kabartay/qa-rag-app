.PHONY: help install install-dev install-all setup run-simple run-enhanced run-evaluate run-generate test lint format type-check clean clean-all docker-build docker-run cost-estimate docs

.DEFAULT_GOAL := help

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)RAG Evaluation System$(NC)"
	@echo "$(GREEN)=====================$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation

check-uv: ## Check if uv is installed
	@which uv > /dev/null || (echo "$(RED)uv not found. Installing...$(NC)" && curl -LsSf https://astral.sh/uv/install.sh | sh)

setup: check-uv ## Initial setup - install uv and create .env
	@echo "$(GREEN)Setting up RAG Evaluation System...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file...$(NC)"; \
		cp .env.example .env 2>/dev/null || echo "ANTHROPIC_API_KEY=\nPINECONE_API_KEY=\nCOHERE_API_KEY=\nVOYAGE_API_KEY=\nREDIS_URL=redis://localhost:6379" > .env; \
		echo "$(GREEN)✓ Created .env file. Please add your API keys!$(NC)"; \
	fi
	@echo "$(GREEN)✓ Setup complete!$(NC)"

venv: check-uv ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	@if [ ! -d .venv ]; then \
		uv venv; \
		echo "$(GREEN)✓ Virtual environment created in .venv/$(NC)"; \
		echo "$(YELLOW)To activate manually: source .venv/bin/activate$(NC)"; \
	else \
		echo "$(YELLOW)Virtual environment already exists$(NC)"; \
		echo "$(YELLOW)To activate manually: source .venv/bin/activate$(NC)"; \
	fi

activate: ## Show how to activate virtual environment
	@echo "$(YELLOW)To activate the virtual environment manually:$(NC)"
	@echo "  source .venv/bin/activate"
	@echo ""
	@echo "$(YELLOW)Or just use make commands which activate it automatically:$(NC)"
	@echo "  make run-simple"
	@echo "  make test"

install: check-uv venv ## Install production dependencies only
	@echo "$(GREEN)Installing production dependencies with uv...$(NC)"
	uv pip install -e .
	@echo "$(GREEN)✓ Production dependencies installed!$(NC)"

install-dev: check-uv venv ## Install with development dependencies
	@echo "$(GREEN)Installing with development dependencies...$(NC)"
	uv pip install -e ".[dev]"
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	. .venv/bin/activate && pre-commit install
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

install-all: check-uv venv ## Install all dependencies (dev + monitoring)
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	uv pip install -e ".[all]"
	@echo "$(GREEN)Downloading NLP models...$(NC)"
	. .venv/bin/activate && python -m spacy download en_core_web_sm
	. .venv/bin/activate && python -m nltk.downloader punkt averaged_perceptron_tagger
	@echo "$(GREEN)✓ Complete installation finished!$(NC)"

sync: check-uv ## Sync dependencies with pyproject.toml
	@echo "$(GREEN)Syncing dependencies...$(NC)"
	uv pip sync
	@echo "$(GREEN)✓ Dependencies synced!$(NC)"

##@ Running Applications

run-simple: ## Run Simple RAG application
	@echo "$(GREEN)Starting Simple RAG application...$(NC)"
	@. .venv/bin/activate && streamlit run apps/rag_app.py --server.port 8501

run-enhanced: ## Run Enhanced RAG with Pinecone + Cohere
	@echo "$(GREEN)Starting Enhanced RAG application...$(NC)"
	@. .venv/bin/activate && streamlit run apps/rag_app_enhanced.py --server.port 8502

run-evaluate: ## Run RAG Evaluator
	@echo "$(GREEN)Starting RAG Evaluator...$(NC)"
	@. .venv/bin/activate && streamlit run apps/rag_evaluator.py --server.port 8503

run-generate: ## Run Q&A Groundtruth Generator
	@echo "$(GREEN)Starting Q&A Generator...$(NC)"
	@. .venv/bin/activate && streamlit run apps/qa_generator.py --server.port 8504

run-all: ## Run all applications on different ports
	@echo "$(GREEN)Starting all applications...$(NC)"
	@echo "$(YELLOW)Simple RAG:      		http://localhost:8501$(NC)"
	@echo "$(YELLOW)Enhanced RAG: 		    http://localhost:8502$(NC)"
	@echo "$(YELLOW)Evaluator:       		http://localhost:8503$(NC)"
	@echo "$(YELLOW)Q&A Generator:          http://localhost:8504$(NC)"
	@trap 'kill 0' EXIT; \
	. .venv/bin/activate && streamlit run apps/rag_app.py --server.port 8501 & \
	. .venv/bin/activate && streamlit run apps/rag_app_enhanced.py --server.port 8502 & \
	. .venv/bin/activate && streamlit run apps/rag_evaluator.py --server.port 8503 & \
	. .venv/bin/activate && streamlit run apps/qa_generator.py --server.port 8504 & \
	wait

##@ Testing & Quality

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	@. .venv/bin/activate && pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@. .venv/bin/activate && pytest tests/ -v --cov=apps --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/index.html$(NC)"

lint: ## Run linter (ruff)
	@echo "$(GREEN)Running linter...$(NC)"
	@. .venv/bin/activate && ruff check tests/ apps/

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	@. .venv/bin/activate && black tests/ apps/
	@. .venv/bin/activate && ruff check --fix tests/ apps/

type-check: ## Run type checker (mypy)
	@echo "$(GREEN)Running type checker...$(NC)"
	@. .venv/bin/activate && mypy src/

check: lint type-check test ## Run all checks (lint, type, test)
	@echo "$(GREEN)✓ All checks passed!$(NC)"

##@ Utilities

cost-estimate: ## Run cost estimation calculator
	@echo "$(GREEN)Running cost calculator...$(NC)"
	@. .venv/bin/activate && python scripts/cost_calculator.py

download-models: ## Download NLP models
	@echo "$(GREEN)Downloading NLP models...$(NC)"
	@. .venv/bin/activate && python -m spacy download en_core_web_sm
	@. .venv/bin/activate && python -m nltk.downloader punkt averaged_perceptron_tagger
	@echo "$(GREEN)✓ Models downloaded!$(NC)"

redis-start: ## Start Redis with Docker
	@echo "$(GREEN)Starting Redis...$(NC)"
	docker run -d -p 6379:6379 --name rag-redis redis:7-alpine
	@echo "$(GREEN)✓ Redis running on localhost:6379$(NC)"

redis-stop: ## Stop Redis container
	@echo "$(YELLOW)Stopping Redis...$(NC)"
	docker stop rag-redis && docker rm rag-redis

##@ Documentation

docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	@mkdir -p docs/build
	@echo "# RAG Evaluation System Documentation" > docs/build/index.md
	@echo "$(GREEN)✓ Documentation generated in docs/build/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation on http://localhost:8000$(NC)"
	python -m http.server 8000 -d docs/build

##@ Docker

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t qa-rag-app:latest .

docker-run: ## Run application in Docker
	@echo "$(GREEN)Running application in Docker...$(NC)"
	docker run -p 8501:8501 -p 8502:8502 \
		--env-file .env \
		qa-rag-app:latest

docker-compose-up: ## Start all services with docker-compose
	@echo "$(GREEN)Starting services with docker-compose...$(NC)"
	docker-compose up -d

docker-compose-down: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose down

##@ Cleaning

clean: ## Clean cache and temporary files
	@echo "$(YELLOW)Cleaning cache files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "$(GREEN)✓ Cleaned!$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(RED)Cleaning everything including venv...$(NC)"
	rm -rf .venv/
	rm -rf dist/
	rm -rf build/
	@echo "$(GREEN)✓ Deep clean complete!$(NC)"
	@echo "$(YELLOW)Run 'make install' to recreate environment$(NC)"

##@ Information

info: ## Show project information
	@echo "$(BLUE)RAG Evaluation System$(NC)"
	@echo "$(GREEN)=====================$(NC)"
	@echo "Python version:  $(python --version)"
	@echo "uv version:      $(uv --version 2>/dev/null || echo 'not installed')"
	@echo "Project root:    $(pwd)"
	@if [ -d .venv ]; then \
		echo "Virtual env:     ✓ Created (.venv/)"; \
	else \
		echo "Virtual env:     ✗ Not created (run 'make venv')"; \
	fi
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  1. make setup      - Initial setup"
	@echo "  2. make install    - Install dependencies"
	@echo "  3. make run-simple - Run application"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make dev           - Full dev setup"
	@echo "  make test          - Run tests"
	@echo "  make format        - Format code"
	@echo ""
	@echo "$(YELLOW)For more commands, run: make help$(NC)"

version: ## Show version information
	@echo "RAG Evaluation System v1.0.0"
	@python --version
	@uv --version 2>/dev/null || echo "uv: not installed"
