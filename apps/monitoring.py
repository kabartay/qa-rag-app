"""
Monitoring utilities for RAG applications.
Supports:
- LangSmith tracing (if LANGSMITH_API_KEY is set)
- Prometheus metrics (local export on port 9100)
"""

import logging
import os
import time
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith (tracing)
try:
    from langsmith import Client
    from langsmith.run_helpers import traceable
except ImportError:
    Client = None  # fallback type

    def traceable(
        *_args: Any, **_kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """No-op fallback decorator."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


# Prometheus (metrics)
try:
    from prometheus_client import Counter, Histogram, start_http_server
except ImportError:
    Counter = None
    Histogram = None
    start_http_server = None

# LangSmith setup
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-evaluation")

if LANGSMITH_API_KEY and Client:
    client = Client(api_key=LANGSMITH_API_KEY)
    LANGSMITH_ENABLED = True
else:
    client = None
    LANGSMITH_ENABLED = False

# Prometheus setup
PROM_PORT = int(os.getenv("PROMETHEUS_PORT", "9100"))
PROM_ENABLED = os.getenv("ENABLE_PROMETHEUS", "true").lower() in ("true", "1", "yes")

if PROM_ENABLED and start_http_server:
    try:
        bind_addr = os.getenv("PROMETHEUS_BIND_ADDR", "127.0.0.1")
        start_http_server(PROM_PORT, addr=bind_addr)
        logging.info(f"Prometheus metrics exporter started on {bind_addr}:{PROM_PORT}")
    except OSError:
        logging.warning("Prometheus metrics server already running or failed to start")

# Prometheus metrics (safe defaults)
if PROM_ENABLED and Counter and Histogram:
    RAG_REQUESTS = Counter("rag_requests_total", "Total number of RAG queries handled")
    CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits from Redis")
    CACHE_MISSES = Counter("rag_cache_misses_total", "Cache misses from Redis")
    RESPONSE_TIME = Histogram("rag_response_seconds", "Time to answer a RAG query (seconds)")
else:
    RAG_REQUESTS = None
    CACHE_HITS = None
    CACHE_MISSES = None
    RESPONSE_TIME = None


# Helper decorators
def traced(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator combining LangSmith tracing and Prometheus metrics.
    Works safely even if monitoring is disabled.
    """
    wrapped_func = func
    if traceable and LANGSMITH_ENABLED:
        wrapped_func = traceable(name=func.__name__)(func)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if RAG_REQUESTS:
            RAG_REQUESTS.inc()

        start_time = time.time()
        try:
            return wrapped_func(*args, **kwargs)
        finally:
            if RESPONSE_TIME:
                RESPONSE_TIME.observe(time.time() - start_time)

    return wrapper


def log_cache_event(hit: bool) -> None:
    """Increment cache hit/miss counters safely."""
    if hit and CACHE_HITS:
        CACHE_HITS.inc()
    elif not hit and CACHE_MISSES:
        CACHE_MISSES.inc()


def get_monitoring_status() -> dict[str, Any]:
    """Return current monitoring configuration."""
    return {
        "langsmith_enabled": LANGSMITH_ENABLED,
        "prometheus_enabled": PROM_ENABLED,
        "project": LANGSMITH_PROJECT if LANGSMITH_ENABLED else None,
        "prometheus_port": PROM_PORT if PROM_ENABLED else None,
    }
