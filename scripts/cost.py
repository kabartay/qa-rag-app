#!/usr/bin/env python3

"""
Cost Calculator for QA RAG Application.
Calculates actual costs based on API usage.
Pricing is up-to-date (as of 2025).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict, cast


@dataclass
class PricingModel:
    """API pricing per 1M tokens/requests"""

    name: str
    input_cost: float | None = None  # $ per 1M input tokens
    output_cost: float | None = None  # $ per 1M output tokens
    embedding_cost: float | None = None  # $ per 1M tokens
    rerank_cost: float | None = None  # $ per 1K searches


class PineconeStorage(TypedDict):
    serverless: float
    read: float
    write: float


PRICING: dict[str, PricingModel | PineconeStorage] = {
    "claude_sonnet_4": PricingModel(
        name="Claude Sonnet 4",
        input_cost=3.00,  # $3 per 1M input tokens
        output_cost=15.00,  # $15 per 1M output tokens
    ),
    "pinecone_embeddings": PricingModel(
        name="Pinecone", embedding_cost=0.08  # $0.08 per 1M tokens
    ),
    "cohere_embeddings": PricingModel(
        name="Cohere Embeddings", embedding_cost=0.12  # $0.12 per 1M tokens
    ),
    "cohere_rerank": PricingModel(name="Cohere Rerank", rerank_cost=2.00),  # $2 per 1K searches
    "pinecone_storage": {
        "serverless": 0.001,  # $0.001 per 1K vectors per month
        "read": 16.00,  # $16 per 1M read units
        "write": 4.00,  # $4 per 1M write units
    },
}


class RAGCostCalculator:
    """Calculate costs for RAG operations"""

    def __init__(self) -> None:
        self.pricing = PRICING

    def estimate_tokens(self, text: str, multiplier: float = 0.75) -> int:
        """Rough token estimation (1 token ≈ 4 chars)"""
        return int(len(text) * multiplier / 4)

    def calculate_simple_rag(
        self,
        num_queries: int = 1000,
        avg_document_tokens: int = 20000,
        avg_context_tokens: int = 3000,
        avg_output_tokens: int = 300,
    ) -> dict[str, Any]:
        """Calculate Simple RAG costs"""

        claude = cast(PricingModel, self.pricing["claude_sonnet_4"])

        avg_query_tokens = 50
        total_input_tokens = num_queries * (avg_context_tokens + avg_query_tokens)
        total_output_tokens = num_queries * avg_output_tokens

        claude_input_cost = (total_input_tokens / 1_000_000) * (claude.input_cost or 0)
        claude_output_cost = (total_output_tokens / 1_000_000) * (claude.output_cost or 0)

        total_cost = claude_input_cost + claude_output_cost

        return {
            "system": "Simple RAG",
            "queries": num_queries,
            "breakdown": {
                "claude_input": f"${claude_input_cost:.2f}",
                "claude_output": f"${claude_output_cost:.2f}",
            },
            "total_cost": f"${total_cost:.2f}",
            "cost_per_query": f"${total_cost / num_queries:.4f}",
            "tokens_used": {
                "input": f"{total_input_tokens:,}",
                "output": f"{total_output_tokens:,}",
            },
        }

    def calculate_enhanced_rag(
        self,
        num_queries: int = 1000,
        num_documents: int = 1,
        avg_chunks_per_doc: int = 100,
        avg_chunk_tokens: int = 300,
        retrieval_top_k: int = 20,
        rerank_top_n: int = 5,
        avg_output_tokens: int = 300,
    ) -> dict[str, Any]:
        """Calculate Enhanced RAG costs (Pinecone + Cohere + Claude)"""

        cohere_embeddings = cast(PricingModel, self.pricing["cohere_embeddings"])
        cohere_rerank = cast(PricingModel, self.pricing["cohere_rerank"])
        claude = cast(PricingModel, self.pricing["claude_sonnet_4"])
        pinecone_storage = cast(PineconeStorage, self.pricing["pinecone_storage"])

        # 1. Embeddings (one-time per document)
        total_embedding_tokens = num_documents * avg_chunks_per_doc * avg_chunk_tokens
        embedding_cost = (total_embedding_tokens / 1_000_000) * (
            cohere_embeddings.embedding_cost or 0
        )

        # 2. Pinecone storage (monthly)
        total_vectors = num_documents * avg_chunks_per_doc
        pinecone_storage_monthly = (total_vectors / 1000) * pinecone_storage["serverless"]

        # 3. Pinecone reads (per query)
        pinecone_read_cost = (num_queries / 1_000_000) * pinecone_storage["read"]

        # 4. Cohere rerank
        rerank_cost = (num_queries / 1000) * (cohere_rerank.rerank_cost or 0)

        # 5. Claude generation
        avg_query_tokens = 50
        context_tokens = rerank_top_n * avg_chunk_tokens
        total_input_tokens = num_queries * (context_tokens + avg_query_tokens)
        total_output_tokens = num_queries * avg_output_tokens

        claude_input_cost = (total_input_tokens / 1_000_000) * (claude.input_cost or 0)
        claude_output_cost = (total_output_tokens / 1_000_000) * (claude.output_cost or 0)

        total_cost = (
            embedding_cost
            + pinecone_storage_monthly
            + pinecone_read_cost
            + rerank_cost
            + claude_input_cost
            + claude_output_cost
        )

        per_query_cost = total_cost / num_queries

        return {
            "system": "Enhanced RAG",
            "queries": num_queries,
            "documents": num_documents,
            "total_vectors": total_vectors,
            "breakdown": {
                "embeddings_onetime": f"${embedding_cost:.2f}",
                "pinecone_storage_monthly": f"${pinecone_storage_monthly:.4f}",
                "pinecone_reads": f"${pinecone_read_cost:.2f}",
                "cohere_rerank": f"${rerank_cost:.2f}",
                "claude_input": f"${claude_input_cost:.2f}",
                "claude_output": f"${claude_output_cost:.2f}",
            },
            "total_cost": f"${total_cost:.2f}",
            "cost_per_query": f"${per_query_cost:.4f}",
            "tokens_used": {
                "embedding": f"{total_embedding_tokens:,}",
                "input": f"{total_input_tokens:,}",
                "output": f"{total_output_tokens:,}",
            },
        }

    def compare_systems(self, num_queries: int = 1000) -> dict[str, Any]:
        """Compare Simple vs Enhanced RAG costs"""

        simple = self.calculate_simple_rag(num_queries)
        enhanced = self.calculate_enhanced_rag(num_queries)

        simple_total = float(simple["total_cost"].replace("$", ""))
        enhanced_total = float(enhanced["total_cost"].replace("$", ""))

        difference = enhanced_total - simple_total
        percent_diff = (difference / simple_total) * 100

        return {
            "queries": num_queries,
            "simple_rag": simple,
            "enhanced_rag": enhanced,
            "comparison": {
                "difference": f"${difference:.2f}",
                "percent_more": f"{percent_diff:.1f}%",
                "recommendation": self._get_recommendation(simple_total, enhanced_total),
            },
        }

    def _get_recommendation(self, simple_cost: float, enhanced_cost: float) -> str:
        """Provide recommendation based on cost comparison"""
        if enhanced_cost < simple_cost * 1.5:
            return "Enhanced RAG worth it for better accuracy"
        elif enhanced_cost < simple_cost * 2:
            return "Enhanced RAG reasonable if quality is critical"
        else:
            return "Simple RAG more cost-effective for basic use cases"


def print_report(data: dict[str, Any]) -> None:
    """Pretty print cost report"""
    print("\n" + "=" * 70)
    print(f"  {data.get('system', 'RAG System')} - Cost Analysis")
    print("=" * 70)
    print(f"\nQueries: {data['queries']:,}")

    if "documents" in data:
        print(f"Documents: {data['documents']}")
        print(f"Total Vectors: {data['total_vectors']:,}")

    print("\nCost Breakdown:")
    for component, cost in data["breakdown"].items():
        print(f"   • {component:.<40} {cost:>12}")

    print(f"\n{'─' * 70}")
    print(f"   Total Cost {' ' * 30} {data['total_cost']:>12}")
    print(f"   Cost per Query {' ' * 27} {data['cost_per_query']:>12}")
    print("=" * 70)


def main() -> None:
    """Main calculator interface"""
    calc = RAGCostCalculator()

    print("\n" + "x" * 35)
    print("  RAG System Cost Calculator")
    print("x" * 35)

    try:
        num_queries = int(input("\nNumber of queries to estimate (default 1000): ") or "1000")
    except ValueError:
        num_queries = 1000

    print("\nCalculating costs...\n")

    simple_result = calc.calculate_simple_rag(num_queries)
    enhanced_result = calc.calculate_enhanced_rag(num_queries)

    print_report(simple_result)
    print_report(enhanced_result)

    comparison = calc.compare_systems(num_queries)["comparison"]

    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"\n   Enhanced RAG costs {comparison['difference']} more")
    print(f"   ({comparison['percent_more']} increase)")
    print(f"\n   💡 {comparison['recommendation']}")
    print("=" * 70 + "\n")

    export = input("Export results to JSON? (y/n): ").lower()
    if export == "y":
        results = {
            "simple_rag": simple_result,
            "enhanced_rag": enhanced_result,
            "comparison": comparison,
        }
        with open("cost.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Exported to cost.json\n")


if __name__ == "__main__":
    main()
