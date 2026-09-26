import sys
from pathlib import Path

# Add Backend/ to Python's import path.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

from ai.rag.hybrid_retriever import HybridRetriever


def main():
    retriever = HybridRetriever(
        bm25_weight=0.5,
        dense_weight=0.5,
    )

    query = "I cannot access my email account"

    results = retriever.retrieve(
        query=query,
        top_k=5,
        candidate_k=20,
    )

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL SMOKE TEST")
    print("=" * 70)

    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    print("\nRanked Results:")
    print("-" * 70)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"index_id={result.index_id} | "
            f"score={result.score:.4f} | "
            f"retriever={result.retriever}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
