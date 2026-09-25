import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ai.rag.hybrid_retriever import HybridRetriever
from ai.rag.reranker import get_reranker


def main():
    query = "I cannot access my email account"

    hybrid = HybridRetriever(
        bm25_weight=0.0,
        dense_weight=1.0,
    )

    candidates = hybrid.retrieve(
        query=query,
        top_k=20,
        candidate_k=20,
    )

    print("\n" + "=" * 80)
    print("HYBRID CANDIDATES")
    print("=" * 80)

    for rank, candidate in enumerate(candidates, start=1):
        print(
            f"{rank:2d}. "
            f"index_id={candidate.index_id:<4} "
            f"score={candidate.score:.4f} "
            f"retriever={candidate.retriever}"
        )

    reranker = get_reranker()

    reranked = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for rank, candidate in enumerate(reranked, start=1):
        print(
            f"{rank:2d}. "
            f"index_id={candidate.index_id:<4} "
            f"score={candidate.score:.4f} "
            f"retriever={candidate.retriever}"
        )

    assert len(reranked) <= 5
    assert all(
        candidate.retriever == "reranker"
        for candidate in reranked
    )

    print("\nReranker smoke test passed.")


if __name__ == "__main__":
    main()