"""
evaluate_retrieval.py — Run ResolveX retrieval evaluation.
"""

import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BACKEND_ROOT))

from ai.rag.retrieval_evaluator import RetrievalEvaluator


QUERIES_PATH = (
    BACKEND_ROOT
    / "data"
    / "evaluation"
    / "retrieval_queries.json"
)

OUTPUT_PATH = (
    BACKEND_ROOT
    / "data"
    / "evaluation"
    / "retrieval_results.json"
)


def main():
    # --------------------------------------------------------------
    # Load evaluation dataset
    # --------------------------------------------------------------

    with open(
        QUERIES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        queries = json.load(file)

    # --------------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------------

    evaluator = RetrievalEvaluator()

    results = evaluator.evaluate_all(
        evaluation_queries=queries,
    )

    # --------------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------------
    # Print summary
    # --------------------------------------------------------------

    print()

    print("=" * 110)
    print("RESOLVEX RETRIEVAL EVALUATION")
    print("=" * 110)

    print(
        f"Evaluation queries: {len(queries)}"
    )

    print()

    print(
        f"{'Retriever':<20}"
        f"{'Recall@5':>12}"
        f"{'Recall@10':>12}"
        f"{'MRR':>12}"
        f"{'Precision@5':>15}"
        f"{'Precision@10':>16}"
        f"{'nDCG@5':>12}"
        f"{'nDCG@10':>12}"
    )

    print("-" * 110)

    for name in [
        "bm25",
        "dense",
        "hybrid",
        "hybrid_reranker",
    ]:
        metrics = results[name]

        print(
            f"{name:<20}"
            f"{metrics['recall@5']:>12.4f}"
            f"{metrics['recall@10']:>12.4f}"
            f"{metrics['mrr']:>12.4f}"
            f"{metrics['precision@5']:>15.4f}"
            f"{metrics['precision@10']:>16.4f}"
            f"{metrics['ndcg@5']:>12.4f}"
            f"{metrics['ndcg@10']:>12.4f}"
        )

    print("-" * 110)

    print()

    print("Detailed results saved to:")
    print(OUTPUT_PATH)

    print("=" * 110)


if __name__ == "__main__":
    main()