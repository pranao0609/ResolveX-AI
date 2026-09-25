"""
evaluate_hybrid_weights.py — Hybrid retrieval weight sweep.

Evaluates BM25 + dense retrieval across multiple fusion weights
using the existing retrieval benchmark.

The experiment keeps the retrieval pipeline unchanged and only
varies the BM25/dense fusion weights.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import json
from typing import Dict, List

from ai.rag.retrieval_evaluator import RetrievalEvaluator
from ai.rag.hybrid_retriever import HybridRetriever


BASE_DIR = Path(__file__).resolve().parents[1]

QUERIES_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "retrieval_queries.json"
)

RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "hybrid_weight_results.json"
)


# BM25 weight values to test.
# Dense weight is automatically:
#
#     dense_weight = 1.0 - bm25_weight
#
BM25_WEIGHTS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


def load_queries() -> List[Dict]:
    """Load retrieval evaluation queries."""

    if not QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval query file not found: {QUERIES_PATH}"
        )

    with open(
        QUERIES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "retrieval_queries.json must contain a list"
        )

    return data


def evaluate_weight(
    queries: List[Dict],
    bm25_weight: float,
) -> Dict:
    dense_weight = 1.0 - bm25_weight

    print()
    print("=" * 70)
    print(
        f"Testing weights: "
        f"BM25={bm25_weight:.1f} | "
        f"Dense={dense_weight:.1f}"
    )
    print("=" * 70)

    hybrid_retriever = HybridRetriever(
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )

    evaluator = RetrievalEvaluator(
        hybrid_retriever=hybrid_retriever,
    )

    results = evaluator.evaluate_retriever(
        retriever_name="hybrid",
        evaluation_queries=queries,
    )

    return {
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "recall_at_5": results["recall@5"],
        "recall_at_10": results["recall@10"],
        "mrr": results["mrr"],
        "precision_at_5": results["precision@5"],
        "precision_at_10": results["precision@10"],
    }


def print_summary(results: List[Dict]) -> None:
    """Print experiment summary."""

    print()
    print("=" * 90)
    print("HYBRID RETRIEVAL WEIGHT SWEEP")
    print("=" * 90)

    print(
        f"{'BM25':>6} "
        f"{'Dense':>7} "
        f"{'Recall@5':>10} "
        f"{'Recall@10':>11} "
        f"{'MRR':>10} "
        f"{'Precision@5':>13} "
        f"{'Precision@10':>14}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['bm25_weight']:>6.1f} "
            f"{result['dense_weight']:>7.1f} "
            f"{result['recall_at_5']:>10.4f} "
            f"{result['recall_at_10']:>11.4f} "
            f"{result['mrr']:>10.4f} "
            f"{result['precision_at_5']:>13.4f} "
            f"{result['precision_at_10']:>14.4f}"
        )

    print("=" * 90)


def save_results(results: List[Dict]) -> None:
    """Save weight sweep results."""

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment": "hybrid_weight_sweep",
        "bm25_weights": BM25_WEIGHTS,
        "query_count": len(load_queries()),
        "results": results,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print()
    print(
        f"Results saved to: {RESULTS_PATH}"
    )


def main() -> None:
    queries = load_queries()

    print("=" * 70)
    print("RESOLVEX HYBRID RETRIEVAL WEIGHT SWEEP")
    print("=" * 70)
    print(f"Evaluation queries: {len(queries)}")
    print(f"Configurations: {len(BM25_WEIGHTS)}")
    print()

    results = []

    for bm25_weight in BM25_WEIGHTS:
        result = evaluate_weight(
            queries=queries,
            bm25_weight=bm25_weight,
        )

        results.append(result)

    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()