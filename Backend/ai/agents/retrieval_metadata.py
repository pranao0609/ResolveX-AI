from __future__ import annotations

from statistics import mean
from typing import Any


def _safe_float(value: Any) -> float | None:
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_statistics(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate lightweight retrieval score statistics.

    These values are descriptive retrieval signals, not calibrated
    probabilities or confidence scores.
    """

    scores: list[float] = []

    for document in documents:
        score = _safe_float(document.get("score"))

        if score is not None:
            scores.append(score)

    if not scores:
        return {
            "top_score": None,
            "second_score": None,
            "score_gap": None,
            "mean_score": None,
            "min_score": None,
            "max_score": None,
        }

    ordered_scores = sorted(scores, reverse=True)

    top_score = ordered_scores[0]

    second_score = ordered_scores[1] if len(ordered_scores) > 1 else None

    score_gap = round(top_score - second_score, 6) if second_score is not None else None

    return {
        "top_score": round(top_score, 6),
        "second_score": (round(second_score, 6) if second_score is not None else None),
        "score_gap": score_gap,
        "mean_score": round(mean(scores), 6),
        "min_score": round(min(scores), 6),
        "max_score": round(max(scores), 6),
    }


def build_retrieval_metadata(
    *,
    strategy: str,
    top_k: int,
    candidate_k: int,
    bm25_weight: float,
    dense_weight: float,
    candidate_count: int,
    documents: list[dict[str, Any]],
    status: str,
    query_length: int,
    fallback_used: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Build the canonical metadata contract for the Retrieval Agent.

    Status values:
        success
        empty
        failure

    Retrieval score signals are descriptive metrics only.
    They must not be interpreted as calibrated confidence values.
    """

    result_count = len(documents)

    statistics = _score_statistics(documents)

    index_ids = [
        int(document["index_id"])
        for document in documents
        if document.get("index_id") is not None
    ]

    retrievers = [
        str(document["retriever"])
        for document in documents
        if document.get("retriever")
    ]

    unique_retrievers = sorted(set(retrievers))

    metadata: dict[str, Any] = {
        "strategy": strategy,
        "top_k": int(top_k),
        "candidate_k": int(candidate_k),
        "bm25_weight": float(bm25_weight),
        "dense_weight": float(dense_weight),
        "candidate_count": int(candidate_count),
        "result_count": int(result_count),
        "query_length": int(query_length),
        "status": status,
        "empty_result": result_count == 0,
        "fallback_used": bool(fallback_used),
        "index_ids": index_ids,
        "retrievers": retrievers,
        "unique_retrievers": unique_retrievers,
        "source_count": len(unique_retrievers),
        "error": error,
        **statistics,
    }

    if status == "success":
        metadata["decision_reason"] = (
            "Relevant retrieval results were resolved "
            "from the configured knowledge-retrieval strategy."
        )

    elif status == "empty":
        metadata["decision_reason"] = (
            "The retrieval strategy completed without "
            "resolvable knowledge-base results."
        )

    elif status == "failure":
        metadata["decision_reason"] = (
            "The retrieval strategy failed and the agent "
            "returned a safe empty-context fallback."
        )

    else:
        metadata["decision_reason"] = "Retrieval completed with an unclassified status."

    return metadata
