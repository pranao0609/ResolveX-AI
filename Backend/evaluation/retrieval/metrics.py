"""
Retrieval evaluation metrics for ResolveX.

Metrics:
    - Recall@K
    - Precision@K
    - MRR
    - nDCG@K
"""

import math
from typing import Iterable, Sequence


def recall_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Iterable[int],
    k: int,
) -> float:
    """Calculate Recall@K."""

    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Iterable[int],
    k: int,
) -> float:
    """Calculate Precision@K."""

    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_ids)

    if not retrieved_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    return sum(
        1 for document_id in retrieved
        if document_id in relevant
    ) / len(retrieved)


def reciprocal_rank(
    retrieved_ids: Sequence[int],
    relevant_ids: Iterable[int],
) -> float:
    """Calculate reciprocal rank."""

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Iterable[int],
    k: int,
) -> float:
    """
    Calculate binary-relevance nDCG@K.

    Documents are either:
        relevant = 1
        non-relevant = 0
    """

    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = retrieved_ids[:k]

    dcg = 0.0

    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant_count = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg