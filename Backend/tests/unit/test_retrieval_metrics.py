import pytest

from evaluation.retrieval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [2, 4]

    assert recall_at_k(retrieved, relevant, 5) == 1.0


def test_recall_at_k_partial():
    retrieved = [1, 2, 3]
    relevant = [2, 4]

    assert recall_at_k(retrieved, relevant, 3) == 0.5


def test_precision_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [2, 4]

    assert precision_at_k(retrieved, relevant, 5) == 0.4


def test_reciprocal_rank():
    retrieved = [10, 20, 30]
    relevant = [30]

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_match():
    retrieved = [10, 20, 30]
    relevant = [40]

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_ndcg_perfect():
    retrieved = [1, 2, 3]
    relevant = [1, 2, 3]

    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_ndcg_partial():
    retrieved = [3, 4, 1]
    relevant = [1, 2, 3]

    score = ndcg_at_k(retrieved, relevant, 3)

    assert 0.0 < score < 1.0


def test_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k([1, 2], [1], 0)