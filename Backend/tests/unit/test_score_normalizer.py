import pytest

from ai.rag.retrieval_models import RetrievalCandidate
from ai.rag.score_normalizer import ScoreNormalizer


def test_min_max_normalization():
    candidates = [
        RetrievalCandidate(
            index_id=1,
            score=10.0,
            retriever="bm25",
        ),
        RetrievalCandidate(
            index_id=2,
            score=20.0,
            retriever="bm25",
        ),
        RetrievalCandidate(
            index_id=3,
            score=30.0,
            retriever="bm25",
        ),
    ]

    normalized = ScoreNormalizer.min_max(candidates)

    assert normalized[0].score == pytest.approx(0.0)
    assert normalized[1].score == pytest.approx(0.5)
    assert normalized[2].score == pytest.approx(1.0)


def test_identical_scores():
    candidates = [
        RetrievalCandidate(
            index_id=1,
            score=5.0,
            retriever="bm25",
        ),
        RetrievalCandidate(
            index_id=2,
            score=5.0,
            retriever="bm25",
        ),
    ]

    normalized = ScoreNormalizer.min_max(candidates)

    assert all(
        candidate.score == pytest.approx(1.0)
        for candidate in normalized
    )


def test_empty_candidates():
    assert ScoreNormalizer.min_max([]) == []


def test_retriever_is_preserved():
    candidates = [
        RetrievalCandidate(
            index_id=1,
            score=2.0,
            retriever="dense",
        ),
        RetrievalCandidate(
            index_id=2,
            score=4.0,
            retriever="dense",
        ),
    ]

    normalized = ScoreNormalizer.min_max(candidates)

    assert normalized[0].retriever == "dense"
    assert normalized[1].retriever == "dense"


def test_scores_are_between_zero_and_one():
    candidates = [
        RetrievalCandidate(
            index_id=1,
            score=3.0,
            retriever="dense",
        ),
        RetrievalCandidate(
            index_id=2,
            score=7.0,
            retriever="dense",
        ),
        RetrievalCandidate(
            index_id=3,
            score=5.0,
            retriever="dense",
        ),
    ]

    normalized = ScoreNormalizer.min_max(candidates)

    assert all(
        0.0 <= candidate.score <= 1.0
        for candidate in normalized
    )