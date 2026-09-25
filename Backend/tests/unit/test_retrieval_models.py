import pytest

from ai.rag.retrieval_models import RetrievalCandidate


def test_retrieval_candidate_creation():
    candidate = RetrievalCandidate(
        index_id=10,
        score=0.85,
        retriever="dense",
    )

    assert candidate.index_id == 10
    assert candidate.score == 0.85
    assert candidate.retriever == "dense"


def test_retrieval_candidate_requires_retriever():
    with pytest.raises(ValueError):
        RetrievalCandidate(
            index_id=10,
            score=0.85,
        )


def test_retrieval_candidate_rejects_negative_index():
    with pytest.raises(ValueError):
        RetrievalCandidate(
            index_id=-1,
            score=0.85,
            retriever="dense",
        )