import pytest

from ai.rag.hybrid_fusion import HybridFusion
from ai.rag.retrieval_models import RetrievalCandidate


def test_equal_weight_fusion():
    fusion = HybridFusion(
        bm25_weight=0.5,
        dense_weight=0.5,
    )

    bm25 = [
        RetrievalCandidate(
            index_id=1,
            score=10.0,
            retriever="bm25",
        ),
        RetrievalCandidate(
            index_id=2,
            score=5.0,
            retriever="bm25",
        ),
    ]

    dense = [
        RetrievalCandidate(
            index_id=1,
            score=0.8,
            retriever="dense",
        ),
        RetrievalCandidate(
            index_id=3,
            score=0.4,
            retriever="dense",
        ),
    ]

    results = fusion.fuse(
        bm25,
        dense,
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].index_id == 1
    assert results[0].score == pytest.approx(1.0)


def test_candidate_present_in_both_gets_combined_score():
    fusion = HybridFusion()

    bm25 = [
        RetrievalCandidate(
            index_id=10,
            score=5.0,
            retriever="bm25",
        ),
        RetrievalCandidate(
            index_id=20,
            score=1.0,
            retriever="bm25",
        ),
    ]

    dense = [
        RetrievalCandidate(
            index_id=10,
            score=1.0,
            retriever="dense",
        ),
        RetrievalCandidate(
            index_id=30,
            score=0.2,
            retriever="dense",
        ),
    ]

    results = fusion.fuse(
        bm25,
        dense,
        top_k=3,
    )

    candidate = next(
        result
        for result in results
        if result.index_id == 10
    )

    assert candidate.score == pytest.approx(1.0)


def test_missing_retriever_score_is_zero():
    fusion = HybridFusion()

    bm25 = [
        RetrievalCandidate(
            index_id=1,
            score=10.0,
            retriever="bm25",
        ),
    ]

    dense = []

    results = fusion.fuse(
        bm25,
        dense,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.5)


def test_top_k_is_respected():
    fusion = HybridFusion()

    bm25 = [
        RetrievalCandidate(
            index_id=i,
            score=float(i),
            retriever="bm25",
        )
        for i in range(10)
    ]

    results = fusion.fuse(
        bm25,
        [],
        top_k=3,
    )

    assert len(results) == 3


def test_weights_are_normalized():
    fusion = HybridFusion(
        bm25_weight=2.0,
        dense_weight=1.0,
    )

    assert fusion.bm25_weight == pytest.approx(2 / 3)
    assert fusion.dense_weight == pytest.approx(1 / 3)


def test_zero_weights_are_rejected():
    with pytest.raises(ValueError):
        HybridFusion(
            bm25_weight=0.0,
            dense_weight=0.0,
        )


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        HybridFusion(
            bm25_weight=-0.1,
            dense_weight=1.0,
        )


def test_invalid_top_k_returns_empty():
    fusion = HybridFusion()

    results = fusion.fuse(
        [],
        [],
        top_k=0,
    )

    assert results == []