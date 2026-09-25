"""
hybrid_fusion.py — Weighted fusion of lexical and dense retrieval.

Combines normalized BM25 and dense retrieval scores using configurable
weights.
"""

from typing import Dict, List

from ai.rag.retrieval_models import RetrievalCandidate
from ai.rag.score_normalizer import ScoreNormalizer


class HybridFusion:
    """Combine BM25 and dense retrieval candidates."""

    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        if bm25_weight < 0:
            raise ValueError(
                "bm25_weight cannot be negative"
            )

        if dense_weight < 0:
            raise ValueError(
                "dense_weight cannot be negative"
            )

        if bm25_weight + dense_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero"
            )

        total_weight = bm25_weight + dense_weight

        self.bm25_weight = bm25_weight / total_weight
        self.dense_weight = dense_weight / total_weight

    def fuse(
        self,
        bm25_candidates: List[RetrievalCandidate],
        dense_candidates: List[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """
        Normalize and combine BM25 and dense candidates.

        Candidates are merged by index_id.
        """

        if top_k <= 0:
            return []

        normalized_bm25 = ScoreNormalizer.min_max(
            bm25_candidates
        )

        normalized_dense = ScoreNormalizer.min_max(
            dense_candidates
        )

        bm25_scores: Dict[int, float] = {
            candidate.index_id: candidate.score
            for candidate in normalized_bm25
        }

        dense_scores: Dict[int, float] = {
            candidate.index_id: candidate.score
            for candidate in normalized_dense
        }

        candidate_ids = set(bm25_scores) | set(dense_scores)

        fused_candidates = []

        for index_id in candidate_ids:
            bm25_score = bm25_scores.get(
                index_id,
                0.0,
            )

            dense_score = dense_scores.get(
                index_id,
                0.0,
            )

            hybrid_score = (
                self.bm25_weight * bm25_score
                + self.dense_weight * dense_score
            )

            fused_candidates.append(
                RetrievalCandidate(
                    index_id=index_id,
                    score=hybrid_score,
                    retriever="hybrid",
                )
            )

        fused_candidates.sort(
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        return fused_candidates[:top_k]