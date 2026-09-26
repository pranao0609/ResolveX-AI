"""
score_normalizer.py — Retrieval score normalization.

Normalizes scores from different retrieval systems into a common
0–1 range before hybrid fusion.
"""

from typing import List

from ai.rag.retrieval_models import RetrievalCandidate


class ScoreNormalizer:
    """Normalize retrieval candidate scores."""

    @staticmethod
    def min_max(
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """
        Apply min-max normalization to candidate scores.

        Formula:

            normalized =
                (score - min_score) /
                (max_score - min_score)

        If all scores are identical, every candidate receives 1.0.
        """

        if not candidates:
            return []

        scores = [candidate.score for candidate in candidates]

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [
                candidate.model_copy(update={"score": 1.0}) for candidate in candidates
            ]

        return [
            candidate.model_copy(
                update={
                    "score": (candidate.score - min_score) / (max_score - min_score)
                }
            )
            for candidate in candidates
        ]
