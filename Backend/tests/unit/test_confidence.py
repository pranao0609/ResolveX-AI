"""
test_confidence.py — Unit tests for the confidence score engine.

Tests compute_confidence() with explicit weight verification,
boundary clamping, and graceful handling of invalid inputs.
"""

import pytest
from ai.confidence.confidence_engine import compute_confidence, _clamp
from ai.config.ai_config import (
    CONFIDENCE_WEIGHT_SIMILARITY,
    CONFIDENCE_WEIGHT_LLM_SCORE,
    CONFIDENCE_WEIGHT_CLASSIFICATION,
)


class TestWeightSanity:
    """Verify weight constants are correctly configured."""

    def test_weights_sum_to_one(self):
        """The three confidence weights must sum to exactly 1.0."""
        total = (
            CONFIDENCE_WEIGHT_SIMILARITY
            + CONFIDENCE_WEIGHT_LLM_SCORE
            + CONFIDENCE_WEIGHT_CLASSIFICATION
        )
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_all_weights_are_positive(self):
        for name, w in [
            ("SIMILARITY", CONFIDENCE_WEIGHT_SIMILARITY),
            ("LLM_SCORE", CONFIDENCE_WEIGHT_LLM_SCORE),
            ("CLASSIFICATION", CONFIDENCE_WEIGHT_CLASSIFICATION),
        ]:
            assert w > 0, f"Weight {name} must be positive"


class TestComputeConfidence:
    """Core behaviour tests for compute_confidence()."""

    def test_perfect_scores_return_one(self):
        score = compute_confidence(1.0, 1.0, 1.0)
        assert score == 1.0

    def test_zero_scores_return_zero(self):
        score = compute_confidence(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_weighted_formula_is_correct(self):
        """Manual calculation must match the function output."""
        sim, llm, cls = 0.8, 0.6, 0.7
        expected = round(
            CONFIDENCE_WEIGHT_SIMILARITY * sim
            + CONFIDENCE_WEIGHT_LLM_SCORE * llm
            + CONFIDENCE_WEIGHT_CLASSIFICATION * cls,
            4,
        )
        assert compute_confidence(sim, llm, cls) == expected

    def test_output_rounded_to_4_decimal_places(self):
        score = compute_confidence(0.333, 0.666, 0.555)
        assert score == round(score, 4)

    def test_output_is_float(self):
        assert isinstance(compute_confidence(0.5, 0.5, 0.5), float)

    def test_high_similarity_dominates_low_others(self):
        """With similarity weight 0.4, a high sim score should pull result up."""
        score_high_sim = compute_confidence(0.99, 0.0, 0.0)
        score_low_sim = compute_confidence(0.0, 0.99, 0.0)
        assert score_high_sim > score_low_sim


class TestBoundaryClamping:
    """Values outside [0, 1] must be clamped, not crash."""

    def test_above_one_clamped_to_one(self):
        score = compute_confidence(2.0, 1.5, 1.1)
        assert score <= 1.0

    def test_negative_values_clamped_to_zero(self):
        score = compute_confidence(-0.5, -1.0, -0.1)
        assert score >= 0.0

    def test_mixed_out_of_range(self):
        score = compute_confidence(1.5, 0.5, -0.2)
        assert 0.0 <= score <= 1.0


class TestClampHelper:
    """Unit tests for the internal _clamp() utility."""

    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_above_hi(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_below_lo(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_exactly_at_boundaries(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_clamp_non_numeric_string_returns_lo(self):
        assert _clamp("bad_value") == 0.0

    def test_clamp_none_returns_lo(self):
        assert _clamp(None) == 0.0

    def test_clamp_custom_range(self):
        assert _clamp(0.3, lo=0.5, hi=1.0) == 0.5
        assert _clamp(1.5, lo=0.0, hi=1.0) == 1.0
