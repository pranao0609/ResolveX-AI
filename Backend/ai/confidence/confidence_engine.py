"""
confidence_engine.py — Composite confidence score calculator.

Formula:
    confidence = (0.4 × similarity_score)
               + (0.3 × llm_score)
               + (0.3 × classification_score)

All input scores must be in [0.0, 1.0].
Output is clamped to [0.0, 1.0].
"""

from ai.config.ai_config import (
    CONFIDENCE_WEIGHT_SIMILARITY,
    CONFIDENCE_WEIGHT_LLM_SCORE,
    CONFIDENCE_WEIGHT_CLASSIFICATION,
)
from app.core.logger import logger


def compute_confidence(
    similarity_score: float,
    llm_score: float,
    classification_score: float,
) -> float:
    """
    Compute the weighted composite confidence score.

    Args:
        similarity_score:     How similar the retrieved KB context is (0–1).
        llm_score:            Proxy for LLM solution quality (0–1).
        classification_score: Confidence of the ticket classifier (0–1).

    Returns:
        Composite confidence float in [0.0, 1.0], rounded to 4 decimal places.
    """
    raw = (
        CONFIDENCE_WEIGHT_SIMILARITY * _clamp(similarity_score)
        + CONFIDENCE_WEIGHT_LLM_SCORE * _clamp(llm_score)
        + CONFIDENCE_WEIGHT_CLASSIFICATION * _clamp(classification_score)
    )
    score = round(_clamp(raw), 4)
    logger.debug(
        f"Confidence: sim={similarity_score:.3f} × {CONFIDENCE_WEIGHT_SIMILARITY} "
        f"+ llm={llm_score:.3f} × {CONFIDENCE_WEIGHT_LLM_SCORE} "
        f"+ cls={classification_score:.3f} × {CONFIDENCE_WEIGHT_CLASSIFICATION} "
        f"= {score}"
    )
    return score


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi] and handle non-numeric input gracefully."""
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo
