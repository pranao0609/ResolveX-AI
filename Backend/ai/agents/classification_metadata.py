"""
classification_metadata.py

Structured metadata produced by the ResolveX Classification Agent.

This module contains metadata only.
It does not perform classification.
"""

from __future__ import annotations

from typing import Any


def build_classification_metadata(
    *,
    primary_category: str,
    primary_confidence: float,
    final_category: str,
    final_confidence: float,
    reclassification_triggered: bool,
    reclassification_used: bool,
    reclassified: bool,
    alternative_category: str | None = None,
    alternative_confidence: float | None = None,
    reclassification_error: str | None = None,
) -> dict[str, Any]:
    """
    Build a structured classification decision record.

    The metadata describes what happened during classification without
    changing the classification decision itself.
    """

    confidence_delta = (
        final_confidence - primary_confidence
    )

    if reclassification_error:
        decision_reason = "reclassification_failed"

    elif reclassified:
        decision_reason = "alternative_classification_selected"

    elif reclassification_used:
        decision_reason = "primary_classification_retained"

    elif reclassification_triggered:
        decision_reason = "reclassification_not_completed"

    else:
        decision_reason = "primary_classification"

    if reclassification_triggered:
        strategy = "primary_plus_zero_shot_reclassification"
    else:
        strategy = "primary_only"

    return {
        "strategy": strategy,

        "primary": {
            "category": primary_category,
            "confidence": primary_confidence,
        },

        "alternative": {
            "category": alternative_category,
            "confidence": alternative_confidence,
        },

        "final": {
            "category": final_category,
            "confidence": final_confidence,
        },

        "reclassification": {
            "triggered": reclassification_triggered,
            "used": reclassification_used,
            "selected": reclassified,
            "error": reclassification_error,
        },

        "confidence_delta": round(
            confidence_delta,
            6,
        ),

        "decision_reason": decision_reason,
    }