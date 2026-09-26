"""
classification_policy.py

Confidence policy for ResolveX classification.

Uses the centralized application thresholds:
- CONFIDENCE_HIGH
- CONFIDENCE_LOW

This policy describes classification reliability.
It does not make the final ticket-resolution decision.
"""

from __future__ import annotations

from typing import Literal

from app.core.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
)

ClassificationConfidenceLevel = Literal[
    "high",
    "medium",
    "low",
]


def classify_confidence_level(
    confidence: float,
) -> ClassificationConfidenceLevel:
    """
    Map classification confidence to a qualitative level.

    Thresholds are sourced from the centralized application
    configuration through app.core.constants.
    """

    confidence = max(
        0.0,
        min(1.0, float(confidence)),
    )

    if confidence >= CONFIDENCE_HIGH:
        return "high"

    if confidence >= CONFIDENCE_LOW:
        return "medium"

    return "low"


def classification_requires_review(
    confidence: float,
) -> bool:
    """
    Determine whether classification confidence is below
    the configured high-confidence threshold.

    This does NOT mean the ticket must be escalated.
    It indicates that downstream stages should treat the
    classification with additional caution.
    """

    return float(confidence) < CONFIDENCE_HIGH


def classification_requires_reclassification(
    confidence: float,
) -> bool:
    """
    Determine whether classification confidence is below
    the configured HITL threshold.

    A low-confidence classification should not be treated
    as reliable enough for normal downstream processing.
    """

    return float(confidence) < CONFIDENCE_LOW
