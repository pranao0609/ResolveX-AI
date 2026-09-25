"""
classification_agent.py

ResolveX Classification Agent.

Responsibilities:
- Execute primary ticket classification
- Validate classifier output
- Apply classification confidence policy
- Trigger controlled reclassification for low-confidence results
- Compare primary and alternative classification
- Preserve the primary result if reclassification fails
- Build structured classification decision metadata
- Provide safe fallback behavior
"""

from __future__ import annotations

from typing import Any

from app.core.logger import logger

from ai.classification.classifier import (
    classify_ticket,
    reclassify_with_zero_shot,
)

from ai.agents.classification_metadata import (
    build_classification_metadata,
)

from ai.agents.classification_policy import (
    classify_confidence_level,
    classification_requires_review,
    classification_requires_reclassification,
)


# ---------------------------------------------------------------------------
# Final categories exposed by ResolveX
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {
    "software",
    "network",
    "hardware",
    "access_permission",
    "security",
    "other",
}


DEFAULT_CATEGORY = "software"
DEFAULT_CONFIDENCE = 0.50


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_category(category: Any) -> str:
    """
    Validate and normalize the classifier category.

    Invalid categories are mapped to the safe default category.
    """

    if not isinstance(category, str):
        logger.warning(
            "Classification agent received non-string category: %r",
            category,
        )
        return DEFAULT_CATEGORY

    normalized = category.strip().lower()

    if normalized not in VALID_CATEGORIES:
        logger.warning(
            "Classification agent received invalid category: %r",
            category,
        )
        return DEFAULT_CATEGORY

    return normalized


def _validate_confidence(confidence: Any) -> float:
    """
    Validate and clamp classifier confidence to [0, 1].
    """

    try:
        value = float(confidence)

    except (TypeError, ValueError):
        logger.warning(
            "Classification agent received invalid confidence: %r",
            confidence,
        )
        return DEFAULT_CONFIDENCE

    if value < 0.0 or value > 1.0:
        logger.warning(
            "Classification confidence outside [0, 1]: %r",
            value,
        )

    return max(
        0.0,
        min(1.0, value),
    )


# ---------------------------------------------------------------------------
# Controlled reclassification
# ---------------------------------------------------------------------------


def _attempt_reclassification(
    text: str,
    primary_category: str,
    primary_confidence: float,
) -> dict[str, Any]:
    """
    Attempt an independent zero-shot reclassification.

    The alternative result replaces the primary result only when
    it provides a strictly higher confidence.

    If reclassification fails, the original result is preserved.
    """

    try:
        alternative_category, alternative_confidence = (
            reclassify_with_zero_shot(text)
        )

        alternative_category = _validate_category(
            alternative_category
        )

        alternative_confidence = _validate_confidence(
            alternative_confidence
        )

        logger.info(
            "Classification recheck: "
            "primary=%s(%.3f), "
            "alternative=%s(%.3f)",
            primary_category,
            primary_confidence,
            alternative_category,
            alternative_confidence,
        )

        # ---------------------------------------------------------------
        # Alternative classification wins
        # ---------------------------------------------------------------

        if alternative_confidence > primary_confidence:
            logger.info(
                "Reclassification selected: "
                "%s -> %s "
                "(%.3f -> %.3f)",
                primary_category,
                alternative_category,
                primary_confidence,
                alternative_confidence,
            )

            return {
                "category": alternative_category,
                "confidence": alternative_confidence,
                "reclassified": True,
                "reclassification_used": True,
                "reclassification_error": None,
                "alternative_category": alternative_category,
                "alternative_confidence": alternative_confidence,
            }

        # ---------------------------------------------------------------
        # Primary classification remains preferred
        # ---------------------------------------------------------------

        logger.info(
            "Primary classification retained after "
            "reclassification check"
        )

        return {
            "category": primary_category,
            "confidence": primary_confidence,
            "reclassified": False,
            "reclassification_used": True,
            "reclassification_error": None,
            "alternative_category": alternative_category,
            "alternative_confidence": alternative_confidence,
        }

    except Exception as exc:
        logger.warning(
            "Classification reclassification failed: %s",
            exc,
        )

        # Never destroy a valid primary classification because
        # the secondary strategy failed.

        return {
            "category": primary_category,
            "confidence": primary_confidence,
            "reclassified": False,
            "reclassification_used": False,
            "reclassification_error": str(exc),
            "alternative_category": None,
            "alternative_confidence": None,
        }


# ---------------------------------------------------------------------------
# Classification Agent
# ---------------------------------------------------------------------------


def run_classification_agent(
    text: str,
    *,
    ticket_id: int | str | None = None,
) -> dict[str, Any]:
    """
    Execute the ResolveX Classification Agent.

    Classification flow:

        Primary Classifier
                |
                v
        Validate Result
                |
                v
        Confidence Policy
                |
          confidence < LOW?
             /       \
           yes       no
            |         |
            v         |
      Reclassification|
            |         |
            v         |
       Compare scores |
            |         |
            +---------+
                |
                v
          Final Result
                |
                v
      Classification Metadata

    Returns:

    {
        "category": str,
        "confidence": float,
        "confidence_level": "high" | "medium" | "low",
        "requires_review": bool,
        "requires_reclassification": bool,
        "reclassified": bool,
        "reclassification_used": bool,
        "reclassification_error": str | None,
        "classification_metadata": dict,
        "fallback_used": bool,
        "error": str | None,
    }
    """

    fallback_used = False
    error_message = None

    try:
        # ===============================================================
        # 1. Input validation
        # ===============================================================

        if not isinstance(text, str):
            raise TypeError(
                "Classification input must be a string"
            )

        cleaned_text = text.strip()

        if not cleaned_text:
            logger.warning(
                "Classification agent received empty ticket text "
                "ticket_id=%s",
                ticket_id,
            )

            confidence = DEFAULT_CONFIDENCE

            classification_metadata = (
                build_classification_metadata(
                    primary_category=DEFAULT_CATEGORY,
                    primary_confidence=confidence,
                    final_category=DEFAULT_CATEGORY,
                    final_confidence=confidence,
                    reclassification_triggered=False,
                    reclassification_used=False,
                    reclassified=False,
                    alternative_category=None,
                    alternative_confidence=None,
                    reclassification_error=None,
                )
            )

            return {
                "category": DEFAULT_CATEGORY,
                "confidence": confidence,
                "confidence_level": classify_confidence_level(
                    confidence
                ),
                "requires_review": classification_requires_review(
                    confidence
                ),
                "requires_reclassification": (
                    classification_requires_reclassification(
                        confidence
                    )
                ),
                "reclassified": False,
                "reclassification_used": False,
                "reclassification_error": None,
                "classification_metadata": (
                    classification_metadata
                ),
                "fallback_used": True,
                "error": "empty_classification_input",
            }

        # ===============================================================
        # 2. Primary classification
        # ===============================================================

        category, confidence = classify_ticket(
            cleaned_text
        )

        # ===============================================================
        # 3. Validate primary classifier result
        # ===============================================================

        category = _validate_category(
            category
        )

        confidence = _validate_confidence(
            confidence
        )

        # ===============================================================
        # 4. Preserve primary classification
        # ===============================================================

        primary_category = category
        primary_confidence = confidence

        # Alternative classification metadata
        alternative_category = None
        alternative_confidence = None

        # Reclassification state
        reclassified = False
        reclassification_used = False
        reclassification_error = None

        # ===============================================================
        # 5. Phase 15.4 — Controlled Reclassification
        # ===============================================================

        reclassification_triggered = (
            classification_requires_reclassification(
                primary_confidence
            )
        )

        if reclassification_triggered:
            logger.info(
                "Low classification confidence detected. "
                "Starting controlled reclassification. "
                "ticket_id=%s "
                "category=%s "
                "confidence=%.3f",
                ticket_id,
                category,
                confidence,
            )

            reclassification_result = (
                _attempt_reclassification(
                    text=cleaned_text,
                    primary_category=primary_category,
                    primary_confidence=primary_confidence,
                )
            )

            # -----------------------------------------------------------
            # Apply final classification
            # -----------------------------------------------------------

            category = reclassification_result[
                "category"
            ]

            confidence = reclassification_result[
                "confidence"
            ]

            # -----------------------------------------------------------
            # Reclassification metadata
            # -----------------------------------------------------------

            reclassified = reclassification_result[
                "reclassified"
            ]

            reclassification_used = (
                reclassification_result[
                    "reclassification_used"
                ]
            )

            reclassification_error = (
                reclassification_result[
                    "reclassification_error"
                ]
            )

            alternative_category = (
                reclassification_result.get(
                    "alternative_category"
                )
            )

            alternative_confidence = (
                reclassification_result.get(
                    "alternative_confidence"
                )
            )

        # ===============================================================
        # 6. Final confidence policy
        # ===============================================================

        confidence_level = classify_confidence_level(
            confidence
        )

        requires_review = classification_requires_review(
            confidence
        )

        # This describes the FINAL confidence.
        requires_reclassification = (
            classification_requires_reclassification(
                confidence
            )
        )

        # ===============================================================
        # 7. Phase 15.5 — Classification Decision Metadata
        # ===============================================================

        classification_metadata = (
            build_classification_metadata(
                primary_category=primary_category,
                primary_confidence=primary_confidence,
                final_category=category,
                final_confidence=confidence,
                reclassification_triggered=(
                    reclassification_triggered
                ),
                reclassification_used=(
                    reclassification_used
                ),
                reclassified=reclassified,
                alternative_category=(
                    alternative_category
                ),
                alternative_confidence=(
                    alternative_confidence
                ),
                reclassification_error=(
                    reclassification_error
                ),
            )
        )

        # ===============================================================
        # 8. Final logging
        # ===============================================================

        logger.info(
            "Classification agent completed "
            "ticket_id=%s "
            "category=%s "
            "confidence=%.3f "
            "confidence_level=%s "
            "requires_review=%s "
            "requires_reclassification=%s "
            "reclassified=%s "
            "reclassification_used=%s "
            "decision_reason=%s "
            "fallback=%s",
            ticket_id,
            category,
            confidence,
            confidence_level,
            requires_review,
            requires_reclassification,
            reclassified,
            reclassification_used,
            classification_metadata[
                "decision_reason"
            ],
            fallback_used,
        )

        # ===============================================================
        # 9. Structured result
        # ===============================================================

        return {
            "category": category,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "requires_review": requires_review,
            "requires_reclassification": (
                requires_reclassification
            ),
            "reclassified": reclassified,
            "reclassification_used": (
                reclassification_used
            ),
            "reclassification_error": (
                reclassification_error
            ),
            "classification_metadata": (
                classification_metadata
            ),
            "fallback_used": fallback_used,
            "error": error_message,
        }

    # ===================================================================
    # Agent-level safe fallback
    # ===================================================================

    except Exception as exc:
        fallback_used = True
        error_message = str(exc)

        logger.exception(
            "Classification agent failed "
            "ticket_id=%s: %s",
            ticket_id,
            exc,
        )

        confidence = DEFAULT_CONFIDENCE

        classification_metadata = (
            build_classification_metadata(
                primary_category=DEFAULT_CATEGORY,
                primary_confidence=confidence,
                final_category=DEFAULT_CATEGORY,
                final_confidence=confidence,
                reclassification_triggered=False,
                reclassification_used=False,
                reclassified=False,
                alternative_category=None,
                alternative_confidence=None,
                reclassification_error=error_message,
            )
        )

        return {
            "category": DEFAULT_CATEGORY,
            "confidence": confidence,
            "confidence_level": classify_confidence_level(
                confidence
            ),
            "requires_review": classification_requires_review(
                confidence
            ),
            "requires_reclassification": (
                classification_requires_reclassification(
                    confidence
                )
            ),
            "reclassified": False,
            "reclassification_used": False,
            "reclassification_error": error_message,
            "classification_metadata": (
                classification_metadata
            ),
            "fallback_used": fallback_used,
            "error": error_message,
        }