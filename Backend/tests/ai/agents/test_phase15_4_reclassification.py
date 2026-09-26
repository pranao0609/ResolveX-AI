from unittest.mock import patch

from ai.agents.classification_agent import (
    run_classification_agent,
)

# ---------------------------------------------------------------------------
# Low confidence -> reclassification -> alternative wins
# ---------------------------------------------------------------------------


def test_low_confidence_triggers_reclassification_and_selects_higher_confidence():
    with (
        patch(
            "ai.agents.classification_agent.classify_ticket",
            return_value=("software", 0.40),
        ),
        patch(
            "ai.agents.classification_agent.reclassify_with_zero_shot",
            return_value=("network", 0.85),
        ) as mock_reclassifier,
    ):

        result = run_classification_agent(
            "The office network connection keeps dropping."
        )

    mock_reclassifier.assert_called_once()

    assert result["category"] == "network"
    assert result["confidence"] == 0.85

    assert result["reclassified"] is True
    assert result["reclassification_used"] is True
    assert result["reclassification_error"] is None

    assert result["confidence_level"] == "high"
    assert result["requires_review"] is False
    assert result["requires_reclassification"] is False

    assert result["fallback_used"] is False


# ---------------------------------------------------------------------------
# Low confidence -> reclassification -> primary remains
# ---------------------------------------------------------------------------


def test_lower_confidence_alternative_does_not_replace_primary():
    with (
        patch(
            "ai.agents.classification_agent.classify_ticket",
            return_value=("software", 0.40),
        ),
        patch(
            "ai.agents.classification_agent.reclassify_with_zero_shot",
            return_value=("network", 0.35),
        ) as mock_reclassifier,
    ):

        result = run_classification_agent("Some application is behaving unexpectedly.")

    mock_reclassifier.assert_called_once()

    assert result["category"] == "software"
    assert result["confidence"] == 0.40

    assert result["reclassified"] is False
    assert result["reclassification_used"] is True
    assert result["reclassification_error"] is None

    assert result["confidence_level"] == "low"
    assert result["requires_review"] is True
    assert result["requires_reclassification"] is True

    assert result["fallback_used"] is False


# ---------------------------------------------------------------------------
# Reclassification failure -> preserve primary
# ---------------------------------------------------------------------------


def test_reclassification_failure_preserves_primary_result():
    with (
        patch(
            "ai.agents.classification_agent.classify_ticket",
            return_value=("software", 0.40),
        ),
        patch(
            "ai.agents.classification_agent.reclassify_with_zero_shot",
            side_effect=RuntimeError("zero-shot model unavailable"),
        ) as mock_reclassifier,
    ):

        result = run_classification_agent("The application has an unknown issue.")

    mock_reclassifier.assert_called_once()

    assert result["category"] == "software"
    assert result["confidence"] == 0.40

    assert result["reclassified"] is False
    assert result["reclassification_used"] is False

    assert result["reclassification_error"] == "zero-shot model unavailable"

    assert result["fallback_used"] is False


# ---------------------------------------------------------------------------
# High confidence -> no reclassification
# ---------------------------------------------------------------------------


def test_high_confidence_does_not_trigger_reclassification():
    with (
        patch(
            "ai.agents.classification_agent.classify_ticket",
            return_value=("hardware", 0.97),
        ),
        patch(
            "ai.agents.classification_agent.reclassify_with_zero_shot",
        ) as mock_reclassifier,
    ):

        result = run_classification_agent(
            "The mouse is broken and no longer responding."
        )

    mock_reclassifier.assert_not_called()

    assert result["category"] == "hardware"
    assert result["confidence"] == 0.97

    assert result["reclassified"] is False
    assert result["reclassification_used"] is False
    assert result["reclassification_error"] is None

    assert result["confidence_level"] == "high"
    assert result["requires_review"] is False
    assert result["requires_reclassification"] is False

    assert result["fallback_used"] is False
