from unittest.mock import patch

from ai.agents.classification_agent import (
    run_classification_agent,
)

# ---------------------------------------------------------------------------
# Primary classification only
# ---------------------------------------------------------------------------


def test_classification_metadata_primary_only():

    with (
        patch(
            "ai.agents.classification_agent.classify_ticket",
            return_value=("hardware", 0.95),
        ),
        patch(
            "ai.agents.classification_agent.reclassify_with_zero_shot",
        ) as mock_reclassifier,
    ):

        result = run_classification_agent("The mouse is broken.")

    mock_reclassifier.assert_not_called()

    metadata = result["classification_metadata"]

    assert metadata["strategy"] == "primary_only"

    assert metadata["primary"]["category"] == "hardware"
    assert metadata["primary"]["confidence"] == 0.95

    assert metadata["final"]["category"] == "hardware"
    assert metadata["final"]["confidence"] == 0.95

    assert metadata["alternative"]["category"] is None
    assert metadata["alternative"]["confidence"] is None

    assert metadata["reclassification"]["triggered"] is False

    assert metadata["reclassification"]["used"] is False

    assert metadata["reclassification"]["selected"] is False

    assert metadata["decision_reason"] == "primary_classification"

    assert metadata["confidence_delta"] == 0.0


# ---------------------------------------------------------------------------
# Alternative classification selected
# ---------------------------------------------------------------------------


def test_classification_metadata_when_reclassification_wins():

    with (
        patch(
            "ai.agents.classification_agent.classify_ticket",
            return_value=("software", 0.40),
        ),
        patch(
            "ai.agents.classification_agent.reclassify_with_zero_shot",
            return_value=("network", 0.85),
        ),
    ):

        result = run_classification_agent("The network connection keeps dropping.")

    metadata = result["classification_metadata"]

    assert metadata["strategy"] == ("primary_plus_zero_shot_reclassification")

    assert metadata["primary"]["category"] == "software"
    assert metadata["primary"]["confidence"] == 0.40

    assert metadata["alternative"]["category"] == "network"
    assert metadata["alternative"]["confidence"] == 0.85

    assert metadata["final"]["category"] == "network"
    assert metadata["final"]["confidence"] == 0.85

    assert metadata["reclassification"]["triggered"] is True

    assert metadata["reclassification"]["used"] is True

    assert metadata["reclassification"]["selected"] is True

    assert metadata["decision_reason"] == "alternative_classification_selected"

    assert metadata["confidence_delta"] == 0.45


# ---------------------------------------------------------------------------
# Alternative rejected
# ---------------------------------------------------------------------------


def test_classification_metadata_when_primary_is_retained():

    with (
        patch(
            "ai.agents.classification_agent.classify_ticket",
            return_value=("software", 0.40),
        ),
        patch(
            "ai.agents.classification_agent.reclassify_with_zero_shot",
            return_value=("network", 0.35),
        ),
    ):

        result = run_classification_agent("The application has an issue.")

    metadata = result["classification_metadata"]

    assert metadata["primary"]["category"] == "software"
    assert metadata["primary"]["confidence"] == 0.40

    assert metadata["alternative"]["category"] == "network"
    assert metadata["alternative"]["confidence"] == 0.35

    assert metadata["final"]["category"] == "software"
    assert metadata["final"]["confidence"] == 0.40

    assert metadata["reclassification"]["triggered"] is True

    assert metadata["reclassification"]["used"] is True

    assert metadata["reclassification"]["selected"] is False

    assert metadata["decision_reason"] == "primary_classification_retained"

    assert metadata["confidence_delta"] == 0.0


# ---------------------------------------------------------------------------
# Reclassification failure
# ---------------------------------------------------------------------------


def test_classification_metadata_when_reclassification_fails():

    with (
        patch(
            "ai.agents.classification_agent.classify_ticket",
            return_value=("software", 0.40),
        ),
        patch(
            "ai.agents.classification_agent.reclassify_with_zero_shot",
            side_effect=RuntimeError("zero-shot model unavailable"),
        ),
    ):

        result = run_classification_agent("The application has an unknown issue.")

    metadata = result["classification_metadata"]

    assert metadata["primary"]["category"] == "software"
    assert metadata["primary"]["confidence"] == 0.40

    assert metadata["alternative"]["category"] is None
    assert metadata["alternative"]["confidence"] is None

    assert metadata["final"]["category"] == "software"
    assert metadata["final"]["confidence"] == 0.40

    assert metadata["reclassification"]["triggered"] is True

    assert metadata["reclassification"]["used"] is False

    assert metadata["reclassification"]["selected"] is False

    assert metadata["reclassification"]["error"] == "zero-shot model unavailable"

    assert metadata["decision_reason"] == "reclassification_failed"

    assert metadata["confidence_delta"] == 0.0
