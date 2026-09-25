from unittest.mock import patch

from ai.agents.classification_agent import (
    DEFAULT_CATEGORY,
    DEFAULT_CONFIDENCE,
    VALID_CATEGORIES,
    run_classification_agent,
)


def test_classification_agent_returns_valid_result():
    with patch(
        "ai.agents.classification_agent.classify_ticket",
        return_value=("hardware", 0.97),
    ):
        result = run_classification_agent(
            "mouse is not working",
            ticket_id=101,
        )

    assert result["category"] == "hardware"
    assert result["confidence"] == 0.97
    assert result["confidence_level"] == "high"
    assert result["requires_review"] is False
    assert result["requires_reclassification"] is False
    assert result["fallback_used"] is False
    assert result["error"] is None


def test_classification_agent_normalizes_category():
    with patch(
        "ai.agents.classification_agent.classify_ticket",
        return_value=(" HARDWARE ", 0.91),
    ):
        result = run_classification_agent(
            "mouse problem",
            ticket_id=102,
        )

    assert result["category"] == "hardware"
    assert result["confidence"] == 0.91
    assert result["confidence_level"] == "high"
    assert result["requires_review"] is False
    assert result["requires_reclassification"] is False


def test_classification_agent_clamps_confidence():
    with patch(
        "ai.agents.classification_agent.classify_ticket",
        return_value=("network", 1.5),
    ):
        result = run_classification_agent(
            "network is down",
            ticket_id=103,
        )

    assert result["category"] == "network"
    assert result["confidence"] == 1.0
    assert result["confidence_level"] == "high"
    assert result["requires_review"] is False
    assert result["requires_reclassification"] is False


def test_classification_agent_handles_invalid_category():
    with patch(
        "ai.agents.classification_agent.classify_ticket",
        return_value=("unknown_category", 0.88),
    ):
        result = run_classification_agent(
            "something is broken",
            ticket_id=104,
        )

    assert result["category"] == DEFAULT_CATEGORY
    assert result["confidence"] == 0.88
    assert result["confidence_level"] == "high"
    assert result["requires_review"] is False
    assert result["requires_reclassification"] is False
    assert result["fallback_used"] is False


def test_classification_agent_handles_invalid_confidence():
    with patch(
        "ai.agents.classification_agent.classify_ticket",
        return_value=("software", "invalid"),
    ):
        result = run_classification_agent(
            "application error",
            ticket_id=105,
        )

    assert result["category"] == "software"
    assert result["confidence"] == DEFAULT_CONFIDENCE
    assert result["confidence_level"] == "medium"
    assert result["requires_review"] is True
    assert result["requires_reclassification"] is False
    assert result["fallback_used"] is False


def test_classification_agent_handles_empty_text():
    result = run_classification_agent(
        "",
        ticket_id=106,
    )

    assert result["category"] == DEFAULT_CATEGORY
    assert result["confidence"] == DEFAULT_CONFIDENCE
    assert result["confidence_level"] == "medium"
    assert result["requires_review"] is True
    assert result["requires_reclassification"] is False
    assert result["fallback_used"] is True
    assert result["error"] == "empty_classification_input"


def test_classification_agent_handles_classifier_failure():
    with patch(
        "ai.agents.classification_agent.classify_ticket",
        side_effect=RuntimeError("classifier unavailable"),
    ):
        result = run_classification_agent(
            "VPN is not connecting",
            ticket_id=107,
        )

    assert result["category"] == DEFAULT_CATEGORY
    assert result["confidence"] == DEFAULT_CONFIDENCE
    assert result["confidence_level"] == "medium"
    assert result["requires_review"] is True
    assert result["requires_reclassification"] is False
    assert result["fallback_used"] is True
    assert "classifier unavailable" in result["error"]


def test_classification_agent_marks_medium_confidence_for_review():
    with patch(
        "ai.agents.classification_agent.classify_ticket",
        return_value=("network", 0.60),
    ):
        result = run_classification_agent(
            "network problem",
            ticket_id=108,
        )

    assert result["category"] == "network"
    assert result["confidence"] == 0.60
    assert result["confidence_level"] == "medium"
    assert result["requires_review"] is True
    assert result["requires_reclassification"] is False
    assert result["fallback_used"] is False
    assert result["error"] is None


def test_classification_agent_marks_low_confidence_for_reclassification():
    with patch(
        "ai.agents.classification_agent.classify_ticket",
        return_value=("software", 0.40),
    ):
        result = run_classification_agent(
            "something is wrong",
            ticket_id=109,
        )

    assert result["category"] == "software"
    assert result["confidence"] == 0.40
    assert result["confidence_level"] == "low"
    assert result["requires_review"] is True
    assert result["requires_reclassification"] is True
    assert result["fallback_used"] is False
    assert result["error"] is None


def test_valid_categories_are_fixed():
    assert VALID_CATEGORIES == {
        "software",
        "network",
        "hardware",
        "access_permission",
        "security",
        "other",
    }   