from unittest.mock import patch

from ai.graph.nodes import ticket_analyzer


def test_ticket_analyzer_uses_classification_agent():
    state = {
        "ticket_id": 201,
        "ticket_text": "Mouse is not working",
        "attachment_paths": None,
        "request_id": "request-201",
        "graph_run_id": "graph-201",
        "errors": [],
        "warnings": [],
        "fallback_used": False,
        "metadata": {},
    }

    with patch(
        "ai.graph.nodes.ticket_analyzer.run_classification_agent",
        return_value={
            "category": "hardware",
            "confidence": 0.97,
            "confidence_level": "high",
            "requires_review": False,
            "requires_reclassification": False,
            "fallback_used": False,
            "error": None,
        },
    ) as mock_agent:

        result = ticket_analyzer(state)

    mock_agent.assert_called_once()

    assert result["category"] == "hardware"
    assert result["category_confidence"] == 0.97
    assert result["classification_confidence_level"] == "high"
    assert result["classification_requires_review"] is False
    assert result["classification_requires_reclassification"] is False
    assert result.get("fallback_used") is None


def test_ticket_analyzer_propagates_classification_fallback():
    state = {
        "ticket_id": 202,
        "ticket_text": "Something is broken",
        "attachment_paths": None,
        "request_id": "request-202",
        "graph_run_id": "graph-202",
        "errors": [],
        "warnings": [],
        "fallback_used": False,
        "metadata": {},
    }

    with patch(
        "ai.graph.nodes.ticket_analyzer.run_classification_agent",
        return_value={
            "category": "software",
            "confidence": 0.50,
            "confidence_level": "medium",
            "requires_review": True,
            "requires_reclassification": False,
            "fallback_used": True,
            "error": "classifier unavailable",
        },
    ):

        result = ticket_analyzer(state)

    assert result["category"] == "software"
    assert result["category_confidence"] == 0.50
    assert result["classification_confidence_level"] == "medium"
    assert result["classification_requires_review"] is True
    assert result["classification_requires_reclassification"] is False
    assert result["fallback_used"] is True

    assert "errors" in result
    assert any("classifier unavailable" in error for error in result["errors"])


def test_ticket_analyzer_handles_classification_agent_failure():
    state = {
        "ticket_id": 203,
        "ticket_text": "VPN is not working",
        "attachment_paths": None,
        "request_id": "request-203",
        "graph_run_id": "graph-203",
        "errors": [],
        "warnings": [],
        "fallback_used": False,
        "metadata": {},
    }

    with patch(
        "ai.graph.nodes.ticket_analyzer.run_classification_agent",
        side_effect=RuntimeError("classification agent unavailable"),
    ):

        result = ticket_analyzer(state)

    assert result["category"] == "software"
    assert result["category_confidence"] == 0.0
    assert result["fallback_used"] is True

    assert "errors" in result
    assert any(
        "classification agent unavailable" in error for error in result["errors"]
    )
