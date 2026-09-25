from unittest.mock import patch

from ai.graph.nodes.resolution import resolution_agent


def test_resolution_uses_conversation_memory():
    state = {
        "ticket_id": 501,
        "ticket_text": "Unable to access email",
        "cleaned_ticket": "Unable to access email",
        "retrieved_context": "Users must use valid credentials.",
        "classification": {
            "category": "authentication",
            "confidence": 0.95,
        },
        "diagnosis_result": {
            "diagnosis": "Email authentication failure",
            "root_cause": "Invalid credentials",
        },
        "conversation_history": [
            {
                "role": "user",
                "content": "Unable to access email",
                "metadata": {"ticket_id": 501},
            },
            {
                "role": "assistant",
                "content": "Email authentication failure",
                "metadata": {
                    "ticket_id": 501,
                    "root_cause": "Invalid credentials",
                },
            },
        ],
        "previous_tickets": [],
    }

    mock_result = {
        "resolution": "Reset the email password and retry authentication.",
        "confidence": 0.91,
        "requires_human": False,
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=mock_result,
    ) as mock_generate:

        result = resolution_agent(state)

    mock_generate.assert_called_once()

    kwargs = mock_generate.call_args.kwargs

    assert kwargs["conversation_history"] == state[
        "conversation_history"
    ]

    assert kwargs["ticket_text"] == "Unable to access email"
    assert kwargs["diagnosis"] == "Email authentication failure"
    assert kwargs["root_cause"] == "Invalid credentials"

    assert result["resolution"] == (
        "Reset the email password and retry authentication."
    )


def test_resolution_uses_historical_memory():
    state = {
        "ticket_id": 502,
        "ticket_text": "VPN authentication failing",
        "cleaned_ticket": "VPN authentication failing",
        "retrieved_context": "VPN requires valid credentials.",
        "classification": {
            "category": "authentication",
            "confidence": 0.94,
        },
        "diagnosis_result": {
            "diagnosis": "VPN authentication failure",
            "root_cause": "Expired credentials",
        },
        "conversation_history": [],
        "previous_tickets": [
            {
                "ticket_id": 401,
                "title": "VPN login failure",
                "description": "User could not connect to VPN.",
                "category": "authentication",
                "solution": "Reset VPN credentials.",
                "confidence": 0.92,
            }
        ],
    }

    mock_result = {
        "resolution": "Reset the VPN credentials.",
        "confidence": 0.93,
        "requires_human": False,
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=mock_result,
    ) as mock_generate:

        result = resolution_agent(state)

    kwargs = mock_generate.call_args.kwargs

    assert "Historical troubleshooting evidence:" in kwargs[
        "context"
    ]

    assert "Reset VPN credentials." in kwargs["context"]

    assert result["resolution"] == (
        "Reset the VPN credentials."
    )


def test_resolution_without_memory_preserves_existing_behavior():
    state = {
        "ticket_id": 503,
        "ticket_text": "Printer is not working",
        "cleaned_ticket": "Printer is not working",
        "retrieved_context": "Check printer power and connection.",
        "classification": {
            "category": "hardware",
            "confidence": 0.90,
        },
        "diagnosis_result": {
            "diagnosis": "Printer connectivity issue",
            "root_cause": "Disconnected network cable",
        },
        "conversation_history": [],
        "previous_tickets": [],
    }

    mock_result = {
        "resolution": "Reconnect the printer network cable.",
        "confidence": 0.90,
        "requires_human": False,
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=mock_result,
    ) as mock_generate:

        result = resolution_agent(state)

    kwargs = mock_generate.call_args.kwargs

    assert kwargs["context"] == (
        "Check printer power and connection."
    )

    assert kwargs["conversation_history"] == []

    assert result["resolution"] == (
        "Reconnect the printer network cable."
    )


def test_resolution_memory_metadata():
    state = {
        "ticket_id": 504,
        "ticket_text": "Application crashes",
        "cleaned_ticket": "Application crashes",
        "retrieved_context": "Restart the application.",
        "classification": {
            "category": "software",
            "confidence": 0.91,
        },
        "diagnosis_result": {
            "diagnosis": "Application runtime failure",
            "root_cause": "Corrupted application state",
        },
        "conversation_history": [
            {
                "role": "user",
                "content": "Application crashes",
            }
        ],
        "previous_tickets": [
            {
                "ticket_id": 300,
                "title": "Previous crash",
                "description": "Application crashed.",
                "solution": "Restart application.",
            }
        ],
    }

    mock_result = {
        "resolution": "Restart the application.",
        "confidence": 0.89,
        "requires_human": False,
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=mock_result,
    ):

        result = resolution_agent(state)

    memory_metadata = result["metadata"]["memory"]

    assert memory_metadata["historical_ticket_count"] == 1
    assert memory_metadata["historical_memory_used"] is True
    assert memory_metadata["conversation_memory_count"] == 1