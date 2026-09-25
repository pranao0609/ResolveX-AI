from unittest.mock import patch


def test_diagnosis_agent_uses_historical_memory():
    historical_tickets = [
        {
            "ticket_id": 101,
            "title": "Keyboard failure",
            "description": (
                "Laptop keyboard stopped responding."
            ),
            "category": "hardware",
            "status": "closed",
            "solution": (
                "Reinstall the keyboard driver."
            ),
            "confidence": 0.92,
            "explanation": None,
            "submitted_by": "user101",
            "created_at": None,
            "updated_at": None,
            "metadata": {},
        }
    ]

    diagnosis_result = type(
        "DiagnosisResult",
        (),
        {
            "problem": "Keyboard failure",
            "possible_root_cause": (
                "Keyboard driver failure."
            ),
            "evidence": [
                "Historical ticket matched."
            ],
            "missing_information": [],
            "confidence": 0.91,
            "model_dump": lambda self: {
                "problem": "Keyboard failure",
                "possible_root_cause": (
                    "Keyboard driver failure."
                ),
                "evidence": [
                    "Historical ticket matched."
                ],
                "missing_information": [],
                "confidence": 0.91,
            },
        },
    )()

    tool_result = {
        "tool": "search_memory",
        "status": "success",
        "data": historical_tickets,
        "result_count": 1,
        "latency_ms": 0.1,
    }

    tool_record = {
        "agent": "diagnosis_agent",
        "tool": "search_memory",
        "category": "read",
        "input": {
            "query": "keyboard not working",
        },
        "result_count": 1,
        "status": "success",
        "latency_ms": 0.1,
        "error": None,
    }

    state = {
        "ticket_id": 10,
        "cleaned_ticket": (
            "Keyboard not working"
        ),
        "category": "hardware",
        "retrieved_context": (
            "Keyboard troubleshooting evidence."
        ),
        "conversation_history": [],
        "previous_tickets": [],
        "tool_calls": [],
        "fallback_used": False,
        "metadata": {},
    }

    captured = {}

    def fake_generate_diagnosis(
        *,
        ticket_text,
        context,
        classification,
    ):
        captured["context"] = context

        return (
            diagnosis_result,
            False,
        )

    with patch(
        "ai.graph.nodes.diagnosis.SessionLocal"
    ) as mock_session:

        mock_db = mock_session.return_value

        with patch(
            "ai.graph.nodes.diagnosis.execute_tool",
            return_value=(
                tool_result,
                tool_record,
            ),
        ):

            with patch(
                "ai.graph.nodes.diagnosis.generate_diagnosis",
                side_effect=fake_generate_diagnosis,
            ):

                from ai.graph.nodes.diagnosis import (
                    diagnosis_agent,
                )

                result = diagnosis_agent(
                    state
                )

    assert len(
        result["previous_tickets"]
    ) == 1

    assert (
        result["previous_tickets"][0][
            "ticket_id"
        ]
        == 101
    )

    assert (
        "Historical Ticket Evidence"
        in captured["context"]
    )

    assert (
        "Reinstall the keyboard driver."
        in captured["context"]
    )

    assert len(
        result["conversation_history"]
    ) == 2

    assert (
        result["conversation_history"][0][
            "role"
        ]
        == "user"
    )

    assert (
        result["conversation_history"][1][
            "role"
        ]
        == "assistant"
    )

    assert len(
        result["tool_calls"]
    ) == 1

    assert (
        result["tool_calls"][0]["tool"]
        == "search_memory"
    )

    assert (
        result["metadata"]["memory"][
            "historical_ticket_count"
        ]
        == 1
    )

    mock_db.close.assert_called_once()

def test_diagnosis_agent_continues_when_memory_fails():
    diagnosis_result = type(
        "DiagnosisResult",
        (),
        {
            "problem": "Network issue",
            "possible_root_cause": (
                "Network adapter unavailable."
            ),
            "evidence": [],
            "missing_information": [],
            "confidence": 0.84,
            "model_dump": lambda self: {
                "problem": "Network issue",
                "possible_root_cause": (
                    "Network adapter unavailable."
                ),
                "evidence": [],
                "missing_information": [],
                "confidence": 0.84,
            },
        },
    )()

    state = {
        "ticket_id": 11,
        "cleaned_ticket": (
            "Network disconnected"
        ),
        "category": "network",
        "retrieved_context": (
            "Network troubleshooting evidence."
        ),
        "conversation_history": [],
        "previous_tickets": [],
        "tool_calls": [],
        "fallback_used": False,
        "metadata": {},
    }

    with patch(
        "ai.graph.nodes.diagnosis.SessionLocal"
    ) as mock_session:

        mock_db = mock_session.return_value

        with patch(
            "ai.graph.nodes.diagnosis.execute_tool",
            side_effect=RuntimeError(
                "database unavailable"
            ),
        ):

            with patch(
                "ai.graph.nodes.diagnosis.generate_diagnosis",
                return_value=(
                    diagnosis_result,
                    False,
                ),
            ):

                from ai.graph.nodes.diagnosis import (
                    diagnosis_agent,
                )

                result = diagnosis_agent(
                    state
                )

    assert (
        result["diagnosis"]
        == "Network issue"
    )

    assert (
        result["fallback_used"]
        is False
    )

    assert (
        result["previous_tickets"]
        == []
    )

    assert (
        result["metadata"]["memory"][
            "historical_ticket_count"
        ]
        == 0
    )

    assert (
        result["metadata"]["memory"][
            "memory_error"
        ]
        is not None
    )

    mock_db.close.assert_called_once()