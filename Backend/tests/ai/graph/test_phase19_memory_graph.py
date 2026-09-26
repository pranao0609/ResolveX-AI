from unittest.mock import patch

from ai.graph.graph import resolvex_graph


def test_graph_contains_memory_compatible_nodes():
    nodes = resolvex_graph.nodes

    assert "initialize_state" in nodes
    assert "ticket_analyzer" in nodes
    assert "retrieval_agent" in nodes
    assert "diagnosis_agent" in nodes
    assert "resolution_agent" in nodes
    assert "verification_agent" in nodes
    assert "decision_agent" in nodes


def test_diagnosis_memory_output_is_resolution_compatible(
    monkeypatch,
):
    from ai.graph.nodes.diagnosis import (
        diagnosis_agent,
    )

    diagnosis_result = type(
        "DiagnosisResult",
        (),
        {
            "problem": ("Keyboard hardware failure"),
            "possible_root_cause": ("Keyboard driver failure"),
            "evidence": ["Historical ticket matched"],
            "missing_information": [],
            "confidence": 0.91,
            "model_dump": lambda self: {
                "problem": ("Keyboard hardware failure"),
                "possible_root_cause": ("Keyboard driver failure"),
                "evidence": ["Historical ticket matched"],
                "missing_information": [],
                "confidence": 0.91,
            },
        },
    )()

    monkeypatch.setattr(
        "ai.graph.nodes.diagnosis.SessionLocal",
        lambda: type(
            "FakeSession",
            (),
            {
                "close": lambda self: None,
            },
        )(),
    )

    monkeypatch.setattr(
        "ai.graph.nodes.diagnosis.execute_tool",
        lambda *args, **kwargs: (
            {
                "tool": "search_memory",
                "status": "success",
                "data": [
                    {
                        "ticket_id": 400,
                        "title": "Keyboard failure",
                        "description": ("Keyboard stopped responding."),
                        "category": "hardware",
                        "status": "closed",
                        "solution": ("Reinstall keyboard driver."),
                        "confidence": 0.93,
                        "explanation": None,
                        "submitted_by": "user400",
                        "created_at": None,
                        "updated_at": None,
                        "metadata": {},
                    }
                ],
                "result_count": 1,
                "latency_ms": 0.1,
            },
            {
                "agent": "diagnosis_agent",
                "tool": "search_memory",
                "category": "read",
                "input": {},
                "result_count": 1,
                "status": "success",
                "latency_ms": 0.1,
                "error": None,
            },
        ),
    )

    monkeypatch.setattr(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        lambda **kwargs: (
            diagnosis_result,
            False,
        ),
    )

    state = {
        "ticket_id": 500,
        "cleaned_ticket": ("Laptop keyboard stopped working"),
        "category": "hardware",
        "retrieved_context": ("Keyboard troubleshooting evidence."),
        "conversation_history": [],
        "previous_tickets": [],
        "tool_calls": [],
        "fallback_used": False,
        "metadata": {},
    }

    result = diagnosis_agent(state)

    # Diagnosis contract
    assert result["diagnosis_problem"] == "Keyboard hardware failure"

    assert result["diagnosis_root_cause"] == "Keyboard driver failure"

    assert result["diagnosis_confidence"] == 0.91

    # Memory contract
    assert len(result["previous_tickets"]) == 1

    assert result["previous_tickets"][0]["ticket_id"] == 400

    assert len(result["conversation_history"]) == 2

    assert len(result["tool_calls"]) == 1

    # Observability contract
    assert result["metadata"]["memory"]["historical_ticket_count"] == 1
