from unittest.mock import patch

from ai.graph.nodes import diagnosis_agent, resolution_agent
from ai.llm.schemas import DiagnosisResult


def test_diagnosis_to_resolution_handoff():
    diagnosis_result = DiagnosisResult(
        problem="Email authentication failure",
        possible_root_cause="Invalid credentials",
        evidence=["Users must use valid credentials."],
        missing_information=[],
        confidence=0.88,
    )

    resolution_result = type(
        "ResolutionResult",
        (),
        {
            "diagnosis": "Email authentication failure",
            "root_cause": "Invalid credentials",
            "resolution_steps": [
                "Verify the account credentials.",
                "Retry authentication.",
            ],
            "evidence": ["Users must use valid credentials."],
            "confidence": 0.91,
            "requires_human": False,
        },
    )()

    state = {
        "ticket_id": 301,
        "cleaned_ticket": "Unable to login to email",
        "retrieved_context": ("Users must use valid credentials."),
        "category": "authentication",
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(diagnosis_result, False),
    ):
        diagnosis_output = diagnosis_agent(state)

    state.update(diagnosis_output)

    assert state["diagnosis_problem"] == ("Email authentication failure")

    assert state["diagnosis_root_cause"] == ("Invalid credentials")

    assert state["diagnosis_evidence"] == ["Users must use valid credentials."]

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(resolution_result, False),
    ) as mock_generate:
        resolution_output = resolution_agent(state)

    mock_generate.assert_called_once_with(
        ticket_text="Unable to login to email",
        context="Users must use valid credentials.",
        diagnosis="Email authentication failure",
        root_cause="Invalid credentials",
        classification="authentication",
        conversation_history=[
            {
                "role": "user",
                "content": "Unable to login to email",
                "metadata": {
                    "ticket_id": 301,
                },
            },
            {
                "role": "assistant",
                "content": "Email authentication failure",
                "metadata": {
                    "ticket_id": 301,
                    "root_cause": "Invalid credentials",
                    "confidence": 0.88,
                },
            },
        ],
    )

    assert resolution_output["resolution_steps"] == [
        "Verify the account credentials.",
        "Retry authentication.",
    ]

    assert resolution_output["evidence"] == ["Users must use valid credentials."]

    assert resolution_output["resolution_confidence"] == 0.91
    assert resolution_output["requires_human"] is False
