from unittest.mock import patch

from ai.graph.nodes import (
    diagnosis_agent,
    resolution_agent,
)
from ai.llm.schemas import DiagnosisResult


def test_diagnosis_agent_populates_state():
    mock_result = DiagnosisResult(
        problem="Email authentication failure",
        possible_root_cause="Invalid credentials",
        evidence=["Users must use valid credentials."],
        missing_information=[],
        confidence=0.88,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 1,
                "cleaned_ticket": "Unable to login to email",
                "retrieved_context": ("Users must use valid credentials."),
            }
        )

    assert result["diagnosis"] == ("Email authentication failure")

    assert result["root_cause"] == ("Invalid credentials")

    assert result["diagnosis_confidence"] == 0.88

    assert result["fallback_used"] is False

    assert result["diagnosis_problem"] == ("Email authentication failure")

    assert result["diagnosis_root_cause"] == ("Invalid credentials")

    assert result["diagnosis_evidence"] == ["Users must use valid credentials."]

    assert result["diagnosis_missing_information"] == []


def test_diagnosis_agent_handles_fallback():
    mock_result = DiagnosisResult(
        problem=("The reported support issue requires " "further investigation."),
        possible_root_cause=(
            "No validated root cause could be "
            "established from the available evidence."
        ),
        evidence=[],
        missing_information=["Additional diagnostic information is required."],
        confidence=0.0,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, True),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 2,
                "cleaned_ticket": "System issue",
                "retrieved_context": "",
            }
        )

    assert result["diagnosis_confidence"] == 0.0
    assert result["fallback_used"] is True


def test_resolution_agent_populates_state():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "resolution_steps": [
                "Verify the account credentials.",
                "Retry authentication.",
            ],
            "evidence": ["KB: Users must use valid credentials."],
            "confidence": 0.91,
            "requires_human": False,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, False),
    ):
        result = resolution_agent(
            {
                "ticket_id": 3,
                "cleaned_ticket": "Unable to login",
                "retrieved_context": ("Users must use valid credentials."),
                "fallback_used": False,
            }
        )

    assert result["resolution_steps"] == [
        "Verify the account credentials.",
        "Retry authentication.",
    ]

    assert result["evidence"] == ["KB: Users must use valid credentials."]

    assert result["resolution_confidence"] == 0.91

    assert result["requires_human"] is False

    assert result["fallback_used"] is False


def test_resolution_agent_handles_fallback():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "resolution_steps": [
                "Review the ticket details.",
                "Collect relevant logs.",
                "Escalate the ticket.",
            ],
            "evidence": [],
            "confidence": 0.0,
            "requires_human": True,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, True),
    ):
        result = resolution_agent(
            {
                "ticket_id": 4,
                "cleaned_ticket": "System issue",
                "retrieved_context": "",
                "fallback_used": False,
            }
        )

    assert result["resolution_confidence"] == 0.0
    assert result["requires_human"] is True
    assert result["fallback_used"] is True


def test_resolution_agent_preserves_previous_fallback():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "resolution_steps": [
                "Review the ticket.",
            ],
            "evidence": [],
            "confidence": 0.75,
            "requires_human": False,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, False),
    ):
        result = resolution_agent(
            {
                "ticket_id": 5,
                "cleaned_ticket": "Email issue",
                "retrieved_context": "Email KB",
                "fallback_used": True,
            }
        )

    assert result["fallback_used"] is True
