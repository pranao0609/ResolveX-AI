from unittest.mock import patch

from ai.graph.nodes import diagnosis_agent
from ai.llm.schemas import DiagnosisResult


def test_diagnosis_agent_populates_structured_state():
    mock_result = DiagnosisResult(
        problem="Password reset email is not being received",
        possible_root_cause=(
            "The reset email may be filtered or blocked "
            "by the user's email provider."
        ),
        evidence=[
            "Password Reset Email Delivery"
        ],
        missing_information=[
            "Whether the user checked spam or junk folders"
        ],
        confidence=0.87,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ) as mock_generate:

        result = diagnosis_agent(
            {
                "ticket_id": 1,
                "cleaned_ticket": (
                    "I cannot receive my password reset email."
                ),
                "category": "software",
                "retrieved_context": (
                    "Password Reset Email Delivery: "
                    "Check inbox, spam and sender filtering."
                ),
            }
        )

    mock_generate.assert_called_once_with(
        ticket_text=(
            "I cannot receive my password reset email."
        ),
        context=(
            "Password Reset Email Delivery: "
            "Check inbox, spam and sender filtering."
        ),
        classification="software",
    )

    assert result["diagnosis_result"] == {
        "problem": (
            "Password reset email is not being received"
        ),
        "possible_root_cause": (
            "The reset email may be filtered or blocked "
            "by the user's email provider."
        ),
        "evidence": [
            "Password Reset Email Delivery"
        ],
        "missing_information": [
            "Whether the user checked spam or junk folders"
        ],
        "confidence": 0.87,
    }

    assert result["diagnosis_problem"] == (
        "Password reset email is not being received"
    )

    assert result["diagnosis_root_cause"] == (
        "The reset email may be filtered or blocked "
        "by the user's email provider."
    )

    assert result["diagnosis_evidence"] == [
        "Password Reset Email Delivery"
    ]

    assert result["diagnosis_missing_information"] == [
        "Whether the user checked spam or junk folders"
    ]

    assert result["diagnosis_confidence"] == 0.87

    # Backward compatibility
    assert result["diagnosis"] == (
        "Password reset email is not being received"
    )

    assert result["root_cause"] == (
        "The reset email may be filtered or blocked "
        "by the user's email provider."
    )

    assert result["fallback_used"] is False


def test_diagnosis_agent_handles_fallback():
    mock_result = DiagnosisResult(
        problem=(
            "The reported support issue requires "
            "further investigation."
        ),
        possible_root_cause=(
            "No validated root cause could be established "
            "from the available evidence."
        ),
        evidence=[],
        missing_information=[
            "Additional diagnostic information is required."
        ],
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
                "category": "software",
                "retrieved_context": "",
            }
        )

    assert result["diagnosis_confidence"] == 0.0
    assert result["diagnosis_evidence"] == []
    assert result["diagnosis_missing_information"] == [
        "Additional diagnostic information is required."
    ]
    assert result["fallback_used"] is True


def test_diagnosis_agent_preserves_existing_fallback():
    mock_result = DiagnosisResult(
        problem="Authentication failure",
        possible_root_cause="Invalid credentials",
        evidence=["Authentication policy"],
        missing_information=[],
        confidence=0.88,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ):

        result = diagnosis_agent(
            {
                "ticket_id": 3,
                "cleaned_ticket": "Unable to log in",
                "category": "authentication",
                "retrieved_context": (
                    "Authentication policy requires valid credentials."
                ),
                "fallback_used": True,
            }
        )

    assert result["fallback_used"] is True
    assert result["diagnosis_confidence"] == 0.88