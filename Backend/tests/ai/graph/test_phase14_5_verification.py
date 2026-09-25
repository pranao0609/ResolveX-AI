from unittest.mock import patch

from ai.graph.nodes import verification_agent


def test_verification_agent_passes_supported_resolution():
    mock_result = type(
        "VerificationResult",
        (),
        {
            "verification_passed": True,
            "verification_reason": (
                "The proposed resolution is directly "
                "supported by the knowledge-base evidence."
            ),
            "confidence": 0.93,
            "supported_by_evidence": True,
            "hallucination_detected": False,
            "complete": True,
            "policy_compliant": True,
            "resolution_correct": True,
            "model_dump": lambda self: {
                "supported_by_evidence": True,
                "hallucination_detected": False,
                "complete": True,
                "policy_compliant": True,
                "resolution_correct": True,
                "confidence": 0.93,
                "verification_passed": True,
                "verification_reason": (
                    "The proposed resolution is directly "
                    "supported by the knowledge-base evidence."
                ),
            },
        },
    )()

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ):
        result = verification_agent(
            {
                "ticket_id": 1,
                "cleaned_ticket": "Unable to access email",
                "diagnosis": "Email authentication failure",
                "root_cause": "Invalid credentials",
                "resolution_steps": [
                    "Verify the account credentials.",
                    "Retry authentication.",
                ],
                "retrieved_context": (
                    "Users must use valid credentials."
                ),
                "fallback_used": False,
            }
        )

    assert result["verification_passed"] is True

    assert result["verification_confidence"] == 0.93

    assert (
        "supported"
        in result["verification_reason"]
    )

    assert result["verification_supported_by_evidence"] is True

    assert result["verification_hallucination_detected"] is False

    assert result["verification_complete"] is True

    assert result["verification_policy_compliant"] is True

    assert result["verification_resolution_correct"] is True

    assert result["verification_result"]["verification_passed"] is True

    assert result["fallback_used"] is False


def test_verification_agent_rejects_unsupported_resolution():
    mock_result = type(
        "VerificationResult",
        (),
        {
            "verification_passed": False,
            "verification_reason": (
                "The proposed resolution is not sufficiently "
                "supported by the available evidence."
            ),
            "confidence": 0.89,
            "supported_by_evidence": False,
            "hallucination_detected": False,
            "complete": True,
            "policy_compliant": True,
            "resolution_correct": False,
            "model_dump": lambda self: {
                "supported_by_evidence": False,
                "hallucination_detected": False,
                "complete": True,
                "policy_compliant": True,
                "resolution_correct": False,
                "confidence": 0.89,
                "verification_passed": False,
                "verification_reason": (
                    "The proposed resolution is not sufficiently "
                    "supported by the available evidence."
                ),
            },
        },
    )()

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ):
        result = verification_agent(
            {
                "ticket_id": 2,
                "cleaned_ticket": "Email is not working",
                "diagnosis": "Email issue",
                "root_cause": "Unknown",
                "resolution_steps": [
                    "Replace the mail server."
                ],
                "retrieved_context": (
                    "No evidence about mail server replacement."
                ),
                "fallback_used": False,
            }
        )

    assert result["verification_passed"] is False

    assert result["verification_confidence"] == 0.89

    assert result["verification_supported_by_evidence"] is False

    assert result["verification_hallucination_detected"] is False

    assert result["verification_complete"] is True

    assert result["verification_policy_compliant"] is True

    assert result["verification_resolution_correct"] is False

    assert result["verification_result"]["verification_passed"] is False

    assert result["fallback_used"] is False


def test_verification_agent_fallback_fails_closed():
    mock_result = type(
        "VerificationResult",
        (),
        {
            "verification_passed": False,
            "verification_reason": (
                "Verification could not be completed."
            ),
            "confidence": 0.0,
            "supported_by_evidence": False,
            "hallucination_detected": True,
            "complete": False,
            "policy_compliant": False,
            "resolution_correct": False,
            "model_dump": lambda self: {
                "supported_by_evidence": False,
                "hallucination_detected": True,
                "complete": False,
                "policy_compliant": False,
                "resolution_correct": False,
                "confidence": 0.0,
                "verification_passed": False,
                "verification_reason": (
                    "Verification could not be completed."
                ),
            },
        },
    )()

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, True),
    ):
        result = verification_agent(
            {
                "ticket_id": 3,
                "cleaned_ticket": "System issue",
                "diagnosis": "Unknown issue",
                "root_cause": "Unknown",
                "resolution_steps": [
                    "Investigate the issue."
                ],
                "retrieved_context": "",
                "fallback_used": False,
            }
        )

    assert result["verification_passed"] is False

    assert result["verification_confidence"] == 0.0

    assert result["verification_supported_by_evidence"] is False

    assert result["verification_hallucination_detected"] is True

    assert result["verification_complete"] is False

    assert result["verification_policy_compliant"] is False

    assert result["verification_resolution_correct"] is False

    assert result["verification_result"]["verification_passed"] is False

    assert result["fallback_used"] is True


def test_verification_agent_preserves_previous_fallback():
    mock_result = type(
        "VerificationResult",
        (),
        {
            "verification_passed": True,
            "verification_reason": (
                "Evidence supports the resolution."
            ),
            "confidence": 0.90,
            "supported_by_evidence": True,
            "hallucination_detected": False,
            "complete": True,
            "policy_compliant": True,
            "resolution_correct": True,
            "model_dump": lambda self: {
                "supported_by_evidence": True,
                "hallucination_detected": False,
                "complete": True,
                "policy_compliant": True,
                "resolution_correct": True,
                "confidence": 0.90,
                "verification_passed": True,
                "verification_reason": (
                    "Evidence supports the resolution."
                ),
            },
        },
    )()

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ):
        result = verification_agent(
            {
                "ticket_id": 4,
                "cleaned_ticket": "Email issue",
                "diagnosis": "Email issue",
                "root_cause": "Credential issue",
                "resolution_steps": [
                    "Verify credentials."
                ],
                "retrieved_context": (
                    "Verify credentials before retrying."
                ),
                "fallback_used": True,
            }
        )

    assert result["verification_passed"] is True

    assert result["verification_confidence"] == 0.90

    assert result["verification_supported_by_evidence"] is True

    assert result["verification_hallucination_detected"] is False

    assert result["verification_complete"] is True

    assert result["verification_policy_compliant"] is True

    assert result["verification_resolution_correct"] is True

    assert result["verification_result"]["verification_passed"] is True

    # Previous fallback state must not be lost.
    assert result["fallback_used"] is True