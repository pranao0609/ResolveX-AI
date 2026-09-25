from unittest.mock import MagicMock, patch

from ai.llm.verification_generator import (
    VerificationResult,
    _render_verification_prompt,
    verify_resolution,
)


def test_verification_result_validation():
    result = VerificationResult(
        supported_by_evidence=True,
        hallucination_detected=False,
        complete=True,
        policy_compliant=True,
        resolution_correct=True,
        confidence=0.95,
)

    assert result.verification_passed is True
    assert result.supported_by_evidence is True
    assert result.hallucination_detected is False
    assert result.complete is True
    assert result.policy_compliant is True
    assert result.resolution_correct is True
    assert result.confidence == 0.95
    assert result.verification_reason


def test_verification_result_rejects_invalid_confidence():
    try:
        VerificationResult(
            verification_passed=True,
            verification_reason="Valid evidence.",
            confidence=1.5,
        )
    except Exception:
        return

    raise AssertionError(
        "Invalid confidence should be rejected"
    )


def test_render_verification_prompt():
    prompt = {
        "system_prompt": "Verify the resolution.",
        "user_prompt": (
            "Ticket: {ticket_text}\n"
            "Diagnosis: {diagnosis}\n"
            "Root cause: {root_cause}\n"
            "Steps: {resolution_steps}\n"
            "Context: {context}"
        ),
    }

    system_prompt, user_prompt = _render_verification_prompt(
        prompt,
        ticket_text="Email login failure",
        diagnosis="Authentication failure",
        root_cause="Invalid credentials",
        resolution_steps=[
            "Verify credentials.",
            "Retry login.",
        ],
        context="KB says credentials must be valid.",
    )

    assert system_prompt == "Verify the resolution."

    assert "Email login failure" in user_prompt

    assert "Authentication failure" in user_prompt

    assert "Invalid credentials" in user_prompt

    assert "1. Verify credentials." in user_prompt

    assert "2. Retry login." in user_prompt

    assert "KB says credentials must be valid." in user_prompt


def test_verify_resolution_success():
    response = MagicMock()

    response.content = """
    {
        "supported_by_evidence": true,
        "hallucination_detected": false,
        "complete": true,
        "policy_compliant": true,
        "resolution_correct": true,
        "confidence": 0.94
    }
    """

    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150

    gateway = MagicMock()

    gateway.generate.return_value = response

    prompt = {
        "system_prompt": "Verify.",
        "user_prompt": (
            "{ticket_text} "
            "{diagnosis} "
            "{root_cause} "
            "{resolution_steps} "
            "{context}"
        ),
        "model": "test-model",
        "temperature": 0.1,
    }

    with patch(
        "ai.llm.verification_generator.load_prompt",
        return_value=prompt,
    ), patch(
        "ai.llm.verification_generator._get_gateway",
        return_value=gateway,
    ):
        result, fallback = verify_resolution(
            ticket_text="Email issue",
            diagnosis="Authentication failure",
            root_cause="Invalid credentials",
            resolution_steps=[
                "Verify credentials."
            ],
            context="Credentials must be valid.",
        )

    assert fallback is False

    assert result.supported_by_evidence is True
    assert result.hallucination_detected is False
    assert result.complete is True
    assert result.policy_compliant is True
    assert result.resolution_correct is True

    assert result.confidence == 0.94

    assert result.verification_passed is True

    assert result.verification_reason

    gateway.generate.assert_called_once()

def test_verify_resolution_fallback():
    gateway = MagicMock()

    gateway.generate.side_effect = Exception(
        "Unexpected failure"
    )

    prompt = {
        "system_prompt": "Verify.",
        "user_prompt": "{ticket_text}",
        "model": "test-model",
        "temperature": 0.1,
    }

    with patch(
        "ai.llm.verification_generator.load_prompt",
        return_value=prompt,
    ), patch(
        "ai.llm.verification_generator._get_gateway",
        return_value=gateway,
    ):
        result, fallback = verify_resolution(
            ticket_text="Email issue",
            diagnosis="Authentication failure",
            root_cause="Unknown",
            resolution_steps=[
                "Investigate."
            ],
            context="",
        )

    assert fallback is True

    assert result.verification_passed is False

    assert result.confidence == 0.0