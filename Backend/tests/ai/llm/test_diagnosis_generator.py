from unittest.mock import MagicMock, patch

from ai.llm.diagnosis_generator import (
    _placeholder_diagnosis,
    generate_diagnosis,
)
from ai.llm.schemas import DiagnosisResult


def test_placeholder_diagnosis_is_safe():
    result = _placeholder_diagnosis()

    assert isinstance(result, DiagnosisResult)
    assert result.confidence == 0.0
    assert result.evidence == []
    assert result.missing_information
    assert "further investigation" in result.problem.lower()


def test_generate_diagnosis_valid_response():
    response = MagicMock()

    response.content = """
    {
      "problem": "Password reset email is not received",
      "possible_root_cause": "Email filtering or delivery blocking",
      "evidence": [
        "Password Reset Email Delivery"
      ],
      "missing_information": [
        "Whether spam folder was checked"
      ],
      "confidence": 0.87
    }
    """

    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 80
    response.usage.total_tokens = 180

    gateway = MagicMock()
    gateway.generate.return_value = response

    with (
        patch(
            "ai.llm.diagnosis_generator._get_gateway",
            return_value=gateway,
        ),
        patch(
            "ai.llm.diagnosis_generator.load_prompt",
            return_value={
                "prompt_id": "diagnosis_prompt_v1",
                "version": 1,
                "model": "test-model",
                "temperature": 0.2,
                "created_at": "2026-09-25",
                "system_prompt": "You are a diagnosis agent.",
                "user_prompt": ("Ticket:\n" "{ticket_text}\n" "Context:\n" "{context}"),
            },
        ),
    ):

        result, fallback = generate_diagnosis(
            ticket_text="Cannot receive password reset email",
            context="Password Reset Email Delivery",
            classification="software",
        )

    assert fallback is False
    assert result.problem == ("Password reset email is not received")
    assert result.possible_root_cause == ("Email filtering or delivery blocking")
    assert result.evidence == ["Password Reset Email Delivery"]
    assert result.missing_information == ["Whether spam folder was checked"]
    assert result.confidence == 0.87

    gateway.generate.assert_called_once()

    call = gateway.generate.call_args
    user_prompt = call.kwargs["user_prompt"]

    assert "Cannot receive password reset email" in user_prompt
    assert "Password Reset Email Delivery" in user_prompt
    assert "software" in user_prompt


def test_generate_diagnosis_gateway_failure_uses_fallback():
    gateway = MagicMock()

    gateway.generate.side_effect = RuntimeError("provider failure")

    with (
        patch(
            "ai.llm.diagnosis_generator._get_gateway",
            return_value=gateway,
        ),
        patch(
            "ai.llm.diagnosis_generator.load_prompt",
            return_value={
                "prompt_id": "diagnosis_prompt_v1",
                "version": 1,
                "model": "test-model",
                "temperature": 0.2,
                "created_at": "2026-09-25",
                "system_prompt": "Diagnosis agent.",
                "user_prompt": ("{ticket_text}\n{context}"),
            },
        ),
    ):

        result, fallback = generate_diagnosis(
            ticket_text="System issue",
            context="No evidence",
            classification="software",
        )

    assert fallback is True
    assert result.confidence == 0.0
    assert result.evidence == []
