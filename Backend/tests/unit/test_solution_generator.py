import json

from ai.llm.solution_generator import _parse_resolution
from ai.llm.schemas import ResolutionResult


def test_parse_valid_structured_response():
    response = json.dumps(
        {
            "diagnosis": "Email client is not synchronizing.",
            "root_cause": "The synchronization service is unavailable.",
            "resolution_steps": [
                "Restart the email client.",
                "Check the synchronization service.",
                "Retry synchronization.",
            ],
            "evidence": [
                "Email Client Not Syncing",
            ],
            "confidence": 0.91,
            "requires_human": False,
        }
    )

    result = _parse_resolution(response)

    assert isinstance(result, ResolutionResult)
    assert result.diagnosis == "Email client is not synchronizing."
    assert result.root_cause == (
        "The synchronization service is unavailable."
    )
    assert len(result.resolution_steps) == 3
    assert result.confidence == 0.91
    assert result.requires_human is False


def test_parse_markdown_wrapped_json():
    response = """```json
{
    "diagnosis": "Password reset email is delayed.",
    "root_cause": "Email delivery service delay.",
    "resolution_steps": [
        "Check spam folder.",
        "Retry password reset."
    ],
    "evidence": [
        "Password Reset Email Delivery"
    ],
    "confidence": 0.88,
    "requires_human": false
}
```"""

    result = _parse_resolution(response)

    assert isinstance(result, ResolutionResult)
    assert result.confidence == 0.88


def test_parse_invalid_json_raises():
    response = "This is not valid JSON."

    try:
        _parse_resolution(response)
        assert False, "Expected parsing to fail"
    except Exception:
        pass


def test_parse_invalid_confidence_raises():
    response = json.dumps(
        {
            "diagnosis": "Test diagnosis",
            "root_cause": "Test root cause",
            "resolution_steps": [
                "Test step",
            ],
            "evidence": [],
            "confidence": 1.5,
            "requires_human": False,
        }
    )

    try:
        _parse_resolution(response)
        assert False, "Expected validation to fail"
    except Exception:
        pass

from unittest.mock import patch


def test_default_prompt_version_is_v1():
    with patch(
        "ai.llm.solution_generator.load_prompt"
    ) as mock_load_prompt:

        mock_load_prompt.return_value = {
            "prompt_id": "resolution_prompt_v1",
            "version": 1,
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.3,
            "created_at": "2026-09-24",
            "system_prompt": "system",
            "user_prompt": "{ticket_text}\n{context}",
        }

        with patch(
            "ai.llm.solution_generator.GROQ_AVAILABLE",
            False,
        ):
            from ai.llm.solution_generator import (
                generate_solution,
            )

            result, fallback = generate_solution(
                "Test ticket",
                "Test context",
            )

        mock_load_prompt.assert_called_once_with(
            "resolution",
            1,
        )

        assert fallback is True
        assert result.requires_human is True


def test_custom_prompt_version_is_loaded():
    with patch(
        "ai.llm.solution_generator.load_prompt"
    ) as mock_load_prompt:

        mock_load_prompt.return_value = {
            "prompt_id": "resolution_prompt_v3",
            "version": 3,
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.3,
            "created_at": "2026-09-24",
            "system_prompt": "system",
            "user_prompt": "{ticket_text}\n{context}",
        }

        with patch(
            "ai.llm.solution_generator.GROQ_AVAILABLE",
            False,
        ):
            from ai.llm.solution_generator import (
                generate_solution,
            )

            result, fallback = generate_solution(
                "Test ticket",
                "Test context",
                prompt_version=3,
            )

        mock_load_prompt.assert_called_once_with(
            "resolution",
            3,
        )

        assert fallback is True
        assert result.requires_human is True