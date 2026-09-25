import pytest

from ai.llm.errors import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ai.llm.solution_generator import (
    _fallback_reason,
    _placeholder_solution,
)


@pytest.mark.parametrize(
    "error,expected_reason",
    [
        (
            LLMAuthenticationError("invalid key"),
            "authentication_error",
        ),
        (
            LLMTimeoutError("request timed out"),
            "timeout",
        ),
        (
            LLMRateLimitError("rate limited"),
            "rate_limit",
        ),
        (
            LLMInvalidResponseError("invalid JSON"),
            "invalid_response",
        ),
    ],
)
def test_fallback_reason(error, expected_reason):
    assert _fallback_reason(error) == expected_reason


def test_placeholder_solution_is_safe():
    result = _placeholder_solution()

    assert result.confidence == 0.0
    assert result.requires_human is True

    assert result.evidence == []

    assert len(result.resolution_steps) >= 1

    assert "validated" in result.root_cause.lower()


def test_placeholder_solution_does_not_claim_specific_root_cause():
    result = _placeholder_solution()

    assert "validated root cause" in result.root_cause.lower()
    assert result.confidence == 0.0
    assert result.requires_human is True

from unittest.mock import patch

from ai.llm.errors import LLMTimeoutError
from ai.llm.solution_generator import generate_solution


def test_generate_solution_uses_safe_fallback_on_llm_failure():
    with patch(
        "ai.llm.solution_generator._get_gateway"
    ) as mock_get_gateway:

        gateway = mock_get_gateway.return_value

        gateway.generate.side_effect = LLMTimeoutError(
            "LLM request timed out"
        )

        resolution, fallback_used = generate_solution(
            ticket_text="Unable to login to email",
            context="Email authentication troubleshooting",
        )

    assert fallback_used is True
    assert resolution.requires_human is True
    assert resolution.confidence == 0.0
    assert resolution.evidence == []