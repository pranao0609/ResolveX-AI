from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from ai.llm.errors import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ai.llm.gateway import LLMGateway
from ai.llm.providers.base import (
    LLMResponse,
    LLMUsage,
)
from ai.llm.providers.groq_provider import GroqProvider
from ai.llm.rate_limiter import RateLimiter
from ai.llm.schemas import ResolutionResult
from ai.llm.solution_generator import (
    generate_solution,
)


def make_llm_response(
    content: str | None = None,
    model: str = "test-model",
) -> LLMResponse:
    """Create a deterministic provider response for tests."""

    if content is None:
        content = (
            '{"diagnosis":"Email authentication failure",'
            '"root_cause":"Invalid credentials",'
            '"resolution_steps":'
            '["Verify credentials","Reset password"],'
            '"evidence":["Authentication logs"],'
            '"confidence":0.9,'
            '"requires_human":false}'
        )

    return LLMResponse(
        content=content,
        usage=LLMUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
        model=model,
        raw_response={"test": True},
    )


def make_gateway(
    provider: Mock,
) -> LLMGateway:
    """Create a gateway with deterministic test configuration."""

    return LLMGateway(
        provider=provider,
        rate_limiter=RateLimiter(
            max_requests=100,
            window_seconds=60,
        ),
    )


def test_successful_gateway_request():
    provider = Mock()

    provider.generate.return_value = make_llm_response()

    gateway = make_gateway(provider)

    response = gateway.generate(
        system_prompt="system",
        user_prompt="user",
        model="test-model",
        max_retries=3,
    )

    assert response.content

    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 50
    assert response.usage.total_tokens == 150

    assert provider.generate.call_count == 1

    assert gateway.last_usage is not None
    assert gateway.last_usage.model == "test-model"
    assert gateway.last_usage.prompt_tokens == 100
    assert gateway.last_usage.completion_tokens == 50
    assert gateway.last_usage.total_tokens == 150
    assert gateway.last_usage.success is True


def test_timeout_retries_then_succeeds():
    provider = Mock()

    provider.generate.side_effect = [
        LLMTimeoutError("request timed out"),
        make_llm_response(),
    ]

    gateway = make_gateway(provider)

    with patch("ai.llm.gateway.time.sleep") as sleep_mock:
        response = gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=2,
        )

    assert response.content
    assert provider.generate.call_count == 2
    sleep_mock.assert_called_once_with(1)

    assert gateway.last_usage is not None
    assert gateway.last_usage.success is True


def test_rate_limit_retries_then_succeeds():
    provider = Mock()

    provider.generate.side_effect = [
        LLMRateLimitError("rate limit exceeded"),
        make_llm_response(),
    ]

    gateway = make_gateway(provider)

    with patch("ai.llm.gateway.time.sleep") as sleep_mock:
        response = gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=2,
        )

    assert response.content
    assert provider.generate.call_count == 2
    sleep_mock.assert_called_once_with(1)

    assert gateway.last_usage is not None
    assert gateway.last_usage.success is True


def test_provider_error_retries_until_exhausted():
    provider = Mock()

    provider.generate.side_effect = LLMProviderError("temporary provider failure")

    gateway = make_gateway(provider)

    with patch("ai.llm.gateway.time.sleep") as sleep_mock:
        with pytest.raises(LLMProviderError):
            gateway.generate(
                system_prompt="system",
                user_prompt="user",
                model="test-model",
                max_retries=3,
            )

    assert provider.generate.call_count == 3

    assert sleep_mock.call_count == 2

    assert sleep_mock.call_args_list[0].args == (1,)
    assert sleep_mock.call_args_list[1].args == (2,)

    assert gateway.last_usage is not None
    assert gateway.last_usage.success is False
    assert gateway.last_usage.error_type == "LLMProviderError"


def test_authentication_error_is_not_retried():
    provider = Mock()

    provider.generate.side_effect = LLMAuthenticationError("invalid API key")

    gateway = make_gateway(provider)

    with pytest.raises(LLMAuthenticationError):
        gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=3,
        )

    assert provider.generate.call_count == 1

    assert gateway.last_usage is not None
    assert gateway.last_usage.success is False
    assert gateway.last_usage.error_type == "LLMAuthenticationError"


def test_invalid_response_is_not_retried():
    provider = Mock()

    provider.generate.return_value = LLMResponse(
        content="not valid resolution JSON",
        usage=LLMUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        ),
        model="test-model",
    )

    gateway = make_gateway(provider)

    response = gateway.generate(
        system_prompt="system",
        user_prompt="user",
        model="test-model",
        max_retries=3,
    )

    with pytest.raises(LLMInvalidResponseError):
        gateway.validate_resolution(response)

    # Provider was called exactly once.
    # Validation happens after generation and does not
    # trigger another provider request.
    assert provider.generate.call_count == 1


def test_gateway_rate_limit_applies_once_per_logical_request():
    provider = Mock()

    provider.generate.side_effect = [
        LLMTimeoutError("timeout"),
        make_llm_response(),
    ]

    rate_limiter = RateLimiter(
        max_requests=1,
        window_seconds=60,
    )

    gateway = LLMGateway(
        provider=provider,
        rate_limiter=rate_limiter,
    )

    with patch("ai.llm.gateway.time.sleep"):
        response = gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=2,
        )

    assert response.content
    assert provider.generate.call_count == 2


def test_gateway_validates_resolution():
    provider = Mock()

    response = make_llm_response()

    gateway = make_gateway(provider)

    result = gateway.validate_resolution(response)

    assert isinstance(
        result,
        ResolutionResult,
    )

    assert result.confidence == 0.9
    assert result.requires_human is False


def test_solution_generator_uses_gateway_and_returns_resolution():
    resolution = ResolutionResult(
        diagnosis="Email authentication failure",
        root_cause="Invalid credentials",
        resolution_steps=[
            "Verify credentials",
            "Reset password",
        ],
        evidence=["Authentication logs"],
        confidence=0.9,
        requires_human=False,
    )

    mock_gateway = Mock()

    mock_gateway.generate.return_value = make_llm_response()

    mock_gateway.validate_resolution.return_value = resolution

    with patch(
        "ai.llm.solution_generator._get_gateway",
        return_value=mock_gateway,
    ):
        result, fallback_used = generate_solution(
            ticket_text="Unable to access email",
            context="Email authentication troubleshooting",
        )

    assert fallback_used is False
    assert result == resolution

    mock_gateway.generate.assert_called_once()
    mock_gateway.validate_resolution.assert_called_once()


def test_solution_generator_falls_back_after_gateway_failure():
    mock_gateway = Mock()

    mock_gateway.generate.side_effect = LLMTimeoutError("LLM timeout")

    with patch(
        "ai.llm.solution_generator._get_gateway",
        return_value=mock_gateway,
    ):
        result, fallback_used = generate_solution(
            ticket_text="Unable to access email",
            context="Email troubleshooting",
        )

    assert fallback_used is True
    assert result.requires_human is True
    assert result.confidence == 0.0
    assert result.evidence == []

    mock_gateway.generate.assert_called_once()


def test_groq_provider_maps_timeout_error():
    fake_client = Mock()

    fake_client.chat.completions.create.side_effect = Exception("Request timed out")

    with patch(
        "ai.llm.providers.groq_provider.Groq",
        return_value=fake_client,
    ):
        provider = GroqProvider(
            api_key="test-api-key",
        )

        with pytest.raises(LLMTimeoutError):
            provider.generate(
                system_prompt="system",
                user_prompt="user",
                model="test-model",
                temperature=0.3,
                max_tokens=100,
                timeout=10,
            )


def test_groq_provider_maps_rate_limit_error():
    fake_client = Mock()

    fake_client.chat.completions.create.side_effect = Exception(
        "429 rate limit exceeded"
    )

    with patch(
        "ai.llm.providers.groq_provider.Groq",
        return_value=fake_client,
    ):
        provider = GroqProvider(
            api_key="test-api-key",
        )

        with pytest.raises(LLMRateLimitError):
            provider.generate(
                system_prompt="system",
                user_prompt="user",
                model="test-model",
                temperature=0.3,
                max_tokens=100,
                timeout=10,
            )


def test_groq_provider_maps_authentication_error():
    fake_client = Mock()

    fake_client.chat.completions.create.side_effect = Exception(
        "401 authentication failed"
    )

    with patch(
        "ai.llm.providers.groq_provider.Groq",
        return_value=fake_client,
    ):
        provider = GroqProvider(
            api_key="test-api-key",
        )

        with pytest.raises(LLMAuthenticationError):
            provider.generate(
                system_prompt="system",
                user_prompt="user",
                model="test-model",
                temperature=0.3,
                max_tokens=100,
                timeout=10,
            )


def test_groq_provider_maps_unknown_error():
    fake_client = Mock()

    fake_client.chat.completions.create.side_effect = Exception(
        "unexpected provider failure"
    )

    with patch(
        "ai.llm.providers.groq_provider.Groq",
        return_value=fake_client,
    ):
        provider = GroqProvider(
            api_key="test-api-key",
        )

        with pytest.raises(LLMProviderError):
            provider.generate(
                system_prompt="system",
                user_prompt="user",
                model="test-model",
                temperature=0.3,
                max_tokens=100,
                timeout=10,
            )


def test_groq_provider_maps_successful_response():
    fake_completion = Mock()

    fake_completion.choices = [Mock(message=Mock(content="test response"))]

    fake_completion.usage = Mock(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    fake_client = Mock()

    fake_client.chat.completions.create.return_value = fake_completion

    with patch(
        "ai.llm.providers.groq_provider.Groq",
        return_value=fake_client,
    ):
        provider = GroqProvider(
            api_key="test-api-key",
        )

        response = provider.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            temperature=0.3,
            max_tokens=100,
            timeout=10,
        )

    assert response.content == "test response"

    assert response.model == "test-model"

    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 50
    assert response.usage.total_tokens == 150

    fake_client.chat.completions.create.assert_called_once()
