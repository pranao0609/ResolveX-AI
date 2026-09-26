from unittest.mock import Mock, patch

import pytest

from ai.llm.errors import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMTimeoutError,
)
from ai.llm.gateway import LLMGateway
from ai.llm.providers.base import LLMResponse, LLMUsage
from ai.llm.rate_limiter import RateLimiter


def make_response() -> LLMResponse:
    return LLMResponse(
        content='{"diagnosis":"test"}',
        usage=LLMUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        model="test-model",
        raw_response={},
    )


def make_gateway(provider):
    return LLMGateway(
        provider=provider,
        rate_limiter=RateLimiter(
            max_requests=100,
            window_seconds=60,
        ),
    )


def test_timeout_is_retried_then_succeeds():
    provider = Mock()

    provider.generate.side_effect = [
        LLMTimeoutError("timeout"),
        make_response(),
    ]

    gateway = make_gateway(provider)

    with patch("ai.llm.gateway.time.sleep"):
        response = gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=2,
        )

    assert response.content == '{"diagnosis":"test"}'
    assert provider.generate.call_count == 2


def test_provider_error_is_retried_then_succeeds():
    provider = Mock()

    provider.generate.side_effect = [
        LLMProviderError("temporary provider error"),
        make_response(),
    ]

    gateway = make_gateway(provider)

    with patch("ai.llm.gateway.time.sleep"):
        response = gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=2,
        )

    assert response.content == '{"diagnosis":"test"}'
    assert provider.generate.call_count == 2


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


def test_retry_exhaustion_raises_last_error():
    provider = Mock()

    provider.generate.side_effect = LLMTimeoutError("timeout")

    gateway = make_gateway(provider)

    with patch("ai.llm.gateway.time.sleep"):
        with pytest.raises(LLMTimeoutError):
            gateway.generate(
                system_prompt="system",
                user_prompt="user",
                model="test-model",
                max_retries=3,
            )

    assert provider.generate.call_count == 3


def test_retry_does_not_consume_extra_rate_limit_slot():
    provider = Mock()

    provider.generate.side_effect = [
        LLMTimeoutError("timeout"),
        make_response(),
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

    assert response.content == '{"diagnosis":"test"}'
    assert provider.generate.call_count == 2
