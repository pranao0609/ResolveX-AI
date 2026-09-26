from types import SimpleNamespace

import pytest

from ai.llm.errors import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMTimeoutError,
)

from ai.llm.gateway import LLMGateway

from ai.llm.providers.base import (
    LLMResponse,
    LLMUsage,
)
from ai.llm.errors import LLMRateLimitError
from ai.llm.rate_limiter import RateLimiter


class FakeProvider:
    def __init__(
        self,
        response=None,
        errors=None,
    ):
        self.response = response
        self.errors = list(errors or [])
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1

        if self.errors:
            error = self.errors.pop(0)
            raise error

        return self.response


def make_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        usage=LLMUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )


def test_gateway_returns_provider_response():
    provider = FakeProvider(response=make_response('{"diagnosis":"Test issue"}'))

    gateway = LLMGateway(provider)

    response = gateway.generate(
        system_prompt="system",
        user_prompt="user",
        model="test-model",
        max_retries=1,
    )

    assert response.content == ('{"diagnosis":"Test issue"}')

    assert provider.calls == 1


def test_gateway_retries_timeout(monkeypatch):
    provider = FakeProvider(
        errors=[
            LLMTimeoutError("timeout"),
        ],
        response=make_response('{"diagnosis":"Recovered"}'),
    )

    gateway = LLMGateway(provider)

    monkeypatch.setattr(
        "ai.llm.gateway.time.sleep",
        lambda _: None,
    )

    response = gateway.generate(
        system_prompt="system",
        user_prompt="user",
        model="test-model",
        max_retries=2,
    )

    assert response.content == ('{"diagnosis":"Recovered"}')

    assert provider.calls == 2


def test_gateway_does_not_retry_authentication_error():
    provider = FakeProvider(
        errors=[
            LLMAuthenticationError("invalid key"),
        ]
    )

    gateway = LLMGateway(provider)

    with pytest.raises(LLMAuthenticationError):
        gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=3,
        )

    assert provider.calls == 1


def test_gateway_validates_resolution():
    provider = FakeProvider()

    gateway = LLMGateway(provider)

    response = make_response("""
        {
            "diagnosis": "Email login failure",
            "root_cause": "Invalid credentials",
            "resolution_steps": [
                "Verify credentials"
            ],
            "evidence": [
                "Password reset procedure"
            ],
            "confidence": 0.9,
            "requires_human": false
        }
        """)

    result = gateway.validate_resolution(response)

    assert result.diagnosis == ("Email login failure")

    assert result.confidence == 0.9
    assert result.requires_human is False


def test_gateway_rejects_invalid_resolution():
    provider = FakeProvider()

    gateway = LLMGateway(provider)

    response = make_response('{"diagnosis": ""}')

    with pytest.raises(LLMInvalidResponseError):
        gateway.validate_resolution(response)


def test_gateway_uses_rate_limiter():
    provider = FakeProvider(response=make_response('{"diagnosis":"Test issue"}'))

    limiter = RateLimiter(
        max_requests=1,
        window_seconds=60,
    )

    gateway = LLMGateway(
        provider,
        rate_limiter=limiter,
    )

    gateway.generate(
        system_prompt="system",
        user_prompt="user",
        model="test-model",
        max_retries=1,
    )

    with pytest.raises(LLMRateLimitError):
        gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=1,
        )

    assert provider.calls == 1


def test_gateway_records_successful_usage():
    provider = FakeProvider(response=make_response('{"diagnosis":"Test issue"}'))

    gateway = LLMGateway(
        provider,
    )

    gateway.generate(
        system_prompt="system",
        user_prompt="user",
        model="test-model",
        max_retries=1,
    )

    assert gateway.last_usage is not None

    assert gateway.last_usage.model == "test-model"

    assert gateway.last_usage.prompt_tokens == 100

    assert gateway.last_usage.completion_tokens == 50

    assert gateway.last_usage.total_tokens == 150

    assert gateway.last_usage.success is True
    assert gateway.last_usage.latency_ms >= 0
    assert gateway.last_usage.request_id


def test_gateway_records_failed_usage():
    provider = FakeProvider(
        errors=[
            LLMTimeoutError("timeout"),
        ]
    )

    gateway = LLMGateway(
        provider,
    )

    with pytest.raises(LLMTimeoutError):
        gateway.generate(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            max_retries=1,
        )

    assert gateway.last_usage is not None

    assert gateway.last_usage.model == "test-model"

    assert gateway.last_usage.success is False

    assert gateway.last_usage.error_type == "LLMTimeoutError"

    assert gateway.last_usage.latency_ms >= 0
