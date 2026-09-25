from types import SimpleNamespace

import pytest

from ai.llm.errors import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

from ai.llm.providers.groq_provider import (
    GroqProvider,
)


class FakeCompletion:
    def __init__(self):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"diagnosis": "Test issue"}'
                )
            )
        ]

        self.usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )


def test_groq_provider_returns_llm_response(monkeypatch):
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return FakeCompletion()

    monkeypatch.setattr(
        "ai.llm.providers.groq_provider.Groq",
        lambda **kwargs: FakeClient(),
    )

    provider = GroqProvider(
        api_key="test-key",
    )

    response = provider.generate(
        system_prompt="system",
        user_prompt="user",
        model="test-model",
        temperature=0.3,
        max_tokens=100,
        timeout=10,
    )

    assert response.content == (
        '{"diagnosis": "Test issue"}'
    )

    assert response.model == "test-model"

    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 50
    assert response.usage.total_tokens == 150


def test_groq_provider_empty_response(monkeypatch):
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    completion = FakeCompletion()
                    completion.choices[0].message.content = ""
                    return completion

    monkeypatch.setattr(
        "ai.llm.providers.groq_provider.Groq",
        lambda **kwargs: FakeClient(),
    )

    provider = GroqProvider(
        api_key="test-key",
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


def test_groq_provider_timeout(monkeypatch):
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise Exception(
                        "Request timed out"
                    )

    monkeypatch.setattr(
        "ai.llm.providers.groq_provider.Groq",
        lambda **kwargs: FakeClient(),
    )

    provider = GroqProvider(
        api_key="test-key",
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


def test_groq_provider_rate_limit(monkeypatch):
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise Exception(
                        "429 rate limit exceeded"
                    )

    monkeypatch.setattr(
        "ai.llm.providers.groq_provider.Groq",
        lambda **kwargs: FakeClient(),
    )

    provider = GroqProvider(
        api_key="test-key",
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