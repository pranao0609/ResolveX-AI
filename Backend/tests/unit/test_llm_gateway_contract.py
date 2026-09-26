from ai.llm.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    LLMUsage,
)

from ai.llm.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)


def test_llm_usage_defaults():
    usage = LLMUsage()

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_llm_usage_values():
    usage = LLMUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150


def test_llm_response_contract():
    usage = LLMUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    response = LLMResponse(
        content='{"diagnosis": "test"}',
        usage=usage,
        model="test-model",
    )

    assert response.content == '{"diagnosis": "test"}'
    assert response.model == "test-model"
    assert response.usage.total_tokens == 15


def test_llm_error_hierarchy():
    errors = [
        LLMTimeoutError(),
        LLMRateLimitError(),
        LLMAuthenticationError(),
        LLMProviderError(),
        LLMInvalidResponseError(),
    ]

    assert all(isinstance(error, LLMError) for error in errors)
