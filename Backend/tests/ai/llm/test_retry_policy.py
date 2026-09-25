import pytest

from ai.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ai.llm.retry_policy import RetryPolicy


@pytest.mark.parametrize(
    "error",
    [
        LLMTimeoutError("timeout"),
        LLMRateLimitError("rate limit"),
        LLMProviderError("provider failure"),
    ],
)
def test_retryable_errors(error):
    decision = RetryPolicy.should_retry(error)

    assert decision.retryable is True
    assert type(error).__name__ in decision.reason


@pytest.mark.parametrize(
    "error",
    [
        LLMAuthenticationError("authentication failed"),
        LLMInvalidResponseError("invalid response"),
        LLMConfigurationError("invalid configuration"),
    ],
)
def test_non_retryable_errors(error):
    decision = RetryPolicy.should_retry(error)

    assert decision.retryable is False
    assert type(error).__name__ in decision.reason


def test_unknown_llm_error_is_not_retryable():
    from ai.llm.errors import LLMError

    error = LLMError("unknown error")

    decision = RetryPolicy.should_retry(error)

    assert decision.retryable is False


def test_unknown_exception_is_not_retryable():
    error = ValueError("unexpected error")

    decision = RetryPolicy.should_retry(error)

    assert decision.retryable is False


@pytest.mark.parametrize(
    "attempt,expected",
    [
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 8),
        (5, 8),
        (10, 8),
    ],
)
def test_backoff_seconds(attempt, expected):
    assert (
        RetryPolicy.backoff_seconds(attempt)
        == expected
    )


def test_backoff_rejects_invalid_attempt():
    with pytest.raises(ValueError):
        RetryPolicy.backoff_seconds(0)