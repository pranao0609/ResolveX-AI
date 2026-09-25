from __future__ import annotations

from dataclasses import dataclass

from ai.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)


@dataclass(frozen=True)
class RetryDecision:
    """
    Result of evaluating whether an LLM failure should be retried.
    """

    retryable: bool
    reason: str


class RetryPolicy:
    """
    Defines retry behavior for ResolveX LLM failures.

    Retryable:
    - timeout
    - rate limit
    - provider-side transient failure

    Non-retryable:
    - authentication failure
    - invalid response
    - configuration failure
    - unknown/non-LLM errors
    """

    RETRYABLE_ERRORS = (
        LLMTimeoutError,
        LLMRateLimitError,
        LLMProviderError,
    )

    NON_RETRYABLE_ERRORS = (
        LLMAuthenticationError,
        LLMInvalidResponseError,
        LLMConfigurationError,
    )

    @classmethod
    def should_retry(cls, error: Exception) -> RetryDecision:
        """
        Determine whether an exception represents a retryable failure.
        """

        if isinstance(error, cls.RETRYABLE_ERRORS):
            return RetryDecision(
                retryable=True,
                reason=f"{type(error).__name__} is retryable",
            )

        if isinstance(error, cls.NON_RETRYABLE_ERRORS):
            return RetryDecision(
                retryable=False,
                reason=f"{type(error).__name__} is non-retryable",
            )

        if isinstance(error, LLMError):
            return RetryDecision(
                retryable=False,
                reason=(
                    f"{type(error).__name__} is an unclassified "
                    "LLM error"
                ),
            )

        return RetryDecision(
            retryable=False,
            reason=(
                f"{type(error).__name__} is not an LLM "
                "retryable error"
            ),
        )

    @staticmethod
    def backoff_seconds(attempt: int) -> int:
        """
        Calculate bounded exponential backoff.

        Attempt 1 -> 1 second
        Attempt 2 -> 2 seconds
        Attempt 3 -> 4 seconds
        Attempt 4+ -> 8 seconds
        """

        if attempt < 1:
            raise ValueError("attempt must be >= 1")

        return min(2 ** (attempt - 1), 8)