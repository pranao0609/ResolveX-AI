from __future__ import annotations

import time

from app.config import settings
from app.core.logger import logger

from ai.config.ai_config import (
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
)

from ai.llm.errors import (
    LLMError,
    LLMInvalidResponseError,
)

from ai.llm.providers.base import (
    BaseLLMProvider,
    LLMResponse,
)

from ai.llm.rate_limiter import RateLimiter
from ai.llm.retry_policy import RetryPolicy
from ai.llm.schemas import ResolutionResult
from ai.llm.usage import LLMUsageRecord


class LLMGateway:
    """
    Central reliability layer for ResolveX LLM calls.

    Responsibilities:
    - provider invocation
    - bounded retries
    - timeout propagation
    - rate limiting
    - retry policy enforcement
    - structured-output validation
    - usage tracking
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        rate_limiter: RateLimiter | None = None,
        retry_policy: type[RetryPolicy] = RetryPolicy,
    ) -> None:
        self.provider = provider
        self.retry_policy = retry_policy

        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = RateLimiter(
                max_requests=settings.LLM_RATE_LIMIT_REQUESTS,
                window_seconds=settings.LLM_RATE_LIMIT_WINDOW_SECONDS,
            )

        self.last_usage: LLMUsageRecord | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = GROQ_TEMPERATURE,
        max_tokens: int = GROQ_MAX_TOKENS,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> LLMResponse:
        """
        Generate an LLM response through the configured provider.

        Retry behavior is delegated to RetryPolicy.
        """

        request_timeout = (
            timeout
            if timeout is not None
            else settings.LLM_TIMEOUT_SECONDS
        )

        retries = (
            max_retries
            if max_retries is not None
            else settings.LLM_MAX_RETRIES
        )

        retries = max(1, retries)

        # One logical request consumes one rate-limit slot.
        # Individual retries do not consume additional slots.
        self.rate_limiter.acquire()

        last_error: LLMError | None = None

        for attempt in range(1, retries + 1):
            start_time = time.perf_counter()

            try:
                response = self.provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout,
                )

                elapsed_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                self.last_usage = LLMUsageRecord.from_response(
                    model=model,
                    usage=response.usage,
                    latency_ms=elapsed_ms,
                )

                logger.info(
                    "LLM Gateway request succeeded "
                    f"attempt={attempt}/{retries} "
                    f"model={model} "
                    f"latency_ms={elapsed_ms:.2f} "
                    f"prompt_tokens="
                    f"{response.usage.prompt_tokens} "
                    f"completion_tokens="
                    f"{response.usage.completion_tokens} "
                    f"total_tokens="
                    f"{response.usage.total_tokens}"
                )

                return response

            except Exception as exc:
                elapsed_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                self.last_usage = LLMUsageRecord.from_error(
                    model=model,
                    latency_ms=elapsed_ms,
                    error_type=type(exc).__name__,
                )

                decision = self.retry_policy.should_retry(exc)

                logger.warning(
                    "LLM Gateway request failed "
                    f"attempt={attempt}/{retries} "
                    f"error={type(exc).__name__} "
                    f"retryable={decision.retryable} "
                    f"reason={decision.reason}"
                )

                if not decision.retryable:
                    if isinstance(exc, LLMError):
                        raise

                    raise LLMError(
                        f"Unexpected LLM gateway error: {exc}"
                    ) from exc

                if attempt >= retries:
                    if isinstance(exc, LLMError):
                        last_error = exc
                    else:
                        last_error = LLMError(
                            f"Unexpected LLM gateway error: {exc}"
                        )

                    break

                delay = self.retry_policy.backoff_seconds(
                    attempt
                )

                logger.info(
                    "LLM Gateway retry scheduled "
                    f"attempt={attempt + 1}/{retries} "
                    f"delay_seconds={delay}"
                )

                time.sleep(delay)

        if last_error is not None:
            raise last_error

        raise LLMError(
            "LLM Gateway failed without a captured error"
        )

    @staticmethod
    def validate_resolution(
        response: LLMResponse,
    ) -> ResolutionResult:
        """
        Validate a provider response against the ResolveX
        structured ResolutionResult contract.
        """

        try:
            return ResolutionResult.model_validate_json(
                response.content
            )

        except Exception as exc:
            raise LLMInvalidResponseError(
                "LLM response failed ResolutionResult validation"
            ) from exc