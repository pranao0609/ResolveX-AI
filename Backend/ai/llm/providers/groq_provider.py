import os
from groq import Groq

from app.config import settings
from ai.llm.errors import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ai.llm.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    LLMUsage,
)


class GroqProvider(BaseLLMProvider):
    """Groq implementation of the generic LLM provider contract."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY
        )

        if not self.api_key or self.api_key == "your-groq-api-key-here":
            raise ValueError("Groq API key is not configured")

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> LLMResponse:
        """Generate a response using the Groq API."""

        try:
            client = Groq(
                api_key=self.api_key,
                timeout=timeout,
            )

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        except Exception as exc:
            error_message = str(exc).lower()

            if "timeout" in error_message or "timed out" in error_message:
                raise LLMTimeoutError(str(exc)) from exc

            if "rate limit" in error_message or "429" in error_message:
                raise LLMRateLimitError(str(exc)) from exc

            if (
                "authentication" in error_message
                or "unauthorized" in error_message
                or "401" in error_message
            ):
                raise LLMAuthenticationError(str(exc)) from exc

            raise LLMProviderError(str(exc)) from exc

        try:
            content = completion.choices[0].message.content

            if not content:
                raise LLMProviderError("Groq returned an empty response")

            usage_data = getattr(
                completion,
                "usage",
                None,
            )

            prompt_tokens = int(
                getattr(
                    usage_data,
                    "prompt_tokens",
                    0,
                )
                or 0
            )

            completion_tokens = int(
                getattr(
                    usage_data,
                    "completion_tokens",
                    0,
                )
                or 0
            )

            total_tokens = int(
                getattr(
                    usage_data,
                    "total_tokens",
                    prompt_tokens + completion_tokens,
                )
                or (prompt_tokens + completion_tokens)
            )

            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            return LLMResponse(
                content=content,
                usage=usage,
                model=model,
                raw_response=completion,
            )

        except LLMProviderError:
            raise

        except Exception as exc:
            raise LLMProviderError(f"Invalid Groq response: {exc}") from exc
