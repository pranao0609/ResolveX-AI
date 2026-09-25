from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMUsage:
    """Token usage returned by an LLM provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Provider-independent response from an LLM."""

    content: str
    usage: LLMUsage
    model: str
    raw_response: Any = None


class BaseLLMProvider(ABC):
    """
    Provider abstraction for ResolveX LLM integrations.

    The gateway depends on this interface rather than on
    a specific provider such as Groq.
    """

    @abstractmethod
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
        """
        Generate a response from the underlying LLM provider.

        Implementations must convert provider-specific responses
        into the provider-independent LLMResponse contract.
        """
        raise NotImplementedError