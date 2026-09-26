from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from ai.llm.providers.base import LLMUsage


@dataclass
class LLMUsageRecord:
    """
    Provider-independent usage record for one LLM request.
    """

    request_id: str = field(default_factory=lambda: str(uuid4()))

    model: str = ""

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0

    latency_ms: float = 0.0

    success: bool = True

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    error_type: str | None = None

    @classmethod
    def from_response(
        cls,
        *,
        model: str,
        usage: LLMUsage,
        latency_ms: float,
    ) -> "LLMUsageRecord":
        """
        Create a usage record from a provider-independent
        LLM response.
        """

        return cls(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
            success=True,
        )

    @classmethod
    def from_error(
        cls,
        *,
        model: str,
        latency_ms: float,
        error_type: str,
    ) -> "LLMUsageRecord":
        """
        Create a usage record for a failed LLM request.
        """

        return cls(
            model=model,
            latency_ms=latency_ms,
            success=False,
            error_type=error_type,
        )

    def to_dict(self) -> dict:
        """
        Serialize the usage record into JSON-compatible data.
        """

        return {
            "request_id": self.request_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
        }
