from __future__ import annotations

import time
from collections import deque
from threading import Lock

from ai.llm.errors import LLMRateLimitError


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Example:
        max_requests=10
        window_seconds=60

    Allows at most 10 requests during any 60-second window.
    """

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float,
    ) -> None:
        if max_requests <= 0:
            raise ValueError(
                "max_requests must be greater than 0"
            )

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be greater than 0"
            )

        self.max_requests = max_requests
        self.window_seconds = window_seconds

        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        """
        Acquire permission for one request.

        Raises:
            LLMRateLimitError:
                When the configured request limit has
                already been reached.
        """

        now = time.monotonic()

        with self._lock:
            self._remove_expired(now)

            if len(self._timestamps) >= self.max_requests:
                raise LLMRateLimitError(
                    "LLM request rate limit exceeded"
                )

            self._timestamps.append(now)

    def _remove_expired(
        self,
        now: float,
    ) -> None:
        cutoff = now - self.window_seconds

        while (
            self._timestamps
            and self._timestamps[0] <= cutoff
        ):
            self._timestamps.popleft()

    def reset(self) -> None:
        """Clear the current rate-limit window."""

        with self._lock:
            self._timestamps.clear()

    @property
    def current_requests(self) -> int:
        """Return the number of active requests in the window."""

        now = time.monotonic()

        with self._lock:
            self._remove_expired(now)

            return len(self._timestamps)