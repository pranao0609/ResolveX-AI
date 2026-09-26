import pytest

from ai.llm.errors import LLMRateLimitError
from ai.llm.rate_limiter import RateLimiter


def test_rate_limiter_allows_requests_within_limit():
    limiter = RateLimiter(
        max_requests=3,
        window_seconds=60,
    )

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert limiter.current_requests == 3


def test_rate_limiter_blocks_requests_over_limit():
    limiter = RateLimiter(
        max_requests=2,
        window_seconds=60,
    )

    limiter.acquire()
    limiter.acquire()

    with pytest.raises(LLMRateLimitError):
        limiter.acquire()


def test_rate_limiter_reset():
    limiter = RateLimiter(
        max_requests=1,
        window_seconds=60,
    )

    limiter.acquire()

    assert limiter.current_requests == 1

    limiter.reset()

    assert limiter.current_requests == 0

    limiter.acquire()

    assert limiter.current_requests == 1


def test_rate_limiter_rejects_invalid_max_requests():
    with pytest.raises(ValueError):
        RateLimiter(
            max_requests=0,
            window_seconds=60,
        )


def test_rate_limiter_rejects_invalid_window():
    with pytest.raises(ValueError):
        RateLimiter(
            max_requests=10,
            window_seconds=0,
        )


def test_rate_limiter_expired_request(monkeypatch):
    current_time = [100.0]

    monkeypatch.setattr(
        "ai.llm.rate_limiter.time.monotonic",
        lambda: current_time[0],
    )

    limiter = RateLimiter(
        max_requests=1,
        window_seconds=10,
    )

    limiter.acquire()

    with pytest.raises(LLMRateLimitError):
        limiter.acquire()

    current_time[0] = 111.0

    limiter.acquire()

    assert limiter.current_requests == 1
