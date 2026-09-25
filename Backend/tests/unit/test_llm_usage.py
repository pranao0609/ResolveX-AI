from datetime import datetime

from ai.llm.providers.base import LLMUsage
from ai.llm.usage import LLMUsageRecord


def test_usage_record_defaults():
    record = LLMUsageRecord()

    assert record.request_id
    assert record.prompt_tokens == 0
    assert record.completion_tokens == 0
    assert record.total_tokens == 0
    assert record.latency_ms == 0.0
    assert record.success is True
    assert record.error_type is None
    assert isinstance(
        record.timestamp,
        datetime,
    )


def test_usage_record_from_response():
    usage = LLMUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    record = LLMUsageRecord.from_response(
        model="test-model",
        usage=usage,
        latency_ms=123.45,
    )

    assert record.model == "test-model"
    assert record.prompt_tokens == 100
    assert record.completion_tokens == 50
    assert record.total_tokens == 150
    assert record.latency_ms == 123.45
    assert record.success is True
    assert record.error_type is None


def test_usage_record_from_error():
    record = LLMUsageRecord.from_error(
        model="test-model",
        latency_ms=500.0,
        error_type="LLMTimeoutError",
    )

    assert record.model == "test-model"
    assert record.latency_ms == 500.0
    assert record.success is False
    assert record.error_type == "LLMTimeoutError"

    assert record.prompt_tokens == 0
    assert record.completion_tokens == 0
    assert record.total_tokens == 0


def test_usage_record_to_dict():
    usage = LLMUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    record = LLMUsageRecord.from_response(
        model="test-model",
        usage=usage,
        latency_ms=25.5,
    )

    data = record.to_dict()

    assert data["model"] == "test-model"
    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 5
    assert data["total_tokens"] == 15
    assert data["latency_ms"] == 25.5
    assert data["success"] is True
    assert isinstance(data["timestamp"], str)
    assert data["request_id"]