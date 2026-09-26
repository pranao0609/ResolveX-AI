"""
test_tracer.py — Unit tests for RunnableConfig creation and trace telemetry extraction.
"""

import os
from unittest.mock import patch

from ai.observability.tracer import (
    extract_trace_telemetry,
    get_langgraph_config,
)


def test_get_langgraph_config_defaults():
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}, clear=True):
        config = get_langgraph_config(ticket_id=999)

        assert config["run_name"] == "ResolveX Ticket 999"
        assert config["configurable"]["thread_id"] == "999"
        assert "resolvex" in config["tags"]
        assert "ticket-graph" in config["tags"]
        assert config["metadata"]["ticket_id"] == "999"
        assert config["metadata"]["policy_mode"] == "shadow"
        assert config["metadata"]["feature_version"] == "v1"


def test_get_langgraph_config_sanitizes_metadata():
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}, clear=True):
        extra = {
            "custom_tag": "val",
            "secret_key": "must_be_stripped",
        }
        config = get_langgraph_config(ticket_id="T101", extra_metadata=extra)

        assert config["metadata"]["custom_tag"] == "val"
        assert "secret_key" not in config["metadata"]


def test_get_langgraph_config_failure_isolation():
    # Force an exception during metadata processing to test fail-open
    with patch(
        "ai.observability.tracer.sanitize_metadata", side_effect=ValueError("Boom!")
    ):
        config = get_langgraph_config(ticket_id=123)
        assert isinstance(config, dict)
        assert "tags" in config
        assert "resolvex" in config["tags"]


def test_extract_trace_telemetry():
    state = {
        "prompt_versions": {"resolution_agent": "v2"},
        "model_versions": {"resolution_agent": "groq/llama3"},
        "latency": {"resolution_agent": 150.5},
        "tool_calls": [
            {
                "tool": "search_knowledge_base",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        ],
        "retrieval_metadata": {"strategy": "hybrid", "candidate_count": 10},
        "policy_action": "auto_resolve",
        "policy_metadata": {"policy_mode": "shadow"},
        "errors": [],
        "warnings": [],
    }

    telemetry = extract_trace_telemetry(state)

    assert telemetry["prompt_versions"]["resolution_agent"] == "v2"
    assert telemetry["model_versions"]["resolution_agent"] == "groq/llama3"
    assert telemetry["latency_ms"]["resolution_agent"] == 150.5
    assert telemetry["tool_calls_count"] == 1
    assert telemetry["token_usage"]["total_tokens"] == 150
    assert telemetry["policy_action"] == "auto_resolve"
