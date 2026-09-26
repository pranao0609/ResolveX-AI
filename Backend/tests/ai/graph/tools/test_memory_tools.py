from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai.graph.tools.executor import execute_tool
from ai.graph.tools.memory_registry import build_memory_registry
from ai.graph.tools.memory_tools import search_memory


def test_memory_registry_contains_search_memory() -> None:
    registry = build_memory_registry()

    assert registry.contains("search_memory")
    assert registry.list_tools() == ["search_memory"]

    definition = registry.get("search_memory")

    assert definition.name == "search_memory"
    assert definition.category.value == "read"
    assert definition.requires_confirmation is False


def test_search_memory_returns_normalized_records() -> None:
    mock_db = MagicMock()

    records = [
        {
            "ticket_id": 10,
            "title": "VPN connection failed",
            "description": "VPN does not connect.",
            "category": "network",
            "status": "closed",
            "solution": "Reset VPN credentials.",
            "confidence": 0.95,
        }
    ]

    with patch("ai.graph.tools.memory_tools.TicketMemoryStore") as memory_class:
        memory = memory_class.return_value

        search_result = MagicMock()
        search_result.records = [MagicMock(to_dict=lambda: records[0])]

        memory.search.return_value = search_result

        result = search_memory(
            db=mock_db,
            query="VPN",
            category="network",
            limit=5,
        )

    assert len(result) == 1
    assert result[0]["ticket_id"] == 10
    assert result[0]["solution"] == "Reset VPN credentials."


def test_memory_tool_execution_creates_audit_record() -> None:
    registry = build_memory_registry()

    expected = [
        {
            "ticket_id": 10,
            "title": "VPN connection failed",
            "description": "VPN does not connect.",
            "category": "network",
            "status": "closed",
            "solution": "Reset VPN credentials.",
        }
    ]

    with patch(
        "ai.graph.tools.memory_registry.search_memory",
        return_value=expected,
    ):
        # Rebuild because the handler is captured during registration.
        registry = build_memory_registry()

        data, record = execute_tool(
            registry,
            agent="diagnosis_agent",
            tool_name="search_memory",
            arguments={
                "query": "VPN",
                "category": "network",
            },
            kwargs={
                "query": "VPN",
                "category": "network",
                "limit": 5,
            },
        )

    assert data["status"] == "success"
    assert data["data"] == expected
    assert data["result_count"] == 1
    assert data["latency_ms"] >= 0

    assert record["agent"] == "diagnosis_agent"
    assert record["tool"] == "search_memory"
    assert record["category"] == "read"
    assert record["status"] == "success"
    assert record["result_count"] == 1
    assert record["latency_ms"] >= 0


def test_memory_tool_failure_is_normalized() -> None:
    with patch(
        "ai.graph.tools.memory_registry.search_memory",
        side_effect=RuntimeError("database unavailable"),
    ):
        registry = build_memory_registry()

        result, record = execute_tool(
            registry,
            agent="diagnosis_agent",
            tool_name="search_memory",
            arguments={"query": "VPN"},
            kwargs={"query": "VPN"},
        )

    assert result["status"] == "error"
    assert result["result_count"] == 0

    assert record["tool"] == "search_memory"
    assert record["agent"] == "diagnosis_agent"
    assert record["category"] == "read"
    assert record["status"] == "error"
    assert record["latency_ms"] >= 0
    assert "database unavailable" in record["error"]
