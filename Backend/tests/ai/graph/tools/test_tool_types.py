from __future__ import annotations

import pytest

from ai.graph.tools.tool_types import (
    ToolCategory,
    ToolDefinition,
    ToolRegistry,
    ToolStatus,
    build_tool_call_record,
)


def test_tool_category_values() -> None:
    assert ToolCategory.READ.value == "read"
    assert ToolCategory.ACTION.value == "action"


def test_tool_status_values() -> None:
    assert ToolStatus.SUCCESS.value == "success"
    assert ToolStatus.ERROR.value == "error"


def test_tool_registry_register_and_get() -> None:
    def fake_tool(query: str) -> list[dict[str, str]]:
        return [{"query": query}]

    definition = ToolDefinition(
        name="fake_search",
        description="Fake search tool",
        category=ToolCategory.READ,
        handler=fake_tool,
    )

    registry = ToolRegistry()

    registry.register(definition)

    assert registry.contains("fake_search")
    assert registry.get("fake_search") is definition
    assert registry.list_tools() == ["fake_search"]


def test_tool_registry_rejects_duplicate_tools() -> None:
    def fake_tool() -> str:
        return "ok"

    definition = ToolDefinition(
        name="duplicate_tool",
        description="Duplicate tool",
        category=ToolCategory.READ,
        handler=fake_tool,
    )

    registry = ToolRegistry()

    registry.register(definition)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition)


def test_tool_registry_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="Unknown ResolveX tool"):
        registry.get("missing_tool")


def test_tool_registry_executes_successfully() -> None:
    def fake_search(query: str) -> list[dict[str, str]]:
        return [
            {"query": query},
            {"query": "second"},
        ]

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="fake_search",
            description="Fake search",
            category=ToolCategory.READ,
            handler=fake_search,
        )
    )

    result = registry.execute(
        "fake_search",
        "keyboard not working",
    )

    assert result["tool"] == "fake_search"
    assert result["status"] == ToolStatus.SUCCESS.value
    assert result["data"] == [
        {"query": "keyboard not working"},
        {"query": "second"},
    ]
    assert result["result_count"] == 2
    assert result["latency_ms"] >= 0


def test_tool_registry_normalizes_scalar_result() -> None:
    def fake_status() -> dict[str, str]:
        return {"status": "operational"}

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="system_status",
            description="Get system status",
            category=ToolCategory.READ,
            handler=fake_status,
        )
    )

    result = registry.execute("system_status")

    assert result["status"] == ToolStatus.SUCCESS.value
    assert result["result_count"] == 1
    assert result["data"] == {"status": "operational"}


def test_tool_registry_captures_errors() -> None:
    def failing_tool() -> None:
        raise RuntimeError("tool failure")

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="failing_tool",
            description="Failing tool",
            category=ToolCategory.ACTION,
            handler=failing_tool,
        )
    )

    result = registry.execute("failing_tool")

    assert result["status"] == ToolStatus.ERROR.value
    assert result["data"] is None
    assert result["result_count"] == 0
    assert result["error"] == "tool failure"
    assert result["latency_ms"] >= 0


def test_build_tool_call_record() -> None:
    def fake_search(query: str) -> list[str]:
        return [query, "result"]

    definition = ToolDefinition(
        name="fake_search",
        description="Fake search",
        category=ToolCategory.READ,
        handler=fake_search,
    )

    result = {
        "tool": "fake_search",
        "status": "success",
        "data": ["one", "two"],
        "result_count": 2,
        "latency_ms": 12.5,
    }

    record = build_tool_call_record(
        agent="retrieval_agent",
        definition=definition,
        arguments="keyboard problem",
        result=result,
    )

    assert record["agent"] == "retrieval_agent"
    assert record["tool"] == "fake_search"
    assert record["category"] == "read"
    assert record["input"] == "keyboard problem"
    assert record["result_count"] == 2
    assert record["status"] == "success"
    assert record["latency_ms"] == 12.5


def test_build_tool_call_record_includes_error() -> None:
    def failing_tool() -> None:
        raise RuntimeError("failure")

    definition = ToolDefinition(
        name="failing_tool",
        description="Failing tool",
        category=ToolCategory.ACTION,
        handler=failing_tool,
    )

    result = {
        "tool": "failing_tool",
        "status": "error",
        "data": None,
        "result_count": 0,
        "latency_ms": 4.0,
        "error": "failure",
    }

    record = build_tool_call_record(
        agent="decision_agent",
        definition=definition,
        arguments={},
        result=result,
    )

    assert record["status"] == "error"
    assert record["error"] == "failure"
