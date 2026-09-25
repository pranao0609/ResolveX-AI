from __future__ import annotations

from ai.graph.tools.executor import execute_tool
from ai.graph.tools.tool_types import (
    ToolCategory,
    ToolDefinition,
    ToolRegistry,
    ToolStatus,
)
from ai.graph.tools.executor import append_tool_call
from ai.graph.state import ResolveXState

def build_test_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="fake_search",
            description="Fake search tool",
            category=ToolCategory.READ,
            handler=lambda query: [
                {"query": query},
                {"query": "second"},
            ],
        )
    )

    return registry


def test_execute_tool_returns_result_and_record() -> None:
    registry = build_test_registry()

    result, record = execute_tool(
        registry,
        agent="retrieval_agent",
        tool_name="fake_search",
        arguments="keyboard not working",
        kwargs={
            "query": "keyboard not working",
        },
    )

    assert result["tool"] == "fake_search"
    assert result["status"] == ToolStatus.SUCCESS.value
    assert result["result_count"] == 2

    assert record["agent"] == "retrieval_agent"
    assert record["tool"] == "fake_search"
    assert record["category"] == "read"
    assert record["input"] == "keyboard not working"
    assert record["result_count"] == 2
    assert record["status"] == "success"
    assert record["latency_ms"] >= 0


def test_execute_tool_records_error() -> None:
    registry = ToolRegistry()

    def failing_tool() -> None:
        raise RuntimeError("database unavailable")

    registry.register(
        ToolDefinition(
            name="failing_tool",
            description="Failing tool",
            category=ToolCategory.READ,
            handler=failing_tool,
        )
    )

    result, record = execute_tool(
        registry,
        agent="retrieval_agent",
        tool_name="failing_tool",
        arguments={},
    )

    assert result["status"] == "error"
    assert result["error"] == "database unavailable"

    assert record["agent"] == "retrieval_agent"
    assert record["tool"] == "failing_tool"
    assert record["status"] == "error"
    assert record["error"] == "database unavailable"


def test_execute_tool_supports_keyword_arguments() -> None:
    registry = ToolRegistry()

    def fake_tool(
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        return [
            {
                "query": query,
                "rank": index,
            }
            for index in range(top_k)
        ]

    registry.register(
        ToolDefinition(
            name="parameterized_search",
            description="Parameterized search",
            category=ToolCategory.READ,
            handler=fake_tool,
        )
    )

    result, record = execute_tool(
        registry,
        agent="retrieval_agent",
        tool_name="parameterized_search",
        arguments={
            "query": "login failure",
            "top_k": 3,
        },
        kwargs={
            "query": "login failure",
            "top_k": 3,
        },
    )

    assert result["status"] == "success"
    assert result["result_count"] == 3
    assert record["result_count"] == 3
    assert record["input"] == {
        "query": "login failure",
        "top_k": 3,
    }

def test_append_tool_call_creates_tool_calls_list() -> None:
    state: ResolveXState = {}

    record = {
        "agent": "retrieval_agent",
        "tool": "search_knowledge_base",
        "category": "read",
        "input": "keyboard not working",
        "result_count": 5,
        "status": "success",
        "latency_ms": 25.5,
    }

    append_tool_call(
        state,
        record,
    )

    assert "tool_calls" in state
    assert len(state["tool_calls"]) == 1
    assert state["tool_calls"][0] == record


def test_append_tool_call_preserves_existing_records() -> None:
    existing_record = {
        "agent": "retrieval_agent",
        "tool": "search_knowledge_base",
        "category": "read",
        "input": "network issue",
        "result_count": 3,
        "status": "success",
        "latency_ms": 12.0,
    }

    new_record = {
        "agent": "retrieval_agent",
        "tool": "search_previous_tickets",
        "category": "read",
        "input": "network issue",
        "result_count": 2,
        "status": "success",
        "latency_ms": 8.0,
    }

    state: ResolveXState = {
        "tool_calls": [
            existing_record,
        ],
    }

    append_tool_call(
        state,
        new_record,
    )

    assert len(state["tool_calls"]) == 2
    assert state["tool_calls"][0] == existing_record
    assert state["tool_calls"][1] == new_record


def test_multiple_tool_calls_are_recorded_in_order() -> None:
    state: ResolveXState = {}

    records = [
        {
            "agent": "retrieval_agent",
            "tool": "search_knowledge_base",
            "category": "read",
            "input": "login failure",
            "result_count": 5,
            "status": "success",
            "latency_ms": 10.0,
        },
        {
            "agent": "retrieval_agent",
            "tool": "search_previous_tickets",
            "category": "read",
            "input": "login failure",
            "result_count": 3,
            "status": "success",
            "latency_ms": 15.0,
        },
    ]

    for record in records:
        append_tool_call(
            state,
            record,
        )

    assert state["tool_calls"] == records