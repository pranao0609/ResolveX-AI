from __future__ import annotations

from typing import Any

from ai.graph.state import ResolveXState
from ai.graph.tools.tool_types import (
    ToolCallRecord,
    ToolRegistry,
    build_tool_call_record,
)


def execute_tool(
    registry: ToolRegistry,
    *,
    agent: str,
    tool_name: str,
    arguments: Any = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[Any, ToolCallRecord]:
    """
    Execute a registered tool and produce an auditable call record.
    """

    if kwargs is None:
        kwargs = {}

    if arguments is None:
        arguments = kwargs

    result = registry.execute(
        tool_name,
        **kwargs,
    )

    definition = registry.get(tool_name)

    record = build_tool_call_record(
        agent=agent,
        definition=definition,
        arguments=arguments,
        result=result,
    )

    return result, record


def append_tool_call(
    state: ResolveXState,
    record: ToolCallRecord,
) -> None:
    """
    Append one tool-call record to the current graph state.

    The state is mutated intentionally because LangGraph nodes
    operate on the shared ResolveXState object.
    """

    tool_calls = state.setdefault(
        "tool_calls",
        [],
    )

    tool_calls.append(record)
