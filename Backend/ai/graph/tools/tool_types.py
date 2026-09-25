from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Protocol, TypedDict


class ToolCategory(StrEnum):
    """High-level classification of ResolveX tools."""

    READ = "read"
    ACTION = "action"


class ToolStatus(StrEnum):
    """Execution status for a tool invocation."""

    SUCCESS = "success"
    ERROR = "error"


class ToolMetadata(TypedDict, total=False):
    """
    Backward-compatible metadata contract for graph tools.
    """

    name: str
    description: str
    category: str


class RetrievalToolProtocol(Protocol):
    """
    Backward-compatible protocol for retrieval tools.

    Existing Phase 16 retrieval implementations continue to satisfy
    this interface.
    """

    def __call__(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        ...


class ToolCallRecord(TypedDict, total=False):
    """
    Auditable record of a single tool invocation.
    """

    agent: str
    tool: str
    category: str
    input: Any
    result_count: int
    status: str
    latency_ms: float
    error: str


class ToolResult(TypedDict, total=False):
    """
    Standardized result returned by the tool execution layer.
    """

    tool: str
    status: str
    data: Any
    result_count: int
    latency_ms: float
    error: str


class ToolCallable(Protocol):
    """Callable interface implemented by registered tools."""

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...


@dataclass(frozen=True)
class ToolDefinition:
    """
    Metadata and callable implementation for one ResolveX tool.
    """

    name: str
    description: str
    category: ToolCategory
    handler: ToolCallable
    requires_confirmation: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Central registry for ResolveX tools.

    The registry manages tool definitions and execution without
    containing tool-specific business logic.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        definition: ToolDefinition,
    ) -> None:
        """Register a tool definition."""

        if definition.name in self._tools:
            raise ValueError(
                f"Tool already registered: {definition.name}"
            )

        self._tools[definition.name] = definition

    def get(
        self,
        name: str,
    ) -> ToolDefinition:
        """Return a registered tool definition."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown ResolveX tool: {name}"
            ) from exc

    def contains(
        self,
        name: str,
    ) -> bool:
        """Return whether a tool is registered."""

        return name in self._tools

    def list_tools(self) -> list[str]:
        """Return registered tool names."""

        return list(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""

        return list(self._tools.values())

    def execute(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute a registered tool and normalize its result.
        """

        definition = self.get(name)

        start = perf_counter()

        try:
            result = definition.handler(
                *args,
                **kwargs,
            )

            latency_ms = (
                perf_counter() - start
            ) * 1000.0

            result_count = _infer_result_count(result)

            return ToolResult(
                tool=name,
                status=ToolStatus.SUCCESS.value,
                data=result,
                result_count=result_count,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = (
                perf_counter() - start
            ) * 1000.0

            return ToolResult(
                tool=name,
                status=ToolStatus.ERROR.value,
                data=None,
                result_count=0,
                latency_ms=latency_ms,
                error=str(exc),
            )


def _infer_result_count(
    result: Any,
) -> int:
    """
    Infer a useful result count for tool-call telemetry.
    """

    if result is None:
        return 0

    if isinstance(
        result,
        (list, tuple, set),
    ):
        return len(result)

    if isinstance(result, Mapping):
        return 1

    return 1


def build_tool_call_record(
    *,
    agent: str,
    definition: ToolDefinition,
    arguments: Any,
    result: ToolResult,
) -> ToolCallRecord:
    """
    Build a serializable audit record from a tool execution result.
    """

    record = ToolCallRecord(
        agent=agent,
        tool=definition.name,
        category=definition.category.value,
        input=arguments,
        result_count=int(
            result.get(
                "result_count",
                0,
            )
        ),
        status=str(
            result.get(
                "status",
                ToolStatus.ERROR.value,
            )
        ),
        latency_ms=float(
            result.get(
                "latency_ms",
                0.0,
            )
        ),
    )

    error = result.get("error")

    if error:
        record["error"] = str(error)

    return record


__all__ = [
    "ToolCategory",
    "ToolStatus",
    "ToolMetadata",
    "RetrievalToolProtocol",
    "ToolCallRecord",
    "ToolResult",
    "ToolCallable",
    "ToolDefinition",
    "ToolRegistry",
    "build_tool_call_record",
]