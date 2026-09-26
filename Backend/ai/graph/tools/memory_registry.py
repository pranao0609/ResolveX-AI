"""
Registry for ResolveX memory tools.

Memory tools are kept separate from retrieval tools so each agent can
request only the capabilities it needs.
"""

from __future__ import annotations

from ai.graph.tools.memory_tools import search_memory
from ai.graph.tools.tool_types import (
    ToolCategory,
    ToolDefinition,
    ToolRegistry,
)


def build_memory_registry() -> ToolRegistry:
    """Build and return the registry of memory tools."""

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="search_memory",
            description=(
                "Search previously resolved ResolveX tickets for "
                "historical troubleshooting evidence."
            ),
            category=ToolCategory.READ,
            handler=search_memory,
        )
    )

    return registry


__all__ = ["build_memory_registry"]
