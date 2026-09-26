from __future__ import annotations

from ai.agents.retrieval_tools import (
    get_retrieval_tools,
    rerank_documents,
    search_knowledge_base,
    search_previous_tickets,
)
from ai.graph.tools.tool_types import (
    ToolCategory,
    ToolDefinition,
    ToolRegistry,
)


def build_retrieval_tool_registry() -> ToolRegistry:
    """
    Build the Phase 19 retrieval-tool registry.

    Existing Phase 16 retrieval implementations are reused directly.
    No retrieval behavior is changed here.
    """

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="search_knowledge_base",
            description=(
                "Search the ResolveX knowledge base for "
                "relevant technical documentation."
            ),
            category=ToolCategory.READ,
            handler=search_knowledge_base,
        )
    )

    registry.register(
        ToolDefinition(
            name="search_previous_tickets",
            description=(
                "Search historical ResolveX tickets "
                "for similar incidents and resolutions."
            ),
            category=ToolCategory.READ,
            handler=search_previous_tickets,
        )
    )

    registry.register(
        ToolDefinition(
            name="rerank_documents",
            description=(
                "Rerank retrieved documents using the " "configured relevance reranker."
            ),
            category=ToolCategory.READ,
            handler=rerank_documents,
        )
    )

    return registry


retrieval_tool_registry = build_retrieval_tool_registry()


__all__ = [
    "get_retrieval_tools",
    "search_knowledge_base",
    "search_previous_tickets",
    "rerank_documents",
    "build_retrieval_tool_registry",
    "retrieval_tool_registry",
]
