from __future__ import annotations

from typing import Any, Callable, Dict

from ai.agents.retrieval_tools import (
    get_retrieval_tools,
    rerank_documents,
    search_knowledge_base,
    search_previous_tickets,
)

RETRIEVAL_TOOLS: Dict[str, Callable[..., Any]] = {
    "search_knowledge_base": search_knowledge_base,
    "search_previous_tickets": search_previous_tickets,
    "rerank_documents": rerank_documents,
}


def get_registered_tool(name: str) -> Callable[..., Any] | None:
    """Return a registered retrieval tool by name."""
    return RETRIEVAL_TOOLS.get(name)


__all__ = [
    "RETRIEVAL_TOOLS",
    "get_registered_tool",
    "get_retrieval_tools",
]
