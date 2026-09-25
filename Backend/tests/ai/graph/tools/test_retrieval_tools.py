from __future__ import annotations

from ai.graph.tools.retrieval_tools import (
    build_retrieval_tool_registry,
    retrieval_tool_registry,
)
from ai.graph.tools.tool_types import ToolCategory


EXPECTED_TOOLS = {
    "search_knowledge_base",
    "search_previous_tickets",
    "rerank_documents",
}


def test_retrieval_registry_contains_expected_tools() -> None:
    registry = build_retrieval_tool_registry()

    assert set(registry.list_tools()) == EXPECTED_TOOLS


def test_retrieval_registry_contains_global_registry() -> None:
    assert set(
        retrieval_tool_registry.list_tools()
    ) == EXPECTED_TOOLS


def test_retrieval_tools_are_read_tools() -> None:
    registry = build_retrieval_tool_registry()

    for tool_name in EXPECTED_TOOLS:
        definition = registry.get(tool_name)

        assert definition.category == ToolCategory.READ


def test_retrieval_tool_handlers_are_callable() -> None:
    registry = build_retrieval_tool_registry()

    for tool_name in EXPECTED_TOOLS:
        definition = registry.get(tool_name)

        assert callable(definition.handler)


def test_retrieval_tool_descriptions_exist() -> None:
    registry = build_retrieval_tool_registry()

    for tool_name in EXPECTED_TOOLS:
        definition = registry.get(tool_name)

        assert definition.description
        assert isinstance(
            definition.description,
            str,
        )


def test_registry_returns_definitions() -> None:
    registry = build_retrieval_tool_registry()

    definitions = registry.definitions()

    assert len(definitions) == 3

    names = {
        definition.name
        for definition in definitions
    }

    assert names == EXPECTED_TOOLS