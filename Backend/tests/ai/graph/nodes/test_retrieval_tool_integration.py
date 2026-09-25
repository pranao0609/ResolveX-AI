from __future__ import annotations

from unittest.mock import patch

from ai.graph.nodes.retrieval import retrieval_agent


def _base_state() -> dict:
    return {
        "ticket_id": 1,
        "cleaned_ticket": "keyboard is not working",
        "tool_calls": [],
        "metadata": {},
        "retrieval_metadata": {},
        "warnings": [],
        "errors": [],
    }


def test_retrieval_agent_records_kb_tool_call() -> None:
    documents = [
        {
            "index_id": 1,
            "title": "Keyboard Troubleshooting",
            "content": "Check the keyboard connection.",
            "source": "kb",
            "score": 0.9,
        },
        {
            "index_id": 2,
            "title": "Keyboard Driver",
            "content": "Check the keyboard driver.",
            "source": "kb",
            "score": 0.8,
        },
    ]

    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=documents,
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            return_value=documents,
        ),
    ):
        result = retrieval_agent(_base_state())

    assert "tool_calls" in result

    tool_calls = result["tool_calls"]

    assert len(tool_calls) == 3

    names = [
        call["tool"]
        for call in tool_calls
    ]

    assert names == [
        "search_knowledge_base",
        "search_previous_tickets",
        "rerank_documents",
    ]

    for call in tool_calls:
        assert call["agent"] == "retrieval_agent"
        assert call["category"] == "read"
        assert call["status"] == "success"
        assert call["latency_ms"] >= 0


def test_retrieval_agent_preserves_existing_tool_metadata() -> None:
    documents = [
        {
            "index_id": 1,
            "title": "Keyboard",
            "content": "Keyboard troubleshooting.",
            "source": "kb",
            "score": 0.9,
        },
        {
            "index_id": 2,
            "title": "Keyboard Driver",
            "content": "Driver troubleshooting.",
            "source": "kb",
            "score": 0.8,
        },
    ]

    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=documents,
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            return_value=documents,
        ),
    ):
        result = retrieval_agent(_base_state())

    metadata_calls = result[
        "retrieval_metadata"
    ]["tool_calls"]

    assert len(metadata_calls) == 3

    assert metadata_calls[0]["tool"] == (
        "search_knowledge_base"
    )

    assert metadata_calls[1]["tool"] == (
        "search_previous_tickets"
    )

    assert metadata_calls[2]["tool"] == (
        "rerank_documents"
    )


def test_retrieval_agent_preserves_existing_state_tool_calls() -> None:
    existing_call = {
        "agent": "ticket_analyzer",
        "tool": "example_tool",
        "category": "read",
        "input": "example",
        "result_count": 1,
        "status": "success",
        "latency_ms": 1.0,
    }

    state = _base_state()
    state["tool_calls"] = [
        existing_call
    ]

    documents = [
        {
            "index_id": 1,
            "title": "Keyboard",
            "content": "Keyboard troubleshooting.",
            "source": "kb",
            "score": 0.9,
        },
        {
            "index_id": 2,
            "title": "Keyboard Driver",
            "content": "Driver troubleshooting.",
            "source": "kb",
            "score": 0.8,
        },
    ]

    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=documents,
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            return_value=documents,
        ),
    ):
        result = retrieval_agent(state)

    assert result["tool_calls"][0] == existing_call
    assert len(result["tool_calls"]) == 4