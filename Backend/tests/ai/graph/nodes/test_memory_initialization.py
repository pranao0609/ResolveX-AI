from ai.graph.nodes.initialization import (
    initialize_graph_state,
)


def test_initialize_graph_state_initializes_memory_fields():
    state = {
        "ticket_id": 100,
        "conversation_history": [
            {
                "role": "user",
                "content": "Previous message",
            }
        ],
        "previous_tickets": [
            {
                "ticket_id": 50,
                "title": "Previous ticket",
            }
        ],
        "tool_calls": [
            {
                "agent": "retrieval_agent",
                "tool": "search_knowledge_base",
                "status": "success",
            }
        ],
    }

    result = initialize_graph_state(state)

    assert "conversation_history" in result
    assert "previous_tickets" in result
    assert "tool_calls" in result

    assert result["conversation_history"] == [
        {
            "role": "user",
            "content": "Previous message",
        }
    ]

    assert result["previous_tickets"] == [
        {
            "ticket_id": 50,
            "title": "Previous ticket",
        }
    ]

    assert result["tool_calls"] == [
        {
            "agent": "retrieval_agent",
            "tool": "search_knowledge_base",
            "status": "success",
        }
    ]


def test_initialize_graph_state_uses_empty_memory_defaults():
    state = {
        "ticket_id": 101,
    }

    result = initialize_graph_state(state)

    assert result["conversation_history"] == []
    assert result["previous_tickets"] == []
    assert result["tool_calls"] == []


def test_initialize_graph_state_preserves_runtime_fields():
    state = {
        "ticket_id": 102,
        "request_id": "request-test",
        "graph_run_id": "graph-test",
        "conversation_history": [],
        "previous_tickets": [],
        "tool_calls": [],
        "errors": ["existing error"],
        "warnings": ["existing warning"],
        "fallback_used": True,
    }

    result = initialize_graph_state(state)

    assert result["request_id"] == "request-test"
    assert result["graph_run_id"] == "graph-test"

    assert result["errors"] == [
        "existing error"
    ]

    assert result["warnings"] == [
        "existing warning"
    ]

    assert result["fallback_used"] is True