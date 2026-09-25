from __future__ import annotations

import pytest

from ai.graph.graph import (
    ask_clarification,
    auto_resolve,
    build_resolvex_graph,
    decision_agent,
    diagnosis_agent,
    escalate,
    human_review,
    initialize_graph_state,
    resolution_agent,
    resolvex_graph,
    retrieval_agent,
    retrieval_decision_agent,
    ticket_analyzer,
    verification_agent,
)
from ai.graph.routing import route_decision, route_retrieval_decision
from ai.graph.state import ResolveXState


def test_graph_imports_successfully():
    """1. Test that graph imports successfully and exports key symbols."""
    import ai.graph as graph_pkg
    import ai.graph.graph as graph_module
    import ai.graph.nodes as nodes_pkg
    import ai.graph.routing as routing_module
    import ai.graph.state as state_module
    import ai.graph.tools as tools_pkg

    assert hasattr(graph_pkg, "resolvex_graph")
    assert hasattr(graph_pkg, "build_resolvex_graph")
    assert hasattr(graph_pkg, "ResolveXState")

    assert hasattr(graph_module, "resolvex_graph")
    assert hasattr(graph_module, "build_resolvex_graph")

    assert hasattr(routing_module, "route_decision")
    assert hasattr(routing_module, "route_retrieval_decision")

    assert hasattr(state_module, "ResolveXState")
    assert hasattr(nodes_pkg, "ticket_analyzer")
    assert hasattr(tools_pkg, "search_knowledge_base")


def test_resolvex_graph_exists():
    """2. Test that resolvex_graph instance exists."""
    assert resolvex_graph is not None


def test_build_resolvex_graph_compiles_successfully():
    """3. Test that build_resolvex_graph() compiles successfully."""
    compiled = build_resolvex_graph()
    assert compiled is not None


def test_all_expected_nodes_are_registered():
    """4. Test that all 12 expected nodes are registered."""
    compiled = build_resolvex_graph()
    graph_nodes = compiled.get_graph().nodes

    expected_nodes = [
        "initialize_state",
        "ticket_analyzer",
        "retrieval_agent",
        "retrieval_decision_agent",
        "diagnosis_agent",
        "resolution_agent",
        "verification_agent",
        "decision_agent",
        "auto_resolve",
        "ask_clarification",
        "human_review",
        "escalate",
    ]

    for node_name in expected_nodes:
        assert node_name in graph_nodes, f"Node {node_name} missing from compiled graph"


def test_all_four_terminal_routes_exist():
    """5. Test that all four Phase 18 terminal routes exist."""
    assert route_decision({"decision": "auto_resolve"}) == "auto_resolve"
    assert route_decision({"decision": "ask_clarification"}) == "ask_clarification"
    assert route_decision({"decision": "human_review"}) == "human_review"
    assert route_decision({"decision": "escalate"}) == "escalate"


def test_unknown_decision_routes_to_human_review():
    """6. Test that unknown or empty decisions fail closed to human_review."""
    assert route_decision({}) == "human_review"
    assert route_decision({"decision": "unknown_decision_type"}) == "human_review"
    assert route_decision({"decision": None}) == "human_review"
    assert route_decision({"decision": ""}) == "human_review"


def test_basic_graph_execution():
    """7. Test that basic graph execution still works."""
    state: ResolveXState = {
        "ticket_id": 999,
        "ticket_text": "Unable to login to work email account",
    }

    result = resolvex_graph.invoke(state)

    assert result["ticket_id"] == 999
    assert result["cleaned_ticket"]
    assert result["category"] in {"software", "hardware", "network", "access"}
    assert "decision" in result
    assert result["decision"] in {
        "auto_resolve",
        "ask_clarification",
        "human_review",
        "escalate",
    }
    assert "metadata" in result


def test_retrieval_retry_routing():
    """8. Test that retrieval retry routing still works."""
    retry_state: ResolveXState = {
        "retrieval_metadata": {
            "retrieval_decision": "retry",
        }
    }
    assert route_retrieval_decision(retry_state) == "retry"

    continue_state: ResolveXState = {
        "retrieval_metadata": {
            "retrieval_decision": "continue",
        }
    }
    assert route_retrieval_decision(continue_state) == "continue"

    default_state: ResolveXState = {}
    assert route_retrieval_decision(default_state) == "continue"
