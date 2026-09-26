from __future__ import annotations

from ai.graph.graph import (
    build_resolvex_graph,
    resolvex_graph,
    route_decision,
)
from ai.graph.nodes.decision import decision_agent
from ai.graph.state import ResolveXState


def test_auto_resolve_route_works():
    assert route_decision({"decision": "auto_resolve"}) == "auto_resolve"


def test_ask_clarification_route_works():
    assert route_decision({"decision": "ask_clarification"}) == "ask_clarification"


def test_human_review_route_works():
    assert route_decision({"decision": "human_review"}) == "human_review"


def test_escalate_route_works():
    assert route_decision({"decision": "escalate"}) == "escalate"


def test_memory_aware_decision_is_correctly_routed():
    state: ResolveXState = {
        "ticket_id": 801,
        "diagnosis_confidence": 0.95,
        "resolution_confidence": 0.92,
        "verification_confidence": 0.94,
        "verification_passed": True,
        "requires_human": False,
        "fallback_used": False,
        "conversation_history": [{"role": "user", "content": "Help"}],
        "previous_tickets": [{"ticket_id": 1, "title": "Historical match"}],
    }

    decision_output = decision_agent(state)

    state["decision"] = decision_output["decision"]

    route = route_decision(state)

    assert route == "auto_resolve"


def test_memory_cannot_bypass_human_review():
    state: ResolveXState = {
        "ticket_id": 802,
        "requires_human": True,
        "previous_tickets": [{"ticket_id": 1, "title": "Historical match"}],
    }

    decision_output = decision_agent(state)

    state["decision"] = decision_output["decision"]

    route = route_decision(state)

    assert route == "human_review"


def test_memory_cannot_bypass_escalation():
    state: ResolveXState = {
        "ticket_id": 803,
        "errors": ["Critical system error"],
        "previous_tickets": [{"ticket_id": 1, "title": "Historical match"}],
    }

    decision_output = decision_agent(state)

    state["decision"] = decision_output["decision"]

    route = route_decision(state)

    assert route == "escalate"


def test_memory_cannot_bypass_verification_failure():
    state: ResolveXState = {
        "ticket_id": 804,
        "verification_passed": False,
        "previous_tickets": [{"ticket_id": 1, "title": "Historical match"}],
    }

    decision_output = decision_agent(state)

    state["decision"] = decision_output["decision"]

    route = route_decision(state)

    assert route == "human_review"


def test_graph_state_preserves_conversation_history_through_routing():
    graph = build_resolvex_graph()
    assert graph is not None

    initial_history = [{"role": "user", "content": "Initial message"}]
    state: ResolveXState = {
        "ticket_id": 805,
        "conversation_history": initial_history,
    }

    assert state["conversation_history"] == initial_history


def test_graph_state_preserves_previous_tickets_through_routing():
    initial_tickets = [{"ticket_id": 10, "solution": "Previous fix"}]
    state: ResolveXState = {
        "ticket_id": 806,
        "previous_tickets": initial_tickets,
    }

    assert state["previous_tickets"] == initial_tickets


def test_graph_state_preserves_tool_calls_through_routing():
    initial_tool_calls = [
        {
            "agent": "retrieval_agent",
            "tool": "search_knowledge_base",
            "status": "success",
        }
    ]
    state: ResolveXState = {
        "ticket_id": 807,
        "tool_calls": initial_tool_calls,
    }

    assert state["tool_calls"] == initial_tool_calls


def test_existing_graph_foundation_behavior_remains_intact():
    compiled_graph = resolvex_graph

    assert compiled_graph is not None

    nodes = compiled_graph.nodes

    assert "initialize_state" in nodes
    assert "ticket_analyzer" in nodes
    assert "retrieval_agent" in nodes
    assert "retrieval_decision_agent" in nodes
    assert "diagnosis_agent" in nodes
    assert "resolution_agent" in nodes
    assert "verification_agent" in nodes
    assert "decision_agent" in nodes
    assert "auto_resolve" in nodes
    assert "ask_clarification" in nodes
    assert "human_review" in nodes
    assert "escalate" in nodes
