from ai.graph.graph import (
    ask_clarification,
    auto_resolve,
    escalate,
    human_review,
    route_decision,
    resolvex_graph,
)


def test_graph_contains_all_phase18_terminal_nodes():
    graph_nodes = resolvex_graph.get_graph().nodes

    assert "auto_resolve" in graph_nodes
    assert "ask_clarification" in graph_nodes
    assert "human_review" in graph_nodes
    assert "escalate" in graph_nodes


def test_route_auto_resolve():
    assert route_decision({"decision": "auto_resolve"}) == "auto_resolve"


def test_route_ask_clarification():
    assert route_decision({"decision": "ask_clarification"}) == "ask_clarification"


def test_route_human_review():
    assert route_decision({"decision": "human_review"}) == "human_review"


def test_route_escalate():
    assert route_decision({"decision": "escalate"}) == "escalate"


def test_route_unknown_defaults_to_human_review():
    assert route_decision({"decision": "unknown_decision"}) == "human_review"


def test_auto_resolve_terminal_node():
    result = auto_resolve(
        {
            "ticket_id": 1,
        }
    )

    assert result["decision"] == "auto_resolve"
    assert result["requires_human"] is False


def test_ask_clarification_terminal_node():
    result = ask_clarification(
        {
            "ticket_id": 2,
        }
    )

    assert result["decision"] == "ask_clarification"
    assert result["requires_human"] is False


def test_human_review_terminal_node():
    result = human_review(
        {
            "ticket_id": 3,
        }
    )

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True


def test_escalate_terminal_node():
    result = escalate(
        {
            "ticket_id": 4,
        }
    )

    assert result["decision"] == "escalate"
    assert result["requires_human"] is True
