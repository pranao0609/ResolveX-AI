from ai.graph.graph import (
    decision_agent,
    route_decision,
)
from ai.graph.graph import resolvex_graph

def test_graph_compiles():
    assert resolvex_graph is not None


def test_graph_has_decision_routing():
    graph = resolvex_graph.get_graph()

    node_names = set(
        graph.nodes.keys()
    )

    assert "decision_agent" in node_names
    assert "auto_resolve" in node_names
    assert "human_review" in node_names
    assert "escalate" in node_names

def base_state():
    return {
        "ticket_id": 1,
        "diagnosis_confidence": 0.90,
        "resolution_confidence": 0.88,
        "verification_confidence": 0.92,
        "verification_passed": True,
        "requires_human": False,
        "fallback_used": False,
        "errors": [],
        "warnings": [],
    }


def test_decision_auto_resolve():
    result = decision_agent(
        base_state()
    )

    assert result["decision"] == "auto_resolve"
    assert result["requires_human"] is False
    assert result["escalation_reason"] == ""


def test_decision_human_review_when_verification_fails():
    state = base_state()

    state["verification_passed"] = False

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True

    assert (
        "verification"
        in result["escalation_reason"].lower()
    )


def test_decision_human_review_when_human_required():
    state = base_state()

    state["requires_human"] = True

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True


def test_decision_human_review_low_diagnosis_confidence():
    state = base_state()

    state["diagnosis_confidence"] = 0.60

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True

    assert (
        "diagnosis"
        in result["escalation_reason"].lower()
    )


def test_decision_human_review_low_resolution_confidence():
    state = base_state()

    state["resolution_confidence"] = 0.60

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True

    assert (
        "resolution"
        in result["escalation_reason"].lower()
    )


def test_decision_human_review_low_verification_confidence():
    state = base_state()

    state["verification_confidence"] = 0.60

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True

    assert (
        "verification"
        in result["escalation_reason"].lower()
    )


def test_decision_human_review_when_fallback_used():
    state = base_state()

    state["fallback_used"] = True

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True

    assert (
        "fallback"
        in result["escalation_reason"].lower()
    )


def test_decision_escalates_when_errors_exist():
    state = base_state()

    state["errors"] = [
        "retrieval_agent: RuntimeError: FAISS unavailable"
    ]

    result = decision_agent(state)

    assert result["decision"] == "escalate"
    assert result["requires_human"] is True

    assert (
        "error"
        in result["escalation_reason"].lower()
    )


def test_route_auto_resolve():
    result = route_decision(
        {
            "decision": "auto_resolve"
        }
    )

    assert result == "auto_resolve"


def test_route_escalate():
    result = route_decision(
        {
            "decision": "escalate"
        }
    )

    assert result == "escalate"


def test_route_human_review():
    result = route_decision(
        {
            "decision": "human_review"
        }
    )

    assert result == "human_review"


def test_route_unknown_defaults_to_human_review():
    result = route_decision(
        {
            "decision": "unknown_decision"
        }
    )

    assert result == "human_review"