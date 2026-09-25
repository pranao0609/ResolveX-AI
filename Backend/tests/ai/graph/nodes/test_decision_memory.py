from __future__ import annotations

from ai.graph.nodes.decision import decision_agent


def _passing_state():
    return {
        "ticket_id": 701,
        "diagnosis_confidence": 0.90,
        "resolution_confidence": 0.88,
        "verification_confidence": 0.92,
        "verification_passed": True,
        "requires_human": False,
        "fallback_used": False,
        "errors": [],
        "warnings": [],
        "conversation_history": [],
        "previous_tickets": [],
    }


def test_decision_consumes_memory_metadata():
    state = _passing_state()
    state["conversation_history"] = [{"role": "user", "content": "Help"}]
    state["previous_tickets"] = [
        {"ticket_id": 10, "title": "VPN issue", "solution": "Reset credentials"}
    ]

    result = decision_agent(state)

    assert result["decision"] == "auto_resolve"
    assert result["policy_features"]["memory_historical_ticket_count"] == 1.0
    assert result["policy_features"]["memory_historical_used"] == 1.0
    assert result["policy_features"]["memory_conversation_count"] == 1.0
    assert result["policy_features"]["memory_has_historical_solution"] == 1.0
    assert result["policy_metadata"]["memory"]["historical_ticket_count"] == 1
    assert result["policy_metadata"]["memory"]["historical_memory_used"] is True


def test_decision_preserves_existing_auto_resolve_behavior():
    state = _passing_state()

    result = decision_agent(state)

    assert result["decision"] == "auto_resolve"
    assert result["requires_human"] is False
    assert result["escalation_reason"] == ""
    assert result["policy_action"] == "auto_resolve"


def test_decision_preserves_human_review_behavior():
    state = _passing_state()
    state["verification_passed"] = False

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True
    assert "verification" in result["escalation_reason"].lower()


def test_decision_preserves_ask_clarification_behavior():
    state = _passing_state()
    state["diagnosis_missing_information"] = ["Log details required"]

    result = decision_agent(state)

    assert result["decision"] == "ask_clarification"
    assert result["requires_human"] is False
    assert "Log details required" in result["escalation_reason"]


def test_decision_preserves_escalation_behavior():
    state = _passing_state()
    state["errors"] = ["Fatal retrieval error"]

    result = decision_agent(state)

    assert result["decision"] == "escalate"
    assert result["requires_human"] is True
    assert "Fatal retrieval error" in result["escalation_reason"]


def test_historical_memory_does_not_bypass_safety_gates():
    # Even when rich historical memory exists, if verification failed or requires_human is True, it MUST NOT auto_resolve.
    state = _passing_state()
    state["verification_passed"] = False
    state["previous_tickets"] = [
        {"ticket_id": 99, "title": "Exact match", "solution": "Fix applied"}
    ]

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True


def test_low_verification_confidence_still_blocks_auto_resolve():
    state = _passing_state()
    state["verification_confidence"] = 0.50
    state["previous_tickets"] = [{"ticket_id": 1, "solution": "Fix"}]

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True


def test_requires_human_still_forces_human_review():
    state = _passing_state()
    state["requires_human"] = True
    state["conversation_history"] = [{"role": "user", "content": "Help"}]

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True


def test_verification_failure_still_forces_human_review():
    state = _passing_state()
    state["verification_passed"] = False

    result = decision_agent(state)

    assert result["decision"] == "human_review"
    assert result["requires_human"] is True


def test_decision_metadata_contains_memory_information():
    state = _passing_state()
    state["previous_tickets"] = [
        {"ticket_id": 5, "title": "Crash", "solution": "Reboot"}
    ]

    result = decision_agent(state)

    assert "memory" in result["metadata"]
    assert result["metadata"]["memory"]["historical_ticket_count"] == 1
    assert result["metadata"]["memory"]["historical_memory_used"] is True
    assert result["metadata"]["memory"]["historical_solution_available"] is True


def test_policy_features_remain_backward_compatible():
    state = _passing_state()
    state["policy_features"] = {"custom_feature": 1.0}

    result = decision_agent(state)

    assert result["policy_features"]["custom_feature"] == 1.0
    assert "diagnosis_confidence" in result["policy_features"]
    assert "resolution_confidence" in result["policy_features"]
    assert "verification_confidence" in result["policy_features"]
