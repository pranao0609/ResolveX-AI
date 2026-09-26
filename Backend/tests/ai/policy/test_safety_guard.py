"""
test_safety_guard.py — Unit tests for SafetyPolicyGuard.
"""

from ai.policy.safety_guard import SafetyPolicyGuard
from ai.policy.action_space import DecisionAction


def test_safety_guard_pass():
    guard = SafetyPolicyGuard()
    state = {
        "diagnosis_confidence": 0.85,
        "resolution_confidence": 0.90,
        "verification_confidence": 0.88,
        "verification_passed": True,
        "requires_human": False,
    }
    res = guard.evaluate_safety("auto_resolve", state)
    assert res.final_action == "auto_resolve"
    assert res.was_overridden is False
    assert res.safety_gate_passed is True


def test_safety_guard_override_unverifed():
    guard = SafetyPolicyGuard()
    state = {"verification_passed": False}
    # Bandit recommends auto_resolve, but verification failed -> override to human_review
    res = guard.evaluate_safety("auto_resolve", state)
    assert res.recommended_action == "auto_resolve"
    assert res.final_action == "human_review"
    assert res.was_overridden is True
    assert res.safety_gate_passed is False
    assert res.gate_failed == "verification_failed"


def test_safety_guard_override_errors():
    guard = SafetyPolicyGuard()
    state = {"errors": ["LLM API Timeout"]}
    res = guard.evaluate_safety("auto_resolve", state)
    assert res.final_action == "escalate"
    assert res.was_overridden is True
    assert res.gate_failed == "errors"


def test_safety_guard_override_missing_info():
    guard = SafetyPolicyGuard()
    state = {"diagnosis_missing_information": ["user_email"]}
    res = guard.evaluate_safety("auto_resolve", state)
    assert res.final_action == "ask_clarification"
    assert res.was_overridden is True


def test_safety_guard_invalid_recommendation():
    guard = SafetyPolicyGuard()
    state = {}
    res = guard.evaluate_safety("invalid_action_xyz", state)
    assert res.final_action == "human_review"
    assert res.was_overridden is True
    assert res.gate_failed == "invalid_recommendation"
