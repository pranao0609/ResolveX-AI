"""
test_fail_closed_security.py — Security tests for fail-closed system behavior under errors and attacks.
"""

import pytest
from ai.policy.safety_guard import SafetyPolicyGuard
from ai.policy.action_space import DecisionAction


def test_fail_closed_on_errors():
    """Verify processing errors force fallback to escalate."""
    guard = SafetyPolicyGuard()
    state = {
        "errors": ["Unhandled LLM exception occurred"],
        "verification_result": {"verification_passed": True},
    }
    res = guard.evaluate_safety("auto_resolve", state)
    assert res.final_action == DecisionAction.ESCALATE.value


def test_fail_closed_on_missing_information():
    """Verify missing diagnostic information forces fallback to ask_clarification."""
    guard = SafetyPolicyGuard()
    state = {
        "diagnosis_missing_information": ["Need CPU utilization details"],
        "verification_result": {"verification_passed": True},
    }
    res = guard.evaluate_safety("auto_resolve", state)
    assert res.final_action == DecisionAction.ASK_CLARIFICATION.value


def test_fail_closed_on_verification_failure():
    """Verify verification failure forces fallback to human_review."""
    guard = SafetyPolicyGuard()
    state = {
        "verification_result": {"verification_passed": False},
    }
    res = guard.evaluate_safety("auto_resolve", state)
    assert res.final_action == DecisionAction.HUMAN_REVIEW.value
