"""
test_unsafe_resolution.py — Security tests for UnsafeActionGuard and SecretLeakageGuard in SafetyPolicyGuard.
"""

import pytest
from ai.policy.safety_guard import SafetyPolicyGuard
from ai.policy.action_space import DecisionAction


def test_unsafe_action_guard_overrides_destructive_drop_database():
    """Verify destructive 'drop database' action forces SafetyPolicyGuard to override auto_resolve to human_review."""
    guard = SafetyPolicyGuard()
    state = {
        "ticket_text": "Request for DB production schema drop privilege",
        "diagnosis": "User requests root access to DROP DATABASE production",
        "resolution_steps": ["Execute DROP DATABASE production"],
        "verification_result": {"verification_passed": True},
        "diagnosis_confidence": 0.9,
        "resolution_confidence": 0.9,
        "verification_confidence": 0.9,
    }

    res = guard.evaluate_safety("auto_resolve", state)
    assert res.was_overridden is True
    assert res.final_action == DecisionAction.HUMAN_REVIEW.value
    assert res.gate_failed == "unsafe_action_guard"


def test_secret_leakage_guard_overrides_auto_resolve():
    """Verify model output containing leaked API key forces SafetyPolicyGuard to override to human_review."""
    guard = SafetyPolicyGuard()
    state = {
        "ticket_text": "Need API key for service integration",
        "diagnosis": "User requested API key",
        "resolution_steps": ["Use gsk_live_api_key_secret_123 to authenticate"],
        "verification_result": {"verification_passed": True},
        "diagnosis_confidence": 0.9,
        "resolution_confidence": 0.9,
        "verification_confidence": 0.9,
    }

    res = guard.evaluate_safety("auto_resolve", state)
    assert res.was_overridden is True
    assert res.final_action == DecisionAction.HUMAN_REVIEW.value
    assert res.gate_failed == "secret_leakage_guard"
