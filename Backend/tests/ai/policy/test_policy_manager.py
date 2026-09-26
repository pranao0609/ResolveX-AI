"""
test_policy_manager.py — Integration unit tests for PolicyManager.
"""

from ai.policy.policy_manager import PolicyManager, PolicyMode


def test_policy_manager_heuristic_mode():
    pm = PolicyManager(mode=PolicyMode.HEURISTIC)
    state = {
        "diagnosis_confidence": 0.85,
        "resolution_confidence": 0.90,
        "verification_confidence": 0.88,
        "verification_passed": True,
    }
    res = pm.evaluate(state)
    assert res["decision"] == "auto_resolve"
    assert res["policy_action"] == "auto_resolve"
    assert res["policy_metadata"]["policy_mode"] == "heuristic"


def test_policy_manager_shadow_mode():
    pm = PolicyManager(mode=PolicyMode.SHADOW, bandit_type="linucb")
    state = {
        "diagnosis_confidence": 0.85,
        "resolution_confidence": 0.90,
        "verification_confidence": 0.88,
        "verification_passed": True,
    }
    res = pm.evaluate(state)
    # In shadow mode, baseline action determines real decision
    assert res["decision"] == "auto_resolve"
    assert res["policy_metadata"]["policy_mode"] == "shadow"
    assert "recommended_action" in res["policy_metadata"]
    assert "baseline_action" in res["policy_metadata"]


def test_policy_manager_bandit_mode_with_safety_override():
    pm = PolicyManager(mode=PolicyMode.BANDIT, bandit_type="linucb")
    # State with verification failure
    state = {
        "diagnosis_confidence": 0.85,
        "resolution_confidence": 0.90,
        "verification_passed": False,
    }
    res = pm.evaluate(state)
    # Even if bandit recommended auto_resolve, safety guard MUST override to human_review
    assert res["decision"] == "human_review"
    assert res["requires_human"] is True
    assert res["policy_metadata"]["policy_mode"] == "bandit"
