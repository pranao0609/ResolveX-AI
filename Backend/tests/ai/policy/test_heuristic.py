"""
test_heuristic.py — Unit tests and regression tests for baseline HeuristicDecisionPolicy.
"""

from ai.policy.heuristic import HeuristicDecisionPolicy
from ai.policy.action_space import DecisionAction


def test_heuristic_auto_resolve():
    hp = HeuristicDecisionPolicy()
    state = {
        "diagnosis_confidence": 0.85,
        "resolution_confidence": 0.90,
        "verification_confidence": 0.88,
        "verification_passed": True,
        "requires_human": False,
        "errors": [],
        "warnings": [],
    }
    action, meta = hp.predict_with_metadata(state)
    assert action == DecisionAction.AUTO_RESOLVE
    assert meta["requires_human"] is False


def test_heuristic_errors_escalate():
    hp = HeuristicDecisionPolicy()
    state = {"errors": ["Database connection error"]}
    action, meta = hp.predict_with_metadata(state)
    assert action == DecisionAction.ESCALATE
    assert meta["requires_human"] is True


def test_heuristic_requires_human():
    hp = HeuristicDecisionPolicy()
    state = {"requires_human": True}
    action, meta = hp.predict_with_metadata(state)
    assert action == DecisionAction.HUMAN_REVIEW


def test_heuristic_missing_info():
    hp = HeuristicDecisionPolicy()
    state = {"diagnosis_missing_information": ["account_id"]}
    action, meta = hp.predict_with_metadata(state)
    assert action == DecisionAction.ASK_CLARIFICATION


def test_heuristic_verification_failed():
    hp = HeuristicDecisionPolicy()
    state = {"verification_passed": False}
    action, meta = hp.predict_with_metadata(state)
    assert action == DecisionAction.HUMAN_REVIEW


def test_heuristic_low_confidence():
    hp = HeuristicDecisionPolicy()
    state = {
        "diagnosis_confidence": 0.50,
        "resolution_confidence": 0.90,
        "verification_confidence": 0.88,
        "verification_passed": True,
    }
    action, meta = hp.predict_with_metadata(state)
    assert action == DecisionAction.HUMAN_REVIEW
