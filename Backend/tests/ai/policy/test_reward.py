"""
test_reward.py — Unit tests for RewardModel.
"""

from ai.policy.reward_model import RewardModel
from ai.policy.action_space import DecisionAction


def test_proxy_reward_autoresolve_success():
    rm = RewardModel()
    state = {
        "verification_result": {"verification_passed": True, "verification_score": 0.9},
        "requires_human": False,
        "errors": [],
    }
    reward = rm.calculate_reward(DecisionAction.AUTO_RESOLVE.value, state)
    assert reward > 0.0


def test_proxy_reward_autoresolve_unsafe():
    rm = RewardModel()
    state = {
        "verification_result": {"verification_passed": False},
        "errors": ["error1"],
    }
    reward = rm.calculate_reward(DecisionAction.AUTO_RESOLVE.value, state)
    assert reward == rm.weights.unsafe_autoresolve


def test_observed_reward():
    rm = RewardModel()
    state = {}
    outcome = {"successful_resolution": True, "customer_satisfied": True}
    reward = rm.calculate_reward("auto_resolve", state, observed_outcome=outcome)
    assert reward == rm.weights.successful_resolution + rm.weights.customer_satisfaction
