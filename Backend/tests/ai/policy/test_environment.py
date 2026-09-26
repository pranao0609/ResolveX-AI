"""
test_environment.py — Unit tests for ContextualBanditEnvironment.
"""

from ai.policy.dataset import DecisionPolicyDataset
from ai.policy.environment import ContextualBanditEnvironment


def test_environment_stepping():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=5, seed=42)
    env = ContextualBanditEnvironment(ds, random_seed=42)

    assert env.has_next()
    res = env.step(0)  # Auto resolve
    assert res.selected_action == "auto_resolve"
    assert res.selected_action_index == 0


def test_environment_reset():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=5, seed=42)
    env = ContextualBanditEnvironment(ds, random_seed=42)

    env.step(0)
    assert env.current_idx == 1
    env.reset()
    assert env.current_idx == 0
