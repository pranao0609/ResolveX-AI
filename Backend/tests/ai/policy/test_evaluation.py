"""
test_evaluation.py — Unit tests for OfflinePolicyEvaluator.
"""

from ai.policy.dataset import DecisionPolicyDataset
from ai.policy.evaluation import OfflinePolicyEvaluator
from ai.policy.linucb import LinUCBPolicy
from ai.policy.heuristic import HeuristicDecisionPolicy


def test_evaluator_heuristic():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=20, seed=42)
    evaluator = OfflinePolicyEvaluator()
    heuristic = HeuristicDecisionPolicy()

    report = evaluator.evaluate(heuristic, ds)
    assert report.sample_count == 20
    assert report.baseline_agreement_rate == 1.0
    assert report.ope_metrics["ope_available"] is True


def test_evaluator_linucb():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=20, seed=42)
    linucb = LinUCBPolicy()
    linucb.fit(ds)

    evaluator = OfflinePolicyEvaluator()
    report = evaluator.evaluate(linucb, ds)

    assert report.sample_count == 20
    assert "mean_observed_reward" in report.reward_metrics
    assert "override_count" in report.safety_metrics
