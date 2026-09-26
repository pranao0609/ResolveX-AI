"""
test_comparison.py — Unit tests for PolicyComparisonEvaluator comparing policies on same dataset.
"""

from ai.evaluation.comparison import PolicyComparisonEvaluator, PolicyComparisonReport
from ai.policy.dataset import DecisionPolicyDataset
from ai.policy.linucb import LinUCBPolicy
from ai.policy.thompson import ThompsonSamplingPolicy


def test_policy_comparison_on_same_dataset():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=25, seed=42)
    evaluator = PolicyComparisonEvaluator(auto_resolve_threshold=0.75)

    linucb = LinUCBPolicy(alpha=0.5, random_seed=42)
    linucb.fit(ds)

    thompson = ThompsonSamplingPolicy(random_seed=42)
    thompson.fit(ds)

    report = evaluator.compare_policies(
        dataset=ds,
        linucb_policy=linucb,
        thompson_policy=thompson,
        dataset_name="Test Comparison Dataset",
        dataset_source="synthetic",
    )

    assert isinstance(report, PolicyComparisonReport)
    assert report.dataset_info["sample_count"] == 25
    assert "heuristic" in report.policy_summaries
    assert "linucb" in report.policy_summaries
    assert "thompson_sampling" in report.policy_summaries

    assert "agreement_rate_heuristic_linucb" in report.comparison
    assert "heuristic_vs_linucb" in report.confusion_matrices
    assert "total_overrides" in report.safety_overrides
