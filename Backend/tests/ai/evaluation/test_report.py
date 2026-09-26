"""
test_report.py — Unit tests for EvaluationReportGenerator.
"""

from ai.evaluation.comparison import PolicyComparisonEvaluator
from ai.evaluation.report_generator import EvaluationReportGenerator
from ai.policy.dataset import DecisionPolicyDataset


def test_report_generator_json_and_markdown():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=15, seed=42)
    evaluator = PolicyComparisonEvaluator()
    report = evaluator.compare_policies(ds, dataset_name="Report Generator Test")

    json_str = EvaluationReportGenerator.to_json(report)
    assert "Report Generator Test" in json_str

    md_str = EvaluationReportGenerator.to_markdown(report)
    assert "# ResolveX Policy & Agent Evaluation Report (Phase 20)" in md_str
    assert "| **Average Reward** |" in md_str
    assert "Heuristic Baseline" in md_str
