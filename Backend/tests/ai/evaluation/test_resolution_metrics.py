"""
test_resolution_metrics.py — Unit tests for ResolutionMetricsEvaluator.
"""

from ai.evaluation.models import EvaluationRecord
from ai.evaluation.resolution_metrics import ResolutionMetricsEvaluator


def test_resolution_metrics_evaluation():
    records = [
        EvaluationRecord(
            evaluation_id="e1",
            ticket_id="t1",
            final_action="auto_resolve",
            auto_resolved=True,
            resolution_success=True,
        ),
        EvaluationRecord(
            evaluation_id="e2",
            ticket_id="t2",
            final_action="human_review",
            auto_resolved=False,
            human_reviewed=True,
            human_accepted=True,
        ),
    ]

    metrics = ResolutionMetricsEvaluator.evaluate(records)
    assert metrics["auto_resolution_rate"].value == 0.5
    assert metrics["resolution_success_rate"].value == 1.0
    assert metrics["human_acceptance_rate"].value == 1.0
    assert metrics["escalation_accuracy"].available is False
