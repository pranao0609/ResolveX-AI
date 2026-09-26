"""
test_policy_metrics.py — Unit tests for PolicyMetricsEvaluator.
"""

from ai.evaluation.models import EvaluationRecord
from ai.evaluation.policy_metrics import PolicyMetricsEvaluator


def test_policy_metrics_evaluation():
    records = [
        EvaluationRecord(
            evaluation_id="e1",
            ticket_id="t1",
            final_action="auto_resolve",
            reward=1.0,
            reward_source="observed",
            decision_latency_ms=5.0,
        ),
        EvaluationRecord(
            evaluation_id="e2",
            ticket_id="t2",
            final_action="human_review",
            reward=0.5,
            reward_source="observed",
            decision_latency_ms=10.0,
        ),
        EvaluationRecord(
            evaluation_id="e3",
            ticket_id="t3",
            final_action="escalate",
            reward=-0.5,
            reward_source="observed",
            decision_latency_ms=15.0,
        ),
    ]

    metrics = PolicyMetricsEvaluator.evaluate(records)
    assert metrics["average_reward"].value == (1.0 + 0.5 - 0.5) / 3.0
    assert metrics["human_escalation_rate"].value == 2.0 / 3.0
    assert metrics["human_review_rate"].value == 1.0 / 3.0
    assert metrics["escalation_rate"].value == 1.0 / 3.0
    assert metrics["mean_decision_latency_ms"].value == 10.0
    assert metrics["cost_per_ticket"].available is False
