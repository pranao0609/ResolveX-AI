"""
test_agent_metrics.py — Unit tests for AgentMetricsEvaluator.
"""

from ai.evaluation.models import EvaluationRecord
from ai.evaluation.agent_metrics import AgentMetricsEvaluator


def test_agent_metrics_evaluation():
    records = [
        EvaluationRecord(
            evaluation_id="e1",
            ticket_id="t1",
            task_success=True,
            agent_latency_ms=100.0,
            tool_calls=2,
        ),
        EvaluationRecord(
            evaluation_id="e2",
            ticket_id="t2",
            task_success=False,
            agent_latency_ms=200.0,
            tool_calls=3,
        ),
    ]

    metrics = AgentMetricsEvaluator.evaluate(records)
    assert metrics["task_success_rate"].value == 0.5
    assert metrics["task_success_rate"].available is True
    assert metrics["mean_agent_latency_ms"].value == 150.0
    assert metrics["tool_call_accuracy"].available is False  # ground truth missing
