"""
test_models.py — Unit tests for EvaluationRecord and MetricValue.
"""

from ai.evaluation.models import EvaluationRecord, MetricValue


def test_metric_value_available():
    mv = MetricValue(
        value=0.85,
        available=True,
        source="telemetry",
        sample_count=10,
        description="Test Metric",
    )
    assert mv.available is True
    assert mv.value == 0.85
    assert mv.to_dict()["source"] == "telemetry"


def test_metric_value_unavailable():
    mv = MetricValue.unavailable("Tool Call Accuracy")
    assert mv.available is False
    assert mv.value is None
    assert mv.source == "missing_ground_truth"


def test_evaluation_record_creation():
    rec = EvaluationRecord(
        evaluation_id="eval_101",
        ticket_id="ticket_101",
        evaluation_source="synthetic",
        task_success=True,
        final_action="auto_resolve",
    )
    assert rec.evaluation_id == "eval_101"
    assert rec.final_action == "auto_resolve"
    assert rec.to_dict()["ticket_id"] == "ticket_101"
