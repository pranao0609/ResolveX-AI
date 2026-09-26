"""
test_runner.py — Unit tests for EvaluationRunner.
"""

from ai.evaluation.runner import EvaluationRunner
from ai.evaluation.hitl_feedback import HumanFeedback
from ai.policy.dataset import DecisionPolicyDataset


def test_runner_evaluate_state():
    runner = EvaluationRunner()
    state = {
        "ticket_id": "t_99",
        "diagnosis_confidence": 0.85,
        "resolution_confidence": 0.90,
        "verification_confidence": 0.88,
        "verification_passed": True,
    }

    record = runner.evaluate_state(state, mode="heuristic")
    assert record.ticket_id == "t_99"
    assert record.final_action == "auto_resolve"
    assert record.evaluation_source == "live_execution"


def test_runner_record_and_evaluate_hitl():
    runner = EvaluationRunner()
    fb = HumanFeedback(
        feedback_id="fb_99",
        ticket_id="t_99",
        accepted=True,
        original_action="auto_resolve",
    )
    runner.record_human_feedback(fb)

    state = {
        "ticket_id": "t_99",
        "diagnosis_confidence": 0.85,
        "resolution_confidence": 0.90,
        "verification_confidence": 0.88,
        "verification_passed": True,
    }

    record = runner.evaluate_state(state, mode="heuristic")
    assert record.evaluation_source == "hitl"
    assert record.reward_source == "hitl_observed"
    assert record.human_accepted is True


def test_runner_evaluate_dataset():
    runner = EvaluationRunner()
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=10, seed=42)

    report = runner.evaluate_dataset(ds, dataset_name="Runner Dataset")
    assert report.dataset_info["name"] == "Runner Dataset"
    assert report.dataset_info["sample_count"] == 10
