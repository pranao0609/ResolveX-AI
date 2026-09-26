"""
test_mlflow_tracker.py — Unit tests for MLflowTracker methods with mocks.
"""

import os
from unittest.mock import MagicMock, patch

from ai.experiments.mlflow_tracker import MLflowTracker


def test_tracker_noop_when_disabled():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "false"}, clear=True):
        assert MLflowTracker.start_run() is None
        # Methods should return safely without raising exceptions
        MLflowTracker.log_params({"param": 1})
        MLflowTracker.log_metrics({"metric": 1.0})
        MLflowTracker.set_tags({"tag": "val"})
        MLflowTracker.log_artifact("non_existent_file.txt")
        MLflowTracker.end_run()


def test_tracker_logs_params_metrics_tags_when_enabled():
    env_vars = {"MLFLOW_ENABLED": "true", "MLFLOW_EXPERIMENT_NAME": "ResolveX-Test"}
    with patch.dict(os.environ, env_vars, clear=True):
        with (
            patch("mlflow.start_run") as mock_start,
            patch("mlflow.log_params") as mock_params,
            patch("mlflow.log_metrics") as mock_metrics,
            patch("mlflow.set_tags") as mock_tags,
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.end_run") as mock_end,
        ):

            MLflowTracker.start_run(run_name="test_run")
            assert mock_start.called

            MLflowTracker.log_params({"policy_mode": "shadow", "alpha": 0.5})
            assert mock_params.called

            MLflowTracker.log_metrics({"avg_reward": 0.85, "latency_ms": 12.5})
            assert mock_metrics.called

            MLflowTracker.set_tags({"env": "test"})
            assert mock_tags.called

            MLflowTracker.end_run()
            assert mock_end.called


def test_tracker_does_not_log_fabricated_zeros_for_none():
    env_vars = {"MLFLOW_ENABLED": "true"}
    with patch.dict(os.environ, env_vars, clear=True):
        with (
            patch("mlflow.log_metrics") as mock_metrics,
            patch("mlflow.active_run", return_value=MagicMock()),
        ):

            metrics_with_none = {
                "available_reward": 0.9,
                "unavailable_ground_truth": None,
            }

            MLflowTracker.log_metrics(metrics_with_none)

            assert mock_metrics.called
            logged_dict = mock_metrics.call_args[0][0]
            assert "available_reward" in logged_dict
            assert "unavailable_ground_truth" not in logged_dict
            assert 0.0 not in logged_dict.values()
