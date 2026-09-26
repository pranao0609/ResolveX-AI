"""
test_evaluation_tracking.py — Unit tests for evaluation run tracking.
"""

import os
from unittest.mock import MagicMock, patch

from ai.experiments.mlflow_tracker import log_evaluation_run


def test_log_evaluation_run_disabled_noop():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "false"}, clear=True):
        report_dict = {
            "eval_id": "eval_001",
            "num_tickets": 50,
            "policy_mode": "shadow",
            "metrics": {"average_reward": 0.88, "unavailable_metric": None},
        }
        log_evaluation_run(report_dict)


def test_log_evaluation_run_enabled_logs_available_metrics():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}, clear=True):
        with (
            patch("mlflow.start_run") as mock_start,
            patch("mlflow.log_params") as mock_params,
            patch("mlflow.log_metrics") as mock_metrics,
            patch("mlflow.log_dict") as mock_dict,
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.end_run") as mock_end,
        ):

            report_dict = {
                "eval_id": "eval_101",
                "num_tickets": 25,
                "policy_mode": "bandit",
                "metrics": {
                    "average_reward": 0.91,
                    "auto_resolution_rate": 0.80,
                    "ground_truth_score": None,  # unavailable metric
                },
            }

            log_evaluation_run(report_dict, run_name="test_eval")

            assert mock_start.called
            assert mock_params.called
            assert mock_metrics.called
            logged_metrics = mock_metrics.call_args[0][0]
            assert "average_reward" in logged_metrics
            assert "ground_truth_score" not in logged_metrics
            assert mock_dict.called
            assert mock_end.called
