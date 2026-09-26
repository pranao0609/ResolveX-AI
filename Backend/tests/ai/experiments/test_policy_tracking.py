"""
test_policy_tracking.py — Unit tests for policy experiment tracking.
"""

import os
from unittest.mock import patch

from ai.experiments.mlflow_tracker import log_policy_experiment


def test_log_policy_experiment_disabled_noop():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "false"}, clear=True):
        # Should complete without error when MLflow is disabled
        log_policy_experiment(
            policy_mode="shadow",
            bandit_type="linucb",
            linucb_alpha=0.5,
            evaluation_metrics={"avg_reward": 0.82},
        )


def test_log_policy_experiment_enabled_calls_mlflow():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}, clear=True):
        with (
            patch(
                "ai.experiments.mlflow_tracker.MLflowTracker.start_run"
            ) as mock_start,
            patch(
                "ai.experiments.mlflow_tracker.MLflowTracker.log_params"
            ) as mock_params,
            patch(
                "ai.experiments.mlflow_tracker.MLflowTracker.log_metrics"
            ) as mock_metrics,
            patch("ai.experiments.mlflow_tracker.MLflowTracker.end_run") as mock_end,
        ):

            log_policy_experiment(
                policy_mode="bandit",
                bandit_type="linucb",
                policy_random_seed=42,
                linucb_alpha=0.5,
                num_samples=100,
                evaluation_metrics={"accuracy": 0.9},
            )

            assert mock_start.called
            assert mock_params.called
            params_logged = mock_params.call_args[0][0]
            assert params_logged["policy_mode"] == "bandit"
            assert params_logged["bandit_type"] == "linucb"
            assert mock_metrics.called
            assert mock_end.called
