"""
test_failure_isolation.py — Unit tests verifying MLflow failures do not crash application.
"""

import os
from unittest.mock import patch

from ai.experiments.mlflow_tracker import (
    MLflowTracker,
    log_policy_experiment,
    log_evaluation_run,
)


def test_mlflow_start_run_exception_fails_open():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}, clear=True):
        with patch(
            "mlflow.start_run", side_effect=RuntimeError("MLflow server offline!")
        ):
            run = MLflowTracker.start_run(run_name="fail_test")
            assert run is None  # Fails open safely without raising exception


def test_mlflow_log_metrics_exception_fails_open():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}, clear=True):
        with (
            patch("mlflow.active_run", return_value=True),
            patch("mlflow.log_metrics", side_effect=Exception("Network error")),
        ):
            # Should log warning internally and not raise
            MLflowTracker.log_metrics({"metric": 1.0})


def test_mlflow_log_params_exception_fails_open():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}, clear=True):
        with (
            patch("mlflow.active_run", return_value=True),
            patch("mlflow.log_params", side_effect=Exception("Connection refused")),
        ):
            # Should log warning internally and not raise
            MLflowTracker.log_params({"param": "value"})


def test_mlflow_log_artifact_exception_fails_open():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}, clear=True):
        with (
            patch("os.path.exists", return_value=True),
            patch("mlflow.active_run", return_value=True),
            patch(
                "mlflow.log_artifact", side_effect=Exception("Storage quota exceeded")
            ),
        ):
            MLflowTracker.log_artifact("test_report.json")


def test_log_policy_experiment_exception_fails_open():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}, clear=True):
        with patch(
            "ai.experiments.mlflow_tracker.MLflowTracker.start_run",
            side_effect=Exception("Fatal MLflow error"),
        ):
            # Should not raise exception
            log_policy_experiment(policy_mode="shadow")


def test_log_evaluation_run_exception_fails_open():
    with patch.dict(os.environ, {"MLFLOW_ENABLED": "true"}, clear=True):
        with patch(
            "ai.experiments.mlflow_tracker.MLflowTracker.start_run",
            side_effect=Exception("Fatal MLflow error"),
        ):
            # Should not raise exception
            log_evaluation_run({"eval_id": "123"})
