"""
test_mlflow_config.py — Unit tests for MLflow configuration & environment setup.
"""

import os
from unittest.mock import patch

from ai.experiments.mlflow_tracker import MLflowTracker


def test_mlflow_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        assert MLflowTracker.is_enabled() is False


def test_mlflow_enabled_truthy():
    truthy_vals = ["true", "True", "1", "yes", "ON"]
    for val in truthy_vals:
        with patch.dict(os.environ, {"MLFLOW_ENABLED": val}, clear=True):
            assert (
                MLflowTracker.is_enabled() is True
            ), f"Failed for MLFLOW_ENABLED={val}"


def test_mlflow_enabled_falsy():
    falsy_vals = ["false", "False", "0", "no", "OFF"]
    for val in falsy_vals:
        with patch.dict(os.environ, {"MLFLOW_ENABLED": val}, clear=True):
            assert (
                MLflowTracker.is_enabled() is False
            ), f"Failed for MLFLOW_ENABLED={val}"


def test_get_tracking_uri_and_experiment():
    env_vars = {
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_EXPERIMENT_NAME": "ResolveX-Test",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        assert MLflowTracker.get_tracking_uri() == "http://localhost:5000"
        assert MLflowTracker.get_experiment_name() == "ResolveX-Test"
