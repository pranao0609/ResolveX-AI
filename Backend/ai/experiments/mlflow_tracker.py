"""
mlflow_tracker.py — MLflow MLOps experiment & run tracking abstraction for ResolveX.

Provides reproducible logging for policies, evaluation metrics, model metadata,
and report artifacts with 100% failure isolation.

When MLFLOW_ENABLED is False (default), all calls become safe no-ops.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional

from app.core.logger import logger
from ai.config.ai_config import (
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_RUN_NAME,
    MLFLOW_TRACKING_URI,
)
from ai.observability.metadata import sanitize_metadata

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore
    HAS_MLFLOW = False


def _to_bool(value: Any) -> bool:
    """Safely convert string/bool/int to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on", "enabled")
    return False


def get_git_commit_sha() -> str | None:
    """
    Safely retrieve the current Git commit SHA without crashing if Git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if sha:
                return sha
    except Exception:
        pass
    return None


class MLflowTracker:
    """
    Dedicated lightweight abstraction for MLflow experiment tracking.
    """

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if MLflow tracking is enabled in env or app config."""
        if not HAS_MLFLOW:
            return False
        try:
            env_val = os.environ.get("MLFLOW_ENABLED")
            if env_val is not None:
                return _to_bool(env_val)
            return _to_bool(MLFLOW_ENABLED)
        except Exception as exc:
            logger.warning(
                f"[MLflow] Error checking enablement: {exc}. Defaulting to False."
            )
            return False

    @classmethod
    def get_tracking_uri(cls) -> str:
        """Retrieve tracking URI from env or settings."""
        return (
            os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI or ""
        ).strip()

    @classmethod
    def get_experiment_name(cls) -> str:
        """Retrieve experiment name from env or settings."""
        return (
            os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or MLFLOW_EXPERIMENT_NAME
            or "ResolveX"
        ).strip()

    @classmethod
    def configure(cls) -> bool:
        """
        Configure tracking URI and experiment for MLflow.
        Returns True if successfully configured, False otherwise.
        """
        if not cls.is_enabled():
            return False

        try:
            uri = cls.get_tracking_uri()
            if uri:
                mlflow.set_tracking_uri(uri)

            exp_name = cls.get_experiment_name()
            if exp_name:
                mlflow.set_experiment(exp_name)

            return True
        except Exception as exc:
            logger.warning(
                f"[MLflow] Failed to configure MLflow tracking: {exc}. "
                "Failing open without disrupting ResolveX execution."
            )
            return False

    @classmethod
    def start_run(
        cls,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ) -> Any:
        """
        Start an MLflow run safely.
        """
        if not cls.is_enabled():
            return None

        try:
            cls.configure()

            if experiment_name:
                mlflow.set_experiment(experiment_name)

            name = (
                run_name or os.environ.get("MLFLOW_RUN_NAME") or MLFLOW_RUN_NAME or None
            )

            # Build system tags
            sys_tags: Dict[str, Any] = {
                "project": "ResolveX",
                "phase": "23",
                "environment": os.environ.get("RESOLVEX_ENV", "development"),
                "pipeline_version": "v1",
            }

            git_sha = get_git_commit_sha()
            if git_sha:
                sys_tags["git_commit"] = git_sha

            if tags:
                sys_tags.update(tags)

            sanitized_tags = sanitize_metadata(sys_tags)
            # Ensure tag values are strings for MLflow API
            str_tags = {
                str(k): str(v) for k, v in sanitized_tags.items() if v is not None
            }

            run = mlflow.start_run(
                run_name=name,
                tags=str_tags,
                nested=nested,
            )
            logger.info(
                f"[MLflow] Started run id='{run.info.run_id}' name='{name or 'default'}'"
            )
            return run
        except Exception as exc:
            logger.warning(
                f"[MLflow] Failed to start MLflow run: {exc}. " "Failing open safely."
            )
            return None

    @classmethod
    def log_params(cls, params: Dict[str, Any]) -> None:
        """
        Log dictionary of parameters safely.
        """
        if not cls.is_enabled():
            return

        try:
            if not isinstance(params, dict):
                return

            sanitized = sanitize_metadata(params)
            # Convert values to strings for MLflow parameter compatibility
            stringified = {
                str(k): str(v) for k, v in sanitized.items() if v is not None
            }

            if stringified and mlflow.active_run():
                mlflow.log_params(stringified)
        except Exception as exc:
            logger.warning(f"[MLflow] Failed to log parameters: {exc}")

    @classmethod
    def log_metrics(cls, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log numeric metrics safely.
        Skipped/unavailable (None) metrics are NOT logged as zeros.
        """
        if not cls.is_enabled():
            return

        try:
            if not isinstance(metrics, dict):
                return

            numeric_metrics: Dict[str, float] = {}
            for k, v in metrics.items():
                if v is None:
                    continue  # Respect unavailable metrics — DO NOT log fabricated 0.0
                if isinstance(v, (int, float, bool)) and not isinstance(v, bool):
                    numeric_metrics[str(k)] = float(v)

            if numeric_metrics and mlflow.active_run():
                mlflow.log_metrics(numeric_metrics, step=step)
        except Exception as exc:
            logger.warning(f"[MLflow] Failed to log metrics: {exc}")

    @classmethod
    def set_tags(cls, tags: Dict[str, Any]) -> None:
        """
        Log tags to active run safely.
        """
        if not cls.is_enabled():
            return

        try:
            if not isinstance(tags, dict):
                return

            sanitized = sanitize_metadata(tags)
            str_tags = {str(k): str(v) for k, v in sanitized.items() if v is not None}

            if str_tags and mlflow.active_run():
                mlflow.set_tags(str_tags)
        except Exception as exc:
            logger.warning(f"[MLflow] Failed to set tags: {exc}")

    @classmethod
    def log_artifact(cls, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log local file artifact safely.
        """
        if not cls.is_enabled():
            return

        try:
            if not os.path.exists(local_path):
                logger.warning(f"[MLflow] Artifact file not found: {local_path}")
                return

            if mlflow.active_run():
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as exc:
            logger.warning(f"[MLflow] Failed to log artifact '{local_path}': {exc}")

    @classmethod
    def log_dict(cls, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """
        Log dictionary as JSON artifact safely.
        """
        if not cls.is_enabled():
            return

        try:
            if not isinstance(dictionary, dict):
                return

            sanitized = sanitize_metadata(dictionary)
            if mlflow.active_run():
                mlflow.log_dict(sanitized, artifact_file)
        except Exception as exc:
            logger.warning(
                f"[MLflow] Failed to log dict artifact '{artifact_file}': {exc}"
            )

    @classmethod
    def end_run(cls, status: str = "FINISHED") -> None:
        """
        End active MLflow run safely.
        """
        if not cls.is_enabled():
            return

        try:
            if mlflow.active_run():
                mlflow.end_run(status=status)
                logger.info(f"[MLflow] Ended run with status='{status}'")
        except Exception as exc:
            logger.warning(f"[MLflow] Failed to end MLflow run: {exc}")


def log_policy_experiment(
    policy_mode: str,
    bandit_type: str = "linucb",
    policy_random_seed: int = 42,
    linucb_alpha: float = 0.5,
    num_samples: Optional[int] = None,
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    High-level tracking helper for Phase 19 Contextual Bandit policy experiments.
    """
    if not MLflowTracker.is_enabled():
        return

    try:
        run_name = f"policy-{policy_mode}-{bandit_type}"
        MLflowTracker.start_run(run_name=run_name)

        params = {
            "policy_mode": policy_mode,
            "bandit_type": bandit_type,
            "policy_random_seed": policy_random_seed,
            "linucb_alpha": linucb_alpha,
            "feature_version": "v1",
            "action_space_version": "v1",
            "safety_policy_version": "v1",
            "reward_model_version": "v1",
        }
        if num_samples is not None:
            params["num_samples"] = num_samples
        if extra_params:
            params.update(extra_params)

        MLflowTracker.log_params(params)

        if evaluation_metrics:
            MLflowTracker.log_metrics(evaluation_metrics)

        MLflowTracker.end_run()
    except Exception as exc:
        logger.warning(f"[MLflow] Failed to log policy experiment: {exc}")


def log_evaluation_run(
    report_dict: Dict[str, Any],
    json_path: Optional[str] = None,
    md_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    """
    High-level tracking helper for Phase 20 RL & Agent Evaluation runs.
    """
    if not MLflowTracker.is_enabled():
        return

    try:
        name = run_name or f"eval-{report_dict.get('eval_id', 'run')}"
        MLflowTracker.start_run(run_name=name)

        # Log parameters
        params = {
            "eval_id": report_dict.get("eval_id"),
            "num_tickets": report_dict.get("num_tickets"),
            "policy_mode": report_dict.get("policy_mode"),
            "policy_type": report_dict.get("policy_type"),
            "feature_version": report_dict.get("feature_version", "v1"),
        }
        MLflowTracker.log_params(params)

        # Log metrics if available
        metrics = report_dict.get("metrics", {})
        if isinstance(metrics, dict):
            MLflowTracker.log_metrics(metrics)

        # Log report artifacts
        if json_path and os.path.exists(json_path):
            MLflowTracker.log_artifact(json_path, artifact_path="reports")
        if md_path and os.path.exists(md_path):
            MLflowTracker.log_artifact(md_path, artifact_path="reports")

        MLflowTracker.log_dict(report_dict, "evaluation_report.json")

        MLflowTracker.end_run()
    except Exception as exc:
        logger.warning(f"[MLflow] Failed to log evaluation run: {exc}")
