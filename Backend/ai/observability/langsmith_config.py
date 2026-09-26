"""
langsmith_config.py — Configuration and environment setup for LangSmith tracing.

Supports optional LangSmith tracing via environment variables or app settings.
Fails open safely if configuration is invalid or API key is missing.
"""

from __future__ import annotations

import os
from typing import Any

from app.core.logger import logger
from ai.config.ai_config import (
    LANGSMITH_API_KEY,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
)


def _to_bool(value: Any) -> bool:
    """
    Safely convert string/bool/int to boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on", "enabled")
    return False


def is_langsmith_enabled() -> bool:
    """
    Check if LangSmith tracing is enabled in environment or settings.

    Returns:
        bool: True if tracing is explicitly enabled.
    """
    try:
        env_val = os.environ.get("LANGSMITH_TRACING")
        if env_val is not None:
            return _to_bool(env_val)

        env_lc_val = os.environ.get("LANGCHAIN_TRACING_V2")
        if env_lc_val is not None:
            return _to_bool(env_lc_val)

        from app.config import settings

        return _to_bool(getattr(settings, "LANGSMITH_TRACING", False))
    except Exception as exc:
        logger.warning(
            f"[LangSmith] Error checking tracing enablement, defaulting to False: {exc}"
        )
        return False


def get_langsmith_api_key() -> str:
    """
    Retrieve the active LangSmith API key from env or settings.
    Explicit env vars (even empty strings) override settings defaults.
    Placeholders are treated as unconfigured (empty string).
    """
    if "LANGSMITH_API_KEY" in os.environ:
        key = os.environ["LANGSMITH_API_KEY"]
    elif "LANGCHAIN_API_KEY" in os.environ:
        key = os.environ["LANGCHAIN_API_KEY"]
    else:
        from app.config import settings

        key = getattr(settings, "LANGSMITH_API_KEY", "")

    cleaned = (key or "").strip()
    if cleaned.lower() in (
        "your_langsmith_api_key_here",
        "your_api_key_here",
        "your-langsmith-api-key-here",
        "your_key_here",
        "",
    ):
        return ""
    return cleaned


def get_langsmith_project() -> str:
    """
    Retrieve the active LangSmith project name from env or settings.
    """
    return (
        os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or LANGSMITH_PROJECT
        or "ResolveX"
    ).strip()


def get_langsmith_endpoint() -> str:
    """
    Retrieve the active LangSmith endpoint from env or settings.
    """
    return (
        os.environ.get("LANGSMITH_ENDPOINT")
        or os.environ.get("LANGCHAIN_ENDPOINT")
        or LANGSMITH_ENDPOINT
        or ""
    ).strip()


def configure_langsmith_environment() -> bool:
    """
    Configure system environment variables for LangChain / LangGraph native tracing.

    Returns:
        bool: True if tracing is active and properly configured, False otherwise.
    """
    try:
        enabled = is_langsmith_enabled()
        if not enabled:
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            return False

        api_key = get_langsmith_api_key()
        if not api_key:
            logger.warning(
                "[LangSmith] LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is missing/empty. "
                "Disabling tracing to prevent runtime failures."
            )
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            return False

        project = get_langsmith_project()
        endpoint = get_langsmith_endpoint()

        # Set standard LangChain environment variables recognized by LangChain-core/LangGraph
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGSMITH_PROJECT"] = project

        if endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = endpoint
            os.environ["LANGSMITH_ENDPOINT"] = endpoint

        logger.info(
            f"[LangSmith] Tracing configured successfully for project='{project}'"
        )
        return True

    except Exception as exc:
        logger.warning(
            f"[LangSmith] Failed to configure LangSmith environment: {exc}. "
            "Failing open with tracing disabled."
        )
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
