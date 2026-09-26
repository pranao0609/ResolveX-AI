"""
tracer.py — Native LangGraph integration & tracing helper functions.

Constructs sanitized RunnableConfig dictionary for LangGraph executions
and provides fail-open helpers for telemetry and trace visibility.
"""

from __future__ import annotations

import os
from typing import Any

from app.core.logger import logger
from ai.observability.langsmith_config import (
    configure_langsmith_environment,
    is_langsmith_enabled,
)
from ai.observability.metadata import sanitize_metadata


def get_langgraph_config(
    ticket_id: Any = None,
    extra_metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Construct a RunnableConfig dictionary for LangGraph graph execution.

    Args:
        ticket_id: Primary identifier for the ticket being processed.
        extra_metadata: Additional safe metadata to attach to root run.
        tags: Tags to associate with the LangSmith trace run.

    Returns:
        dict compatible with LangGraph invoke(state, config=...)
    """
    try:
        # Check and configure environment variables for LangSmith
        tracing_active = configure_langsmith_environment()

        base_tags = ["resolvex", "ticket-graph"]
        if tags:
            base_tags.extend(tags)

        safe_ticket_id = str(ticket_id) if ticket_id is not None else "unknown"

        root_metadata = {
            "ticket_id": safe_ticket_id,
            "environment": os.environ.get("RESOLVEX_ENV", "development"),
            "policy_mode": "shadow",
            "feature_version": "v1",
            "tracing_enabled": tracing_active,
        }

        if extra_metadata:
            root_metadata.update(extra_metadata)

        sanitized_meta = sanitize_metadata(root_metadata)

        config: dict[str, Any] = {
            "run_name": f"ResolveX Ticket {safe_ticket_id}",
            "configurable": {
                "thread_id": safe_ticket_id,
            },
            "tags": sorted(list(set(base_tags))),
            "metadata": sanitized_meta,
        }

        return config

    except Exception as exc:
        logger.warning(
            f"[Observability] Failed to build LangGraph config: {exc}. "
            "Falling back to basic config."
        )
        return {
            "tags": ["resolvex"],
            "metadata": {
                "ticket_id": str(ticket_id) if ticket_id is not None else "unknown"
            },
        }


def extract_trace_telemetry(state: dict[str, Any]) -> dict[str, Any]:
    """
    Extract observable telemetry summary from state for tracing or logging.

    Exposes node, LLM, tool, retrieval, prompt, model, token, latency, and policy telemetry
    in a sanitized format without modifying state.
    """
    try:
        telemetry = {
            "prompt_versions": state.get("prompt_versions", {}),
            "model_versions": state.get("model_versions", {}),
            "latency_ms": state.get("latency", {}),
            "tool_calls_count": len(state.get("tool_calls", [])),
            "retrieval_metadata": state.get("retrieval_metadata", {}),
            "policy_action": state.get("policy_action"),
            "policy_metadata": state.get("policy_metadata", {}),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
        }

        # Calculate token usage if available in state or tool records
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        state_tokens = state.get("token_usage") or (
            state.get("metadata", {}).get("token_usage")
            if isinstance(state.get("metadata"), dict)
            else None
        )
        if isinstance(state_tokens, dict):
            prompt_tokens = state_tokens.get("prompt_tokens", 0)
            completion_tokens = state_tokens.get("completion_tokens", 0)
            total_tokens = state_tokens.get("total_tokens", 0)

        if total_tokens == 0:
            for tool_rec in state.get("tool_calls", []):
                if isinstance(tool_rec, dict):
                    u = (
                        tool_rec.get("usage")
                        if isinstance(tool_rec.get("usage"), dict)
                        else tool_rec
                    )
                    if isinstance(u, dict):
                        prompt_tokens += u.get("prompt_tokens", 0)
                        completion_tokens += u.get("completion_tokens", 0)
                        total_tokens += u.get("total_tokens", 0)

        if total_tokens > 0:
            telemetry["token_usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        return sanitize_metadata(telemetry)
    except Exception as exc:
        logger.warning(f"[Observability] Error extracting trace telemetry: {exc}")
        return {}
