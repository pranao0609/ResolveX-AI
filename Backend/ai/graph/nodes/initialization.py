from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ai.graph.state import ResolveXState


def _ensure_runtime_metadata(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Initialize graph execution metadata.

    Creates stable identifiers for a single graph execution
    without introducing an external persistence dependency.
    """

    request_id = state.get("request_id")

    if not request_id:
        request_id = str(uuid.uuid4())

    graph_run_id = state.get("graph_run_id")

    if not graph_run_id:
        graph_run_id = str(uuid.uuid4())

    metadata = dict(
        state.get("metadata", {})
    )

    if "started_at" not in metadata:
        metadata["started_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

    metadata["ticket_id"] = state.get(
        "ticket_id"
    )

    metadata["last_stage"] = (
        "initialize_state"
    )

    return {
        "request_id": request_id,
        "graph_run_id": graph_run_id,
        "metadata": metadata,
    }


def initialize_graph_state(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Initialize runtime state before the first agent executes.
    """

    runtime = _ensure_runtime_metadata(
        state
    )

    existing_errors = list(
        state.get("errors", [])
    )

    existing_warnings = list(
        state.get("warnings", [])
    )

    return {
        "request_id": runtime["request_id"],
        "graph_run_id": runtime["graph_run_id"],
        "metadata": runtime["metadata"],
        "errors": existing_errors,
        "warnings": existing_warnings,
        "fallback_used": bool(
            state.get("fallback_used", False)
        ),
        "conversation_history": list(
            state.get(
                "conversation_history",
                [],
            )
            or []
        ),
        "previous_tickets": list(
            state.get(
                "previous_tickets",
                [],
            )
            or []
        ),
        "tool_calls": list(
            state.get(
                "tool_calls",
                [],
            )
            or []
        ),
    }


def _stage_metadata(
    state: ResolveXState,
    stage: str,
    completed: bool = False,
) -> dict[str, Any]:
    """
    Create updated metadata for a graph stage.

    Metadata is intentionally lightweight and serializable so that
    it can later be connected to LangGraph checkpointing,
    Redis, PostgreSQL, or another persistence backend.
    """

    metadata = dict(
        state.get("metadata", {})
    )

    metadata["last_stage"] = stage

    metadata["updated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    if completed:
        metadata["completed_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

    return metadata


def _append_error(
    state: ResolveXState,
    stage: str,
    exc: Exception | str,
) -> list[str]:
    """
    Append an error while preserving all previous errors.
    """

    errors = list(
        state.get("errors", [])
    )

    if isinstance(exc, Exception):
        errors.append(
            f"{stage}: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        errors.append(
            f"{stage}: {exc}"
        )

    return errors


__all__ = [
    "_ensure_runtime_metadata",
    "initialize_graph_state",
    "_stage_metadata",
    "_append_error",
]
