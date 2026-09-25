"""
Utilities for safe ResolveX graph state handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ai.graph.state import ResolveXState


def append_error(
    state: ResolveXState,
    stage: str,
    exc: Exception,
) -> list[str]:
    """
    Return an updated error list without mutating the input state.
    """

    errors = list(
        state.get("errors", [])
    )

    errors.append(
        f"{stage}: "
        f"{type(exc).__name__}: {exc}"
    )

    return errors


def append_warning(
    state: ResolveXState,
    stage: str,
    message: str,
) -> list[str]:
    """
    Return an updated warning list without mutating the input state.
    """

    warnings = list(
        state.get("warnings", [])
    )

    warnings.append(
        f"{stage}: {message}"
    )

    return warnings


def create_state_snapshot(
    state: ResolveXState,
    stage: str,
) -> dict[str, Any]:
    """
    Create a lightweight persistence-ready snapshot.

    This does not write to a database. It creates a serializable
    representation that can later be connected to LangGraph
    checkpointing, Redis, PostgreSQL, or another persistence layer.
    """

    metadata = dict(
        state.get("metadata", {})
    )

    metadata["last_stage"] = stage

    metadata["updated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    return {
        "request_id": state.get(
            "request_id"
        ),
        "graph_run_id": state.get(
            "graph_run_id"
        ),
        "ticket_id": state.get(
            "ticket_id"
        ),
        "stage": stage,
        "metadata": metadata,
    }