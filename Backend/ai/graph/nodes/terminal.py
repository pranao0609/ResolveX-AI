from __future__ import annotations

from typing import Any

from ai.graph.nodes.initialization import _stage_metadata
from ai.graph.state import ResolveXState
from app.core.logger import logger


def auto_resolve(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Terminal node for safely verified automatic resolution.
    """

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=auto_resolve "
        f"decision=auto_resolve"
    )

    metadata = _stage_metadata(
        state,
        "auto_resolve",
        completed=True,
    )

    return {
        "decision": "auto_resolve",
        "requires_human": False,
        "metadata": metadata,
    }


def human_review(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Terminal node for tickets requiring human review.
    """

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=human_review "
        f"decision=human_review"
    )

    metadata = _stage_metadata(
        state,
        "human_review",
        completed=True,
    )

    return {
        "decision": "human_review",
        "requires_human": True,
        "metadata": metadata,
    }


def ask_clarification(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Terminal node for tickets requiring additional information
    from the requester before resolution can safely continue.
    """

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=ask_clarification "
        f"decision=ask_clarification"
    )

    metadata = _stage_metadata(
        state,
        "ask_clarification",
        completed=True,
    )

    return {
        "decision": "ask_clarification",
        "requires_human": False,
        "metadata": metadata,
    }


def escalate(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Terminal node for tickets requiring escalation.
    """

    logger.warning(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=escalate "
        f"decision=escalate"
    )

    metadata = _stage_metadata(
        state,
        "escalate",
        completed=True,
    )

    return {
        "decision": "escalate",
        "requires_human": True,
        "metadata": metadata,
    }


__all__ = [
    "auto_resolve",
    "human_review",
    "ask_clarification",
    "escalate",
]
