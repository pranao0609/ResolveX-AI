from __future__ import annotations

from ai.graph.state import ResolveXState


def route_retrieval_decision(
    state: ResolveXState,
) -> str:
    """
    Route the graph after retrieval quality evaluation.

    Returns:
        continue -> proceed to diagnosis
        retry    -> execute retrieval again
    """

    retrieval_metadata = state.get(
        "retrieval_metadata",
        {},
    ) or {}

    decision = retrieval_metadata.get(
        "retrieval_decision",
        "continue",
    )

    if decision == "retry":
        return "retry"

    return "continue"


def route_decision(
    state: ResolveXState,
) -> str:
    """
    Route the graph after the decision node.

    Explicit routing values keep the graph deterministic
    and easy to test.
    """

    decision = state.get(
        "decision",
        "human_review",
    )

    if decision == "auto_resolve":
        return "auto_resolve"

    if decision == "ask_clarification":
        return "ask_clarification"

    if decision == "escalate":
        return "escalate"

    if decision == "human_review":
        return "human_review"

    # Fail closed for unknown decision values.
    return "human_review"


__all__ = [
    "route_retrieval_decision",
    "route_decision",
]
