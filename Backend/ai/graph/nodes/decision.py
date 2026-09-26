from __future__ import annotations

from typing import Any

from ai.graph.state import ResolveXState
from ai.policy.policy_manager import PolicyManager
from app.config import settings
from app.core.logger import logger


def decision_agent(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Make the final resolution decision using PolicyManager.

    Supports modes:
    - heuristic (default)
    - shadow
    - bandit

    In all modes, SafetyPolicyGuard enforces authoritative deterministic safety limits.
    """
    mode = getattr(settings, "POLICY_MODE", "heuristic")
    bandit_type = getattr(settings, "BANDIT_TYPE", "linucb")
    auto_resolve_threshold = getattr(settings, "AUTO_RESOLVE_THRESHOLD", 0.75)
    linucb_alpha = getattr(settings, "LINUCB_ALPHA", 0.5)
    random_seed = getattr(settings, "POLICY_RANDOM_SEED", 42)

    manager = PolicyManager(
        mode=mode,
        bandit_type=bandit_type,
        auto_resolve_threshold=auto_resolve_threshold,
        linucb_alpha=linucb_alpha,
        random_seed=random_seed,
    )

    response = manager.evaluate(state)

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=decision_agent "
        f"mode={mode} "
        f"decision={response['decision']} "
        f"reason={response['escalation_reason']}"
    )

    return response


__all__ = [
    "decision_agent",
]
