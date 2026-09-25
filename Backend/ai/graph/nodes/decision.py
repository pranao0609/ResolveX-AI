from __future__ import annotations

from typing import Any

from ai.graph.nodes.initialization import _stage_metadata
from ai.graph.state import ResolveXState
from app.core.logger import logger


def decision_agent(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Make the final deterministic resolution decision.

    Decision outcomes:

    - auto_resolve
    - ask_clarification
    - human_review
    - escalate

    The policy is deterministic and acts as the safety boundary
    before automated resolution. Memory provides contextual telemetry
    and features, but memory does NOT override safety policy gates.
    """

    AUTO_RESOLVE_THRESHOLD = 0.75

    diagnosis_confidence = float(
        state.get(
            "diagnosis_confidence",
            0.0,
        )
    )

    resolution_confidence = float(
        state.get(
            "resolution_confidence",
            0.0,
        )
    )

    verification_confidence = float(
        state.get(
            "verification_confidence",
            0.0,
        )
    )

    verification_passed = bool(
        state.get(
            "verification_passed",
            False,
        )
    )

    requires_human = bool(
        state.get(
            "requires_human",
            False,
        )
    )

    fallback_used = bool(
        state.get(
            "fallback_used",
            False,
        )
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    warnings = list(
        state.get(
            "warnings",
            [],
        )
    )

    missing_information = list(
        state.get(
            "diagnosis_missing_information",
            [],
        )
        or []
    )

    conversation_history = state.get(
        "conversation_history",
        [],
    )

    previous_tickets = state.get(
        "previous_tickets",
        [],
    )

    if not isinstance(previous_tickets, list):
        previous_tickets = []

    metadata = _stage_metadata(
        state,
        "decision_agent",
    )

    # ---------------------------------------------------------------
    # Compute Policy Features & Policy Metadata
    # ---------------------------------------------------------------

    conversation_count = (
        len(conversation_history)
        if isinstance(conversation_history, (list, tuple))
        else (1 if conversation_history else 0)
    )

    has_historical_solution = any(
        bool(t.get("solution"))
        for t in previous_tickets
        if isinstance(t, dict)
    )

    policy_features = dict(state.get("policy_features", {}) or {})
    policy_features.update(
        {
            "diagnosis_confidence": diagnosis_confidence,
            "resolution_confidence": resolution_confidence,
            "verification_confidence": verification_confidence,
            "memory_historical_ticket_count": float(len(previous_tickets)),
            "memory_historical_used": 1.0 if previous_tickets else 0.0,
            "memory_conversation_count": float(conversation_count),
            "memory_has_historical_solution": (
                1.0 if has_historical_solution else 0.0
            ),
        }
    )

    policy_metadata = dict(metadata)
    policy_metadata["memory"] = {
        "historical_ticket_count": len(previous_tickets),
        "historical_memory_used": bool(previous_tickets),
        "conversation_memory_count": conversation_count,
        "historical_solution_available": has_historical_solution,
    }

    metadata["memory"] = {
        "historical_ticket_count": len(previous_tickets),
        "historical_memory_used": bool(previous_tickets),
        "conversation_memory_count": conversation_count,
        "historical_solution_available": has_historical_solution,
    }

    def _build_response(
        decision_val: str,
        requires_human_val: bool,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "decision": decision_val,
            "policy_action": decision_val,
            "policy_features": policy_features,
            "policy_metadata": policy_metadata,
            "requires_human": requires_human_val,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 1. Critical execution errors
    # ---------------------------------------------------------------

    if errors:
        reason = (
            "Automated processing encountered one or more "
            "errors: "
            + "; ".join(errors)
        )

        logger.warning(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=escalate "
            f"reason=processing_errors"
        )

        return _build_response("escalate", True, reason)

    # ---------------------------------------------------------------
    # 2. Explicit human requirement
    # ---------------------------------------------------------------

    if requires_human:
        reason = (
            "The generated resolution explicitly requires "
            "human intervention."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=requires_human"
        )

        return _build_response("human_review", True, reason)

    # ---------------------------------------------------------------
    # 3. Missing information
    # ---------------------------------------------------------------

    if missing_information:
        reason = (
            "Additional information is required before the "
            "resolution can be safely completed: "
            + "; ".join(
                str(item)
                for item in missing_information
                if str(item).strip()
            )
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=ask_clarification "
            f"reason=missing_information"
        )

        return _build_response("ask_clarification", False, reason)

    # ---------------------------------------------------------------
    # 4. Verification failure
    # ---------------------------------------------------------------

    if not verification_passed:
        reason = (
            "The proposed resolution did not pass the "
            "verification gate."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=verification_failed"
        )

        return _build_response("human_review", True, reason)

    # ---------------------------------------------------------------
    # 5. Verification confidence
    # ---------------------------------------------------------------

    if verification_confidence < AUTO_RESOLVE_THRESHOLD:
        reason = (
            "Verification confidence is below the "
            f"auto-resolution threshold of "
            f"{AUTO_RESOLVE_THRESHOLD:.2f}."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=low_verification_confidence"
        )

        return _build_response("human_review", True, reason)

    # ---------------------------------------------------------------
    # 6. Diagnosis confidence
    # ---------------------------------------------------------------

    if diagnosis_confidence < AUTO_RESOLVE_THRESHOLD:
        reason = (
            "Diagnosis confidence is below the "
            f"auto-resolution threshold of "
            f"{AUTO_RESOLVE_THRESHOLD:.2f}."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=low_diagnosis_confidence"
        )

        return _build_response("human_review", True, reason)

    # ---------------------------------------------------------------
    # 7. Resolution confidence
    # ---------------------------------------------------------------

    if resolution_confidence < AUTO_RESOLVE_THRESHOLD:
        reason = (
            "Resolution confidence is below the "
            f"auto-resolution threshold of "
            f"{AUTO_RESOLVE_THRESHOLD:.2f}."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=low_resolution_confidence"
        )

        return _build_response("human_review", True, reason)

    # ---------------------------------------------------------------
    # 8. Fallback protection
    # ---------------------------------------------------------------

    if fallback_used:
        reason = (
            "One or more AI pipeline components used a fallback "
            "path. Automatic resolution is therefore disabled."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=fallback_used"
        )

        return _build_response("human_review", True, reason)

    # ---------------------------------------------------------------
    # 9. Warnings
    # ---------------------------------------------------------------

    if warnings:
        reason = (
            "The automated pipeline produced warnings that "
            "prevent automatic resolution: "
            + "; ".join(
                str(item)
                for item in warnings
                if str(item).strip()
            )
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=pipeline_warnings"
        )

        return _build_response("human_review", True, reason)

    # ---------------------------------------------------------------
    # 10. All safety gates passed
    # ---------------------------------------------------------------

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=decision_agent "
        f"decision=auto_resolve "
        f"reason=all_gates_passed"
    )

    return _build_response("auto_resolve", False, "")


__all__ = [
    "decision_agent",
]
