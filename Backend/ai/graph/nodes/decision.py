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
    before automated resolution.
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

    metadata = _stage_metadata(
        state,
        "decision_agent",
    )

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

        return {
            "decision": "escalate",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

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

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

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

        return {
            "decision": "ask_clarification",
            "requires_human": False,
            "escalation_reason": reason,
            "metadata": metadata,
        }

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

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 5. Verification confidence
    # ---------------------------------------------------------------

    if (
        verification_confidence
        < AUTO_RESOLVE_THRESHOLD
    ):

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

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 6. Diagnosis confidence
    # ---------------------------------------------------------------

    if (
        diagnosis_confidence
        < AUTO_RESOLVE_THRESHOLD
    ):

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

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 7. Resolution confidence
    # ---------------------------------------------------------------

    if (
        resolution_confidence
        < AUTO_RESOLVE_THRESHOLD
    ):

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

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

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

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

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

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 10. All safety gates passed
    # ---------------------------------------------------------------

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=decision_agent "
        f"decision=auto_resolve "
        f"reason=all_gates_passed"
    )

    return {
        "decision": "auto_resolve",
        "requires_human": False,
        "escalation_reason": "",
        "metadata": metadata,
    }


__all__ = [
    "decision_agent",
]
