from __future__ import annotations

from typing import Any

from ai.graph.nodes.initialization import _append_error, _stage_metadata
from ai.graph.state import ResolveXState
from ai.llm.verification_generator import verify_resolution
from app.core.logger import logger


def verification_agent(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Verify whether the generated resolution is sufficiently
    supported by the ticket, diagnosis, and retrieved evidence.

    The detailed verification dimensions are preserved in state,
    while verification_passed remains the compatibility gate used
    by downstream decision logic.
    """

    ticket_text = (
        state.get("cleaned_ticket", "")
        or ""
    )

    diagnosis = (
        state.get("diagnosis", "")
        or ""
    )

    root_cause = (
        state.get("root_cause", "")
        or ""
    )

    resolution_steps = (
        state.get("resolution_steps", [])
        or []
    )

    context = (
        state.get("retrieved_context", "")
        or ""
    )

    try:

        verification_result, fallback_used = (
            verify_resolution(
                ticket_text=ticket_text,
                diagnosis=diagnosis,
                root_cause=root_cause,
                resolution_steps=resolution_steps,
                context=context,
            )
        )

        existing_fallback = bool(
            state.get(
                "fallback_used",
                False,
            )
        )

        combined_fallback = (
            existing_fallback
            or fallback_used
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=verification_agent "
            f"passed={verification_result.verification_passed} "
            f"evidence={verification_result.supported_by_evidence} "
            f"hallucination={verification_result.hallucination_detected} "
            f"complete={verification_result.complete} "
            f"policy={verification_result.policy_compliant} "
            f"correct={verification_result.resolution_correct} "
            f"confidence={verification_result.confidence:.3f} "
            f"fallback={fallback_used}"
        )

        metadata = _stage_metadata(
            state,
            "verification_agent",
        )

        return {
            # ----------------------------------------------------------
            # Existing compatibility fields
            # ----------------------------------------------------------
            "verification_passed": bool(
                verification_result.verification_passed
            ),
            "verification_reason": (
                verification_result.verification_reason
            ),
            "verification_confidence": float(
                verification_result.confidence
            ),

            # ----------------------------------------------------------
            # Detailed verification dimensions
            # ----------------------------------------------------------
            "verification_supported_by_evidence": bool(
                verification_result.supported_by_evidence
            ),
            "verification_hallucination_detected": bool(
                verification_result.hallucination_detected
            ),
            "verification_complete": bool(
                verification_result.complete
            ),
            "verification_policy_compliant": bool(
                verification_result.policy_compliant
            ),
            "verification_resolution_correct": bool(
                verification_result.resolution_correct
            ),

            # Full structured result for observability/evaluation.
            "verification_result": (
                verification_result.model_dump()
            ),

            "fallback_used": combined_fallback,
            "metadata": metadata,
        }

    except Exception as exc:

        logger.exception(
            f"Verification agent failed for "
            f"ticket_id={state.get('ticket_id')}: {exc}"
        )

        errors = _append_error(
            state,
            "verification_agent",
            exc,
        )

        metadata = _stage_metadata(
            state,
            "verification_agent",
        )

        return {
            # Existing compatibility fields
            "verification_passed": False,
            "verification_reason": (
                "Resolution verification failed. "
                "Human review is required."
            ),
            "verification_confidence": 0.0,

            # Fail closed for every detailed verification dimension.
            "verification_supported_by_evidence": False,
            "verification_hallucination_detected": True,
            "verification_complete": False,
            "verification_policy_compliant": False,
            "verification_resolution_correct": False,

            "verification_result": {
                "supported_by_evidence": False,
                "hallucination_detected": True,
                "complete": False,
                "policy_compliant": False,
                "resolution_correct": False,
                "confidence": 0.0,
                "verification_passed": False,
                "verification_reason": (
                    "Resolution verification failed. "
                    "Human review is required."
                ),
            },

            "fallback_used": True,
            "errors": errors,
            "metadata": metadata,
        }


__all__ = [
    "verification_agent",
    "verify_resolution",
]
