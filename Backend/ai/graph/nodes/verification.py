from __future__ import annotations

from typing import Any

from ai.graph.nodes.initialization import _append_error, _stage_metadata
from ai.graph.state import ResolveXState
from ai.llm.verification_generator import verify_resolution
from app.core.logger import logger


def _format_historical_tickets(
    previous_tickets: list[dict[str, Any]],
) -> str:
    """Format historical tickets into troubleshooting evidence context."""

    if not previous_tickets:
        return ""

    sections: list[str] = []

    for index, ticket in enumerate(previous_tickets, start=1):
        title = ticket.get("title", "")
        description = ticket.get("description", "")
        solution = ticket.get("solution", "")
        category = ticket.get("category", "")
        confidence = ticket.get("confidence")

        parts = [
            f"Historical Ticket {index}:",
            f"Category: {category}" if category else "",
            f"Title: {title}" if title else "",
            f"Description: {description}" if description else "",
            f"Previous Solution: {solution}" if solution else "",
            f"Confidence: {confidence}" if confidence is not None else "",
        ]

        section = "\n".join(part for part in parts if part)

        if section:
            sections.append(section)

    return "\n\n".join(sections)


def _format_conversation_history(
    conversation_history: Any,
) -> str:
    """Format conversation history into conversation context."""

    if not conversation_history:
        return ""

    if isinstance(conversation_history, str):
        return conversation_history.strip()

    if isinstance(conversation_history, (list, tuple)):
        entries: list[str] = []

        for entry in conversation_history:
            if isinstance(entry, str) and entry.strip():
                entries.append(entry.strip())

            elif isinstance(entry, dict):
                role = entry.get("role", "unknown")
                content = entry.get("content", "")

                if content:
                    entries.append(f"{str(role).capitalize()}: {content}")

        return "\n".join(entries)

    return str(conversation_history).strip()


def _build_verification_context(
    retrieved_context: str,
    previous_tickets: list[dict[str, Any]],
    conversation_history: Any,
) -> str:
    """
    Build a comprehensive verification context separating RAG evidence,
    historical ticket evidence, and conversation context.
    """

    hist_text = _format_historical_tickets(previous_tickets)
    conv_text = _format_conversation_history(conversation_history)

    if not hist_text.strip() and not conv_text.strip():
        return retrieved_context

    parts: list[str] = []

    if retrieved_context and retrieved_context.strip():
        parts.append(
            f"=== Current Retrieved Evidence ===\n{retrieved_context.strip()}"
        )

    if hist_text and hist_text.strip():
        parts.append(
            f"=== Historical Troubleshooting Evidence ===\n{hist_text.strip()}"
        )

    if conv_text and conv_text.strip():
        parts.append(
            f"=== Conversation Context ===\n{conv_text.strip()}"
        )

    return "\n\n".join(parts) if parts else retrieved_context


def verification_agent(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Verify whether the generated resolution is sufficiently
    supported by the ticket, diagnosis, retrieved evidence, and memory.

    Memory (historical tickets and conversation context) is included as
    supporting context, but historical memory does NOT automatically cause
    verification to pass.
    """

    ticket_text = (
        state.get("cleaned_ticket", "")
        or state.get("ticket_text", "")
        or ""
    )

    diagnosis = (
        state.get("diagnosis", "")
        or state.get("diagnosis_problem", "")
        or ""
    )

    root_cause = (
        state.get("root_cause", "")
        or state.get("diagnosis_root_cause", "")
        or ""
    )

    resolution_steps = (
        state.get("resolution_steps", [])
        or []
    )

    retrieved_context = (
        state.get("retrieved_context", "")
        or ""
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

    verification_context = _build_verification_context(
        retrieved_context=retrieved_context,
        previous_tickets=previous_tickets,
        conversation_history=conversation_history,
    )

    try:
        verification_result, fallback_used = verify_resolution(
            ticket_text=ticket_text,
            diagnosis=diagnosis,
            root_cause=root_cause,
            resolution_steps=resolution_steps,
            context=verification_context,
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

        metadata = _stage_metadata(
            state,
            "verification_agent",
        )

        metadata.update(
            {
                "memory": {
                    "historical_ticket_count": len(previous_tickets),
                    "historical_memory_used": bool(previous_tickets),
                    "conversation_memory_count": (
                        len(conversation_history)
                        if isinstance(conversation_history, (list, tuple))
                        else (1 if conversation_history else 0)
                    ),
                }
            }
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
            f"historical_memory={len(previous_tickets)} "
            f"conversation_memory={len(conversation_history) if isinstance(conversation_history, (list, tuple)) else (1 if conversation_history else 0)} "
            f"fallback={fallback_used}"
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

        metadata.update(
            {
                "memory": {
                    "historical_ticket_count": len(previous_tickets),
                    "historical_memory_used": bool(previous_tickets),
                    "conversation_memory_count": (
                        len(conversation_history)
                        if isinstance(conversation_history, (list, tuple))
                        else (1 if conversation_history else 0)
                    ),
                }
            }
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
