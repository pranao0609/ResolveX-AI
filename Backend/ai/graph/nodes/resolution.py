from __future__ import annotations

from typing import Any

from ai.graph.nodes.initialization import _append_error, _stage_metadata
from ai.graph.state import ResolveXState
from ai.llm.solution_generator import generate_solution
from app.core.logger import logger


def _normalize_solution_result(
    generated: Any,
) -> tuple[Any, bool]:
    """
    Normalize both the production generator contract and mocked
    single-result contract.
    """
    if isinstance(generated, tuple) and len(generated) == 2:
        return generated[0], bool(generated[1])

    return generated, False


def _get_field(obj: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] is not None:
                return obj[key]
        return default
    for key in keys:
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                return val
    return default


def _normalize_resolution_steps(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, (list, tuple)):
        return [
            str(item)
            for item in value
            if str(item).strip()
        ]

    return [str(value)]


def _build_historical_resolution_context(
    previous_tickets: list[dict[str, Any]],
) -> str:
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
            f"Historical Ticket {index}",
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


def _build_resolution_context(
    retrieved_context: str,
    previous_tickets: list[dict[str, Any]],
) -> str:
    historical_context = _build_historical_resolution_context(previous_tickets)

    if retrieved_context and historical_context:
        return (
            f"{retrieved_context}\n\n"
            "Historical troubleshooting evidence:\n"
            f"{historical_context}"
        )

    if historical_context:
        return f"Historical troubleshooting evidence:\n{historical_context}"

    return retrieved_context


def resolution_agent(state: ResolveXState) -> dict[str, Any]:
    try:
        ticket_text = (
            state.get("cleaned_ticket")
            or state.get("ticket_text", "")
        )

        retrieved_context = state.get(
            "retrieved_context",
            "",
        )

        classification = state.get(
            "classification",
            state.get("category", ""),
        )

        if isinstance(classification, dict):
            classification = classification.get("category", "") or ""

        diagnosis = state.get("diagnosis", "")
        root_cause = state.get("root_cause", "")

        diagnosis_result = state.get(
            "diagnosis_result",
            {},
        )

        if isinstance(diagnosis_result, dict):
            diagnosis = diagnosis_result.get(
                "diagnosis",
                diagnosis,
            )

            root_cause = diagnosis_result.get(
                "root_cause",
                root_cause,
            )

        conversation_history = state.get("conversation_history", [])

        previous_tickets = state.get(
            "previous_tickets",
            [],
        )

        if not isinstance(previous_tickets, list):
            previous_tickets = []

        resolution_context = _build_resolution_context(
            retrieved_context,
            previous_tickets,
        )

        generated_solution = generate_solution(
            ticket_text=ticket_text,
            context=resolution_context,
            diagnosis=diagnosis,
            root_cause=root_cause,
            classification=classification,
            conversation_history=conversation_history,
        )

        resolution, generator_fallback_used = _normalize_solution_result(
            generated_solution
        )

        raw_steps = _get_field(
            resolution,
            "resolution_steps",
            "resolution",
            "solution",
            default=[],
        )

        if isinstance(raw_steps, str):
            resolution_steps = [raw_steps] if raw_steps.strip() else []
            legacy_resolution = raw_steps
        else:
            resolution_steps = _normalize_resolution_steps(raw_steps)
            legacy_resolution = resolution_steps

        raw_evidence = _get_field(
            resolution,
            "resolution_evidence",
            "evidence",
            default=[],
        )

        if isinstance(raw_evidence, (list, tuple, set)):
            resolution_evidence = [str(item) for item in raw_evidence]
        elif raw_evidence:
            resolution_evidence = [str(raw_evidence)]
        else:
            resolution_evidence = []

        raw_confidence = _get_field(
            resolution,
            "resolution_confidence",
            "confidence",
            default=0.0,
        )

        try:
            resolution_confidence = float(raw_confidence or 0.0)
        except (ValueError, TypeError):
            resolution_confidence = 0.0

        raw_requires_human = _get_field(
            resolution,
            "requires_human",
            default=False,
        )
        requires_human = bool(raw_requires_human)

        raw_fallback = _get_field(
            resolution,
            "fallback_used",
            default=False,
        )
        generator_fallback_used = generator_fallback_used or bool(raw_fallback)

        previous_fallback_used = bool(
            state.get(
                "fallback_used",
                False,
            )
        )

        fallback_used = (
            previous_fallback_used
            or bool(generator_fallback_used)
        )

        metadata = _stage_metadata(
            state,
            "resolution",
        )

        metadata.update(
            {
                "memory": {
                    "historical_ticket_count": len(
                        previous_tickets
                    ),
                    "historical_memory_used": bool(
                        previous_tickets
                    ),
                    "conversation_memory_count": (
                        len(conversation_history)
                        if isinstance(
                            conversation_history,
                            (list, tuple),
                        )
                        else (
                            1
                            if conversation_history
                            else 0
                        )
                    ),
                }
            }
        )

        logger.info(
            "Resolution agent completed "
            "confidence=%.3f requires_human=%s "
            "fallback=%s historical_memory=%d "
            "conversation_memory=%d",
            resolution_confidence,
            requires_human,
            fallback_used,
            len(previous_tickets),
            (
                len(conversation_history)
                if isinstance(
                    conversation_history,
                    (list, tuple),
                )
                else (
                    1
                    if conversation_history
                    else 0
                )
            ),
        )

        return {
            # Structured result
            "resolution_result": resolution,

            # Phase 17 canonical fields
            "resolution_steps": resolution_steps,
            "resolution_evidence": resolution_evidence,
            "resolution_confidence": resolution_confidence,
            "requires_human": requires_human,
            "fallback_used": fallback_used,

            # Legacy aliases
            "resolution": legacy_resolution,
            "solution": legacy_resolution,
            "evidence": resolution_evidence,
            "confidence": resolution_confidence,

            # Metadata
            "metadata": metadata,
        }

    except Exception as exc:
        errors = _append_error(
            state,
            stage="resolution",
            exc=exc,
        )

        logger.exception(
            "Resolution agent failed"
        )

        metadata = _stage_metadata(state, "resolution")

        return {
            "resolution_result": None,

            "resolution_steps": [],
            "resolution_evidence": [],
            "resolution_confidence": 0.0,
            "requires_human": True,
            "fallback_used": True,

            # Legacy aliases
            "resolution": [],
            "solution": [],
            "evidence": [],
            "confidence": 0.0,

            "errors": errors,
            "metadata": metadata,
        }