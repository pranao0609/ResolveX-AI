from __future__ import annotations

from typing import Any

from ai.graph.nodes.initialization import (
    _append_error,
    _stage_metadata,
)
from ai.graph.state import ResolveXState
from ai.graph.tools.executor import execute_tool
from ai.graph.tools.memory_registry import (
    build_memory_registry,
)
from ai.llm.diagnosis_generator import generate_diagnosis
from ai.memory.conversation_memory import (
    ConversationMemory,
)
from app.core.logger import logger
from app.database import SessionLocal


# ---------------------------------------------------------------------
# Memory Tool Registry
# ---------------------------------------------------------------------

memory_registry = build_memory_registry()


# ---------------------------------------------------------------------
# Historical Memory Context
# ---------------------------------------------------------------------


def _build_historical_memory_context(
    previous_tickets: list[dict[str, Any]],
) -> str:
    """
    Convert historical ticket memory into compact
    evidence context for diagnosis.
    """

    if not previous_tickets:
        return ""

    sections: list[str] = []

    for index, ticket in enumerate(
        previous_tickets,
        start=1,
    ):
        sections.append(
            "\n".join(
                [
                    f"[Historical Ticket {index}]",
                    (
                        f"Category: "
                        f"{ticket.get('category', '')}"
                    ),
                    (
                        f"Title: "
                        f"{ticket.get('title', '')}"
                    ),
                    (
                        f"Problem: "
                        f"{ticket.get('description', '')}"
                    ),
                    (
                        f"Previous Solution: "
                        f"{ticket.get('solution', '')}"
                    ),
                    (
                        f"Confidence: "
                        f"{ticket.get('confidence')}"
                    ),
                ]
            )
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------
# Conversation Memory Helpers
# ---------------------------------------------------------------------


def _conversation_contains_ticket(
    conversation_memory: ConversationMemory,
    ticket_id: int | str | None,
) -> bool:
    """
    Check whether the current ticket has already been
    added to short-term conversation memory.
    """

    if ticket_id is None:
        return False

    for event in conversation_memory.history:
        if not isinstance(event, dict):
            continue

        metadata = event.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            continue

        if metadata.get("ticket_id") == ticket_id:
            return True

    return False


# ---------------------------------------------------------------------
# Diagnosis Agent
# ---------------------------------------------------------------------


def diagnosis_agent(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Diagnosis Agent.

    Determines what is actually happening and identifies the
    most likely evidence-supported root cause.

    The agent consumes:

        - cleaned ticket
        - classification
        - retrieved evidence
        - short-term conversation memory
        - historical resolved-ticket memory

    It does NOT generate resolution instructions.

    Phase 19.3 memory responsibilities:

        1. Load short-term conversation memory.
        2. Search historical resolved tickets.
        3. Record memory tool telemetry.
        4. Inject historical evidence into diagnosis context.
        5. Preserve memory state in ResolveXState.
        6. Continue safely when historical memory is unavailable.
    """

    # =================================================================
    # 1. Read Existing State
    # =================================================================

    ticket_text = (
        state.get(
            "cleaned_ticket",
            "",
        )
        or ""
    )

    context = (
        state.get(
            "retrieved_context",
            "",
        )
        or ""
    )

    classification = (
        state.get(
            "category",
            "",
        )
        or ""
    )

    # =================================================================
    # 2. Initialize Memory State
    # =================================================================

    conversation_memory = (
        ConversationMemory.from_state(
            state
        )
    )

    previous_tickets = list(
        state.get(
            "previous_tickets",
            [],
        )
        or []
    )

    tool_calls = list(
        state.get(
            "tool_calls",
            [],
        )
        or []
    )

    ticket_id = state.get(
        "ticket_id"
    )

    # ---------------------------------------------------------------
    # Add current ticket to short-term memory only once
    # ---------------------------------------------------------------

    if (
        ticket_text
        and not _conversation_contains_ticket(
            conversation_memory,
            ticket_id,
        )
    ):
        conversation_memory.append(
            role="user",
            content=ticket_text,
            ticket_id=ticket_id,
        )

    # =================================================================
    # 3. Search Historical Ticket Memory
    # =================================================================

    memory_result: dict[str, Any] | None = None
    memory_error: str | None = None

    db = None

    try:
        db = SessionLocal()

        memory_result, memory_record = (
            execute_tool(
                memory_registry,
                agent="diagnosis_agent",
                tool_name="search_memory",
                kwargs={
                    "db": db,
                    "query": ticket_text,
                    "category": classification,
                    "exclude_ticket_id": ticket_id,
                    "resolved_only": True,
                    "limit": 5,
                },
            )
        )

        # -----------------------------------------------------------
        # Preserve normalized tool telemetry
        # -----------------------------------------------------------

        tool_calls.append(
            memory_record
        )

        # -----------------------------------------------------------
        # Tool execution itself can fail without raising because
        # ToolRegistry.execute() normalizes exceptions into ToolResult.
        # -----------------------------------------------------------

        if not isinstance(
            memory_result,
            dict,
        ):
            memory_error = (
                "search_memory returned "
                "an invalid result."
            )

        elif (
            memory_result.get("status")
            != "success"
        ):
            memory_error = (
                memory_result.get(
                    "error"
                )
                or "search_memory failed."
            )

        else:
            previous_tickets = list(
                memory_result.get(
                    "data",
                    [],
                )
                or []
            )

    except Exception as exc:
        memory_error = (
            f"{type(exc).__name__}: {exc}"
        )

        logger.warning(
            "Historical memory unavailable "
            f"for ticket_id={ticket_id}: "
            f"{memory_error}"
        )

        # -----------------------------------------------------------
        # If SessionLocal() or another unexpected operation fails
        # before execute_tool() creates a normalized record, create
        # explicit telemetry so the failure remains observable.
        # -----------------------------------------------------------

        tool_calls.append(
            {
                "agent": "diagnosis_agent",
                "tool": "search_memory",
                "category": "read",
                "input": {
                    "query": ticket_text,
                    "category": classification,
                    "exclude_ticket_id": ticket_id,
                    "resolved_only": True,
                    "limit": 5,
                },
                "result_count": 0,
                "status": "error",
                "latency_ms": 0.0,
                "error": memory_error,
            }
        )

    finally:
        if db is not None:
            try:
                db.close()
            except Exception as exc:
                logger.warning(
                    "Failed to close memory "
                    f"database session for "
                    f"ticket_id={ticket_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

    # =================================================================
    # 4. Build Diagnosis Context
    # =================================================================

    historical_context = (
        _build_historical_memory_context(
            previous_tickets
        )
    )

    diagnosis_context = context

    if historical_context:
        if diagnosis_context:
            diagnosis_context = (
                f"{diagnosis_context}\n\n"
                "=== Historical Ticket Evidence ===\n"
                f"{historical_context}"
            )
        else:
            diagnosis_context = (
                "=== Historical Ticket Evidence ===\n"
                f"{historical_context}"
            )

    # =================================================================
    # 5. Generate Diagnosis
    # =================================================================

    try:
        diagnosis_result, fallback_used = (
            generate_diagnosis(
                ticket_text=ticket_text,
                context=diagnosis_context,
                classification=classification,
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

        # =============================================================
        # 6. Add Diagnosis Response to Conversation Memory
        # =============================================================

        conversation_memory.append(
            role="assistant",
            content=diagnosis_result.problem,
            ticket_id=ticket_id,
            root_cause=(
                diagnosis_result.possible_root_cause
            ),
            confidence=float(
                diagnosis_result.confidence
            ),
        )

        # =============================================================
        # 7. Metadata
        # =============================================================

        logger.info(
            f"ticket_id={ticket_id} "
            f"stage=diagnosis_agent "
            f"confidence="
            f"{diagnosis_result.confidence:.3f} "
            f"evidence_count="
            f"{len(diagnosis_result.evidence)} "
            f"missing_information_count="
            f"{len(diagnosis_result.missing_information)} "
            f"historical_memory_count="
            f"{len(previous_tickets)} "
            f"conversation_memory_count="
            f"{conversation_memory.count} "
            f"fallback={fallback_used}"
        )

        metadata = _stage_metadata(
            state,
            "diagnosis_agent",
        )

        metadata["diagnosis"] = {
            "confidence": float(
                diagnosis_result.confidence
            ),
            "evidence_count": len(
                diagnosis_result.evidence
            ),
            "missing_information_count": len(
                diagnosis_result.missing_information
            ),
            "fallback_used": fallback_used,
        }

        metadata["memory"] = {
            "historical_ticket_count": (
                len(previous_tickets)
            ),
            "historical_memory_used": bool(
                previous_tickets
            ),
            "conversation_memory_count": (
                conversation_memory.count
            ),
            "memory_error": memory_error,
        }

        # =============================================================
        # 8. Successful State Update
        # =============================================================

        return {
            # ---------------------------------------------------------
            # Structured diagnosis contract
            # ---------------------------------------------------------

            "diagnosis_result": (
                diagnosis_result.model_dump()
            ),

            # ---------------------------------------------------------
            # Explicit diagnosis state fields
            # ---------------------------------------------------------

            "diagnosis_problem": (
                diagnosis_result.problem
            ),

            "diagnosis_root_cause": (
                diagnosis_result.possible_root_cause
            ),

            "diagnosis_evidence": (
                diagnosis_result.evidence
            ),

            "diagnosis_missing_information": (
                diagnosis_result.missing_information
            ),

            "diagnosis_confidence": float(
                diagnosis_result.confidence
            ),

            # ---------------------------------------------------------
            # Backward-compatible fields
            # ---------------------------------------------------------

            "diagnosis": (
                diagnosis_result.problem
            ),

            "root_cause": (
                diagnosis_result.possible_root_cause
            ),

            # ---------------------------------------------------------
            # Runtime / fallback
            # ---------------------------------------------------------

            "fallback_used": combined_fallback,

            # ---------------------------------------------------------
            # Phase 19.3 memory state
            # ---------------------------------------------------------

            "conversation_history": (
                conversation_memory.to_context()
            ),

            "previous_tickets": (
                previous_tickets
            ),

            "tool_calls": tool_calls,

            # ---------------------------------------------------------
            # Metadata
            # ---------------------------------------------------------

            "metadata": metadata,
        }

    # =================================================================
    # 9. Diagnosis Failure
    # =================================================================

    except Exception as exc:
        logger.exception(
            "Diagnosis agent failed for "
            f"ticket_id={ticket_id}: {exc}"
        )

        errors = _append_error(
            state,
            "diagnosis_agent",
            exc,
        )

        metadata = _stage_metadata(
            state,
            "diagnosis_agent",
        )

        metadata["memory"] = {
            "historical_ticket_count": (
                len(previous_tickets)
            ),
            "historical_memory_used": bool(
                previous_tickets
            ),
            "conversation_memory_count": (
                conversation_memory.count
            ),
            "memory_error": memory_error,
        }

        fallback_result = {
            "problem": (
                "The reported support issue requires "
                "further investigation."
            ),
            "possible_root_cause": (
                "No validated root cause could be "
                "established from the available evidence."
            ),
            "evidence": [],
            "missing_information": [
                "Additional diagnostic information "
                "is required."
            ],
            "confidence": 0.0,
        }

        return {
            # ---------------------------------------------------------
            # Structured diagnosis contract
            # ---------------------------------------------------------

            "diagnosis_result": (
                fallback_result
            ),

            # ---------------------------------------------------------
            # Explicit diagnosis state fields
            # ---------------------------------------------------------

            "diagnosis_problem": (
                fallback_result["problem"]
            ),

            "diagnosis_root_cause": (
                fallback_result[
                    "possible_root_cause"
                ]
            ),

            "diagnosis_evidence": [],

            "diagnosis_missing_information": (
                fallback_result[
                    "missing_information"
                ]
            ),

            "diagnosis_confidence": 0.0,

            # ---------------------------------------------------------
            # Backward compatibility
            # ---------------------------------------------------------

            "diagnosis": (
                fallback_result["problem"]
            ),

            "root_cause": (
                fallback_result[
                    "possible_root_cause"
                ]
            ),

            # ---------------------------------------------------------
            # Runtime
            # ---------------------------------------------------------

            "fallback_used": True,

            "errors": errors,

            # ---------------------------------------------------------
            # Preserve memory even on diagnosis failure
            # ---------------------------------------------------------

            "conversation_history": (
                conversation_memory.to_context()
            ),

            "previous_tickets": (
                previous_tickets
            ),

            "tool_calls": tool_calls,

            "metadata": metadata,
        }


__all__ = [
    "diagnosis_agent",
    "generate_diagnosis",
]