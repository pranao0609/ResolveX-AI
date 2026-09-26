"""
executor.py — Production execution adapter for the ResolveX LangGraph.

This module is responsible for:

    Ticket ORM object
        ↓
    ResolveXState
        ↓
    resolvex_graph.invoke()
        ↓
    Legacy/API-compatible pipeline result

The graph itself remains independent of the database layer.
"""

from __future__ import annotations

from typing import Any

from app.core.logger import logger
from ai.confidence.confidence_engine import compute_confidence
from ai.explainability.explainer import explain

from ai.graph.graph import resolvex_graph
from ai.observability import get_langgraph_config


def execute_resolvex_graph(
    ticket: Any,
    include_evaluation_details: bool = False,
) -> dict:
    """
    Execute the complete ResolveX LangGraph for a ticket.

    Args:
        ticket:
            Ticket ORM object or compatible object.

        include_evaluation_details:
            When True, include graph state details useful for
            debugging/evaluation.

    Returns:
        Dictionary compatible with the existing ResolutionService
        pipeline output contract.
    """

    ticket_id = getattr(ticket, "id", None)

    logger.info(
        f"[GraphExecutor] Starting graph execution " f"for ticket_id={ticket_id}"
    )

    ticket_text = _build_ticket_text(ticket)
    attachment_paths = getattr(
        ticket,
        "attachment_paths",
        None,
    )

    initial_state = {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "attachment_paths": attachment_paths,
        "errors": [],
        "warnings": [],
        "fallback_used": False,
        "metadata": {},
    }

    # ------------------------------------------------------------------
    # Execute LangGraph with Observability Config
    # ------------------------------------------------------------------

    run_config = get_langgraph_config(
        ticket_id=ticket_id,
    )

    graph_state = resolvex_graph.invoke(
        initial_state,
        config=run_config,
    )

    logger.info(
        f"[GraphExecutor] Graph completed "
        f"for ticket_id={ticket_id} "
        f"decision={graph_state.get('decision')} "
        f"graph_run_id={graph_state.get('graph_run_id')}"
    )

    # ------------------------------------------------------------------
    # Extract graph outputs
    # ------------------------------------------------------------------

    category = graph_state.get(
        "category",
        "other",
    )

    category_confidence = float(
        graph_state.get(
            "category_confidence",
            0.0,
        )
    )

    diagnosis = graph_state.get(
        "diagnosis",
        "",
    )

    root_cause = graph_state.get(
        "root_cause",
        "",
    )

    resolution_steps = list(
        graph_state.get(
            "resolution_steps",
            [],
        )
        or []
    )

    evidence = list(
        graph_state.get(
            "evidence",
            [],
        )
        or []
    )

    diagnosis_confidence = float(
        graph_state.get(
            "diagnosis_confidence",
            0.0,
        )
    )

    resolution_confidence = float(
        graph_state.get(
            "resolution_confidence",
            0.0,
        )
    )

    verification_confidence = float(
        graph_state.get(
            "verification_confidence",
            0.0,
        )
    )

    verification_passed = bool(
        graph_state.get(
            "verification_passed",
            False,
        )
    )

    requires_human = bool(
        graph_state.get(
            "requires_human",
            False,
        )
    )

    fallback_used = bool(
        graph_state.get(
            "fallback_used",
            False,
        )
    )

    decision = graph_state.get(
        "decision",
        "escalate",
    )

    errors = list(
        graph_state.get(
            "errors",
            [],
        )
        or []
    )

    warnings = list(
        graph_state.get(
            "warnings",
            [],
        )
        or []
    )

    # ------------------------------------------------------------------
    # Composite confidence
    # ------------------------------------------------------------------

    context_docs = list(
        graph_state.get(
            "retrieved_documents",
            [],
        )
        or []
    )

    similarity_score = _compute_similarity_score(context_docs)

    confidence = compute_confidence(
        similarity_score=similarity_score,
        llm_score=resolution_confidence,
        classification_score=category_confidence,
    )

    # ------------------------------------------------------------------
    # Human-readable solution
    # ------------------------------------------------------------------

    solution = _format_solution(
        diagnosis=diagnosis,
        root_cause=root_cause,
        resolution_steps=resolution_steps,
        evidence=evidence,
    )

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    explanation = _build_explanation(
        ticket_text=graph_state.get(
            "cleaned_ticket",
            ticket_text,
        ),
        category=category,
        context_docs=context_docs,
        solution=solution,
        confidence=confidence,
    )

    # ------------------------------------------------------------------
    # Normalize final decision
    # ------------------------------------------------------------------

    auto_resolved = (
        decision == "auto_resolve"
        and not requires_human
        and not fallback_used
        and not errors
    )

    escalated = not auto_resolved

    # ------------------------------------------------------------------
    # Build production-compatible result
    # ------------------------------------------------------------------

    result = {
        # Existing API fields
        "solution": solution,
        "diagnosis": diagnosis,
        "root_cause": root_cause,
        "resolution_steps": resolution_steps,
        "evidence": evidence,
        "llm_confidence": resolution_confidence,
        "requires_human": requires_human,
        "confidence": confidence,
        "category": category,
        "explanation": explanation,
        "fallback_used": fallback_used,
        # Graph-specific decision information
        "decision": decision,
        "auto_resolved": auto_resolved,
        "escalated_to_human": escalated,
        "escalation_reason": graph_state.get(
            "escalation_reason",
            "",
        ),
        # Agent confidence signals
        "category_confidence": category_confidence,
        "diagnosis_confidence": diagnosis_confidence,
        "resolution_confidence": resolution_confidence,
        "verification_confidence": verification_confidence,
        "verification_passed": verification_passed,
        # Graph execution metadata
        "request_id": graph_state.get("request_id"),
        "graph_run_id": graph_state.get("graph_run_id"),
        "graph_metadata": graph_state.get(
            "metadata",
            {},
        ),
        # Reliability information
        "errors": errors,
        "warnings": warnings,
    }

    # ------------------------------------------------------------------
    # Evaluation/debug information
    # ------------------------------------------------------------------

    if include_evaluation_details:
        result["evaluation"] = {
            "cleaned_text": graph_state.get(
                "cleaned_ticket",
                "",
            ),
            "context_docs": context_docs,
            "retrieval_metadata": graph_state.get(
                "retrieval_metadata",
                {},
            ),
            "category_confidence": category_confidence,
            "diagnosis_confidence": diagnosis_confidence,
            "resolution_confidence": resolution_confidence,
            "verification_confidence": verification_confidence,
            "verification_passed": verification_passed,
            "decision": decision,
            "request_id": graph_state.get("request_id"),
            "graph_run_id": graph_state.get("graph_run_id"),
            "graph_metadata": graph_state.get(
                "metadata",
                {},
            ),
        }

    return result


# ======================================================================
# Helpers
# ======================================================================


def _build_ticket_text(ticket: Any) -> str:
    """
    Build the initial ticket text supplied to LangGraph.

    Title is included when available so the graph has the same
    contextual information used by the application layer.
    """

    title = (
        getattr(
            ticket,
            "title",
            "",
        )
        or ""
    )

    description = (
        getattr(
            ticket,
            "description",
            "",
        )
        or ""
    )

    if title and description:
        return f"{title}\n\n{description}"

    return title or description


def _format_solution(
    diagnosis: str,
    root_cause: str,
    resolution_steps: list[str],
    evidence: list[str],
) -> str:
    """
    Convert structured graph output into the existing human-readable
    solution representation.
    """

    steps = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(
            resolution_steps,
            start=1,
        )
    )

    if not steps:
        steps = "No resolution steps generated."

    evidence_text = (
        "\n".join(f"- {item}" for item in evidence)
        if evidence
        else "No specific evidence cited."
    )

    return (
        f"Diagnosis:\n"
        f"{diagnosis or 'No diagnosis generated.'}\n\n"
        f"Root Cause:\n"
        f"{root_cause or 'No root cause established.'}\n\n"
        f"Resolution Steps:\n"
        f"{steps}\n\n"
        f"Evidence:\n"
        f"{evidence_text}"
    )


def _compute_similarity_score(
    context_docs: list[dict],
) -> float:
    """
    Derive the retrieval similarity signal from the actual
    retrieved document scores.
    """

    if not context_docs:
        return 0.0

    scores = [
        float(document.get("score", 0.0))
        for document in context_docs
        if isinstance(document, dict)
    ]

    if not scores:
        return 0.0

    return max(scores)


def _build_explanation(
    ticket_text: str,
    category: str,
    context_docs: list[dict],
    solution: str,
    confidence: float,
) -> str:
    """
    Preserve the existing explainability interface.

    Explainability failures must not prevent the graph result
    from reaching the API.
    """

    try:
        return explain(
            ticket_text=ticket_text,
            category=category,
            context_docs=context_docs,
            solution=solution,
            confidence=confidence,
        )

    except Exception as exc:
        logger.exception(
            "[GraphExecutor] Explainability failed: " f"{type(exc).__name__}: {exc}"
        )

        return (
            "Explanation could not be generated. "
            "The resolution was produced by the ResolveX "
            "agent graph."
        )
