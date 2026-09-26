from __future__ import annotations

from typing import Any

from ai.agents.classification_agent import run_classification_agent
from ai.graph.nodes.initialization import _append_error, _stage_metadata
from ai.graph.state import ResolveXState
from ai.preprocessing.file_parser import parse_attachments
from ai.preprocessing.text_cleaner import clean_text
from app.core.logger import logger


def ticket_analyzer(
    state: ResolveXState,
) -> dict[str, Any]:
    """
    Analyze the incoming support ticket.

    Responsibilities:
        1. Clean ticket text.
        2. Extract attachment text.
        3. Execute the Classification Agent.
        4. Write validated results into graph state.
    """

    ticket_text = state.get("ticket_text", "") or ""

    attachment_paths = state.get("attachment_paths")

    try:
        # -----------------------------------------------------------
        # Step 1: Text preprocessing
        # -----------------------------------------------------------

        cleaned_text = clean_text(ticket_text)

        # -----------------------------------------------------------
        # Step 2: Attachment extraction
        # -----------------------------------------------------------

        if attachment_paths:

            if isinstance(
                attachment_paths,
                str,
            ):
                paths = [
                    path.strip() for path in attachment_paths.split(",") if path.strip()
                ]

            else:
                paths = [
                    str(path).strip() for path in attachment_paths if str(path).strip()
                ]

            if paths:
                extracted = parse_attachments(paths)

                if extracted:
                    cleaned_text = (
                        f"{cleaned_text}\n\n" f"[Attachments]\n" f"{extracted}"
                    )

        # -----------------------------------------------------------
        # Step 3: Classification Agent
        # -----------------------------------------------------------

        classification = run_classification_agent(
            cleaned_text,
            ticket_id=state.get("ticket_id"),
        )

        category = classification["category"]
        classification_score = classification["confidence"]
        classification_confidence_level = classification["confidence_level"]
        classification_requires_review = classification["requires_review"]
        classification_requires_reclassification = classification[
            "requires_reclassification"
        ]
        classification_fallback = classification["fallback_used"]
        classification_error = classification.get("error")
        reclassified = classification.get("reclassified", False)
        reclassification_used = classification.get("reclassification_used", False)
        reclassification_error = classification.get("reclassification_error")

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=ticket_analyzer "
            f"category={category} "
            f"score={classification_score:.3f} "
            f"confidence_level={classification_confidence_level} "
            f"requires_review={classification_requires_review} "
            f"requires_reclassification="
            f"{classification_requires_reclassification} "
            f"fallback={classification_fallback} "
            f"text_length={len(cleaned_text)} "
            f"request_id={state.get('request_id')} "
            f"graph_run_id={state.get('graph_run_id')}"
        )

        # -----------------------------------------------------------
        # Step 4: Stage metadata
        # -----------------------------------------------------------

        metadata = _stage_metadata(
            state,
            "ticket_analyzer",
        )

        result: dict[str, Any] = {
            "cleaned_ticket": cleaned_text,
            "category": category,
            "category_confidence": float(classification_score),
            "classification_confidence_level": (classification_confidence_level),
            "classification_requires_review": (classification_requires_review),
            "classification_requires_reclassification": (
                classification_requires_reclassification
            ),
            "reclassified": reclassified,
            "reclassification_used": reclassification_used,
            "reclassification_error": reclassification_error,
            "metadata": metadata,
        }

        # -----------------------------------------------------------
        # Step 5: Propagate classification fallback/error
        # -----------------------------------------------------------

        if classification_fallback:
            result["fallback_used"] = True

        if classification_error:
            result["errors"] = _append_error(
                state,
                "classification_agent",
                classification_error,
            )

        return result

    except Exception as exc:

        logger.exception(
            f"Ticket analyzer failed for " f"ticket_id={state.get('ticket_id')}: {exc}"
        )

        metadata = _stage_metadata(
            state,
            "ticket_analyzer",
        )

        errors = _append_error(
            state,
            "ticket_analyzer",
            exc,
        )

        return {
            "cleaned_ticket": (cleaned_text if "cleaned_text" in locals() else ""),
            "category": "software",
            "category_confidence": 0.0,
            "errors": errors,
            "fallback_used": True,
            "metadata": metadata,
        }


__all__ = [
    "ticket_analyzer",
    "run_classification_agent",
]
