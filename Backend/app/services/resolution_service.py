"""
resolution_service.py — Orchestrates AI ticket resolution.

Phase 14.9:
    Production resolution execution is handled by the
    ResolveX LangGraph instead of the legacy monolithic pipeline.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.repositories.ticket_repo import TicketRepository
from app.schemas.resolution_schema import ResolutionResult
from app.core.exceptions import TicketNotFoundException
from app.core.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    STATUS_AUTO_RESOLVED,
    STATUS_ESCALATED,
)
from app.core.logger import logger
from app.core.expert_resolvers import get_best_expert_resolver

from ai.graph.executor import execute_resolvex_graph


class ResolutionService:

    def __init__(self, db: Session):
        self.repo = TicketRepository(db)

    async def resolve(
        self,
        ticket_id: int,
        force: bool = False,
    ) -> ResolutionResult:
        """
        Execute ResolveX LangGraph for a ticket.

        Decision:

            graph decision == auto_resolve
                → STATUS_AUTO_RESOLVED

            graph decision == human_review
                → STATUS_ESCALATED

            graph decision == escalate
                → STATUS_ESCALATED

        The existing API response contract is preserved.
        """

        # ==============================================================
        # Load ticket
        # ==============================================================

        ticket = self.repo.get_by_id(ticket_id)

        if not ticket:
            raise TicketNotFoundException(ticket_id)

        # ==============================================================
        # Existing-resolution protection
        # ==============================================================

        if (
            ticket.status == STATUS_AUTO_RESOLVED
            and not force
        ):
            logger.info(
                f"Ticket {ticket_id} already resolved. "
                f"Skipping graph execution."
            )

            return self._build_result(ticket)

        logger.info(
            f"Starting ResolveX LangGraph "
            f"for ticket_id={ticket_id}"
        )

        # ==============================================================
        # Phase 14.9 — LangGraph execution
        # ==============================================================

        graph_output = execute_resolvex_graph(
            ticket=ticket,
        )

        # ==============================================================
        # Extract graph result
        # ==============================================================

        solution = graph_output.get(
            "solution",
            "",
        )

        confidence = float(
            graph_output.get(
                "confidence",
                0.0,
            )
        )

        category = graph_output.get(
            "category",
            "other",
        )

        explanation = graph_output.get(
            "explanation",
            "",
        )

        decision = graph_output.get(
            "decision",
            "escalate",
        )

        requires_human = bool(
            graph_output.get(
                "requires_human",
                False,
            )
        )

        fallback_used = bool(
            graph_output.get(
                "fallback_used",
                False,
            )
        )

        errors = graph_output.get(
            "errors",
            [],
        )

        # ==============================================================
        # Final application-level decision
        # ==============================================================

        auto_resolved = (
            decision == "auto_resolve"
            and not requires_human
            and not fallback_used
            and not errors
        )

        escalated = not auto_resolved

        if auto_resolved:
            new_status = STATUS_AUTO_RESOLVED
        else:
            new_status = STATUS_ESCALATED

        # ==============================================================
        # Expert Resolver Assignment
        # ==============================================================

        if new_status == STATUS_ESCALATED:

            ticket_text = (
                f"{ticket.title or ''} "
                f"{ticket.description or ''}"
            )

            resolver = get_best_expert_resolver(
                category,
                ticket_text,
            )

            ticket.assigned_resolver_id = resolver["id"]
            ticket.assigned_resolver_name = resolver["name"]
            ticket.assigned_resolver_category = resolver["category"]
            ticket.assigned_at = datetime.now(
                timezone.utc
            )

            logger.info(
                f"Assigned expert resolver "
                f"{resolver['name']} "
                f"to ticket {ticket_id}"
            )

        # ==============================================================
        # Persist result
        # ==============================================================

        ticket.solution = solution
        ticket.confidence = confidence
        ticket.category = category
        ticket.explanation = explanation
        ticket.status = new_status

        self.repo.update(ticket)

        logger.info(
            f"Ticket {ticket_id} → "
            f"status={new_status}, "
            f"decision={decision}, "
            f"confidence={confidence:.3f}"
        )

        # ==============================================================
        # API-compatible response
        # ==============================================================

        return ResolutionResult(
            ticket_id=ticket_id,
            category=category,
            solution=solution,
            confidence=confidence,
            auto_resolved=auto_resolved,
            escalated_to_human=escalated,
            explanation=explanation,
            assigned_resolver_id=(
    str(ticket.assigned_resolver_id)
    if ticket.assigned_resolver_id is not None
    else None
),
            assigned_resolver_name=(
                ticket.assigned_resolver_name
            ),
            assigned_resolver_category=(
                ticket.assigned_resolver_category
            ),
        )

    # ==================================================================
    # GET /resolve/{ticket_id}
    # ==================================================================

    def get_resolution(
        self,
        ticket_id: int,
    ) -> ResolutionResult:
        """
        Return the stored resolution result for a ticket.
        """

        ticket = self.repo.get_by_id(ticket_id)

        if not ticket:
            raise TicketNotFoundException(ticket_id)

        return self._build_result(ticket)

    # ==================================================================
    # Stored result → API schema
    # ==================================================================

    def _build_result(
        self,
        ticket,
    ) -> ResolutionResult:
        """
        Convert persisted ticket information into the existing
        ResolutionResult schema.
        """

        confidence = ticket.confidence or 0.0

        return ResolutionResult(
            ticket_id=ticket.id,
            category=ticket.category,
            solution=ticket.solution,
            confidence=confidence,
            auto_resolved=(
                ticket.status == STATUS_AUTO_RESOLVED
            ),
            escalated_to_human=(
                ticket.status == STATUS_ESCALATED
            ),
            explanation=ticket.explanation,
            assigned_resolver_id=(
            str(ticket.assigned_resolver_id)
            if ticket.assigned_resolver_id is not None
            else None
        ),
            assigned_resolver_name=(
                ticket.assigned_resolver_name
            ),
            assigned_resolver_category=(
                ticket.assigned_resolver_category
            ),
        )