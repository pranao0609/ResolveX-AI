"""
hitl_service.py — Human-in-the-Loop review logic.
Handles assignment of escalated tickets to human agents and recording their solutions.
"""

from sqlalchemy.orm import Session

from app.repositories.ticket_repo import TicketRepository
from app.repositories.user_repo import UserRepository
from app.schemas.resolution_schema import HITLReviewRequest
from app.core.exceptions import TicketNotFoundException, ResolveXException
from app.core.constants import STATUS_CLOSED, STATUS_ESCALATED
from app.core.logger import logger


class HITLService:
    def __init__(self, db: Session):
        self.ticket_repo = TicketRepository(db)
        self.user_repo = UserRepository(db)

    def submit_review(self, payload: HITLReviewRequest) -> None:
        """
        Record a human agent's resolution for an escalated ticket.
        1. Validate ticket exists and is in ESCALATED state.
        2. Record agent solution.
        3. Optionally close the ticket.
        """
        ticket = self.ticket_repo.get_by_id(payload.ticket_id)
        if not ticket:
            raise TicketNotFoundException(payload.ticket_id)

        if ticket.status != STATUS_ESCALATED:
            raise ResolveXException(
                status_code=400,
                detail=f"Ticket {payload.ticket_id} is not in escalated state (current: {ticket.status})",
            )

        # Verify agent exists
        agent = self.user_repo.get_by_id(payload.agent_id)
        if not agent:
            raise ResolveXException(
                status_code=404, detail=f"Agent {payload.agent_id} not found"
            )

        # Update ticket with human solution
        ticket.solution = payload.agent_solution
        ticket.assigned_to = agent.email
        if payload.close_ticket:
            ticket.status = STATUS_CLOSED

        self.ticket_repo.update(ticket)
        logger.info(
            f"HITL review: ticket {payload.ticket_id} reviewed by agent {agent.email}, "
            f"closed={payload.close_ticket}"
        )

    def get_escalated_tickets(self):
        """Return all tickets currently awaiting human review."""
        return self.ticket_repo.list_all(status=STATUS_ESCALATED, limit=100)
