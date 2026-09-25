"""
PostgreSQL-backed historical ticket memory.

This implementation reuses the existing TicketRepository instead of
introducing a separate memory database or ORM model.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ai.memory.base import MemoryStore
from ai.memory.models import (
    MemoryRecord,
    MemorySearchQuery,
    MemorySearchResult,
)
from app.models.ticket_model import Ticket
from app.repositories.ticket_repo import TicketRepository


class TicketMemoryStore(MemoryStore):
    """
    Historical memory backed by the existing tickets table.
    """

    RESOLVED_STATUSES = (
        "auto_resolved",
        "closed",
    )

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = TicketRepository(db)

    @classmethod
    def _to_memory_record(
        cls,
        ticket: Ticket,
    ) -> MemoryRecord:
        """Convert a SQLAlchemy Ticket into an agent-safe memory record."""

        return MemoryRecord(
            ticket_id=ticket.id,
            title=ticket.title,
            description=ticket.description,
            category=ticket.category,
            status=ticket.status,
            solution=ticket.solution,
            confidence=ticket.confidence,
            explanation=ticket.explanation,
            submitted_by=ticket.submitted_by,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
        )

    def get(
        self,
        ticket_id: int,
    ) -> MemoryRecord | None:
        """Retrieve one historical ticket."""

        ticket = self.repo.get_by_id(ticket_id)

        if ticket is None:
            return None

        return self._to_memory_record(ticket)

    def search(
        self,
        query: MemorySearchQuery,
    ) -> MemorySearchResult:
        """
        Search historical tickets using structured filters.

        This phase intentionally uses PostgreSQL filtering and lexical
        matching. Semantic memory retrieval can be added later without
        changing the MemoryStore interface.
        """

        db_query = self.db.query(Ticket)

        if query.resolved_only:
            db_query = db_query.filter(
                Ticket.status.in_(self.RESOLVED_STATUSES)
            )

        if query.category:
            db_query = db_query.filter(
                Ticket.category == query.category
            )

        if query.submitted_by:
            db_query = db_query.filter(
                Ticket.submitted_by == query.submitted_by
            )

        if query.exclude_ticket_id is not None:
            db_query = db_query.filter(
                Ticket.id != query.exclude_ticket_id
            )

        if query.query:
            search_term = f"%{query.query}%"

            db_query = db_query.filter(
                Ticket.title.ilike(search_term)
                | Ticket.description.ilike(search_term)
                | Ticket.solution.ilike(search_term)
            )

        db_query = db_query.order_by(
            Ticket.created_at.desc()
        )

        tickets = db_query.limit(query.limit).all()

        records = [
            self._to_memory_record(ticket)
            for ticket in tickets
        ]

        return MemorySearchResult(
            records=records,
            total=len(records),
            query=query,
            metadata={
                "backend": "postgresql",
                "source": "tickets",
                "semantic_search": False,
                "resolved_statuses": list(self.RESOLVED_STATUSES),
            },
        )