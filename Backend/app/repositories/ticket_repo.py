"""
ticket_repo.py — Repository layer for Ticket database operations.
All raw SQLAlchemy queries go here; services call these methods.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List

from app.models.ticket_model import Ticket


class TicketRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, ticket: Ticket) -> Ticket:
        """Persist a new ticket to the database."""
        self.db.add(ticket)
        self.db.commit()
        self.db.refresh(ticket)
        return ticket

    def get_by_id(self, ticket_id: int) -> Optional[Ticket]:
        """Return a single ticket by primary key, or None."""
        return self.db.query(Ticket).filter(Ticket.id == ticket_id).first()

    def list_all(
        self,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Ticket]:
        """Return a paginated list of tickets, with optional filters."""
        query = self.db.query(Ticket)
        if status:
            query = query.filter(Ticket.status == status)
        if category:
            query = query.filter(Ticket.category == category)
        return query.offset(skip).limit(limit).all()

    def count(
        self, status: Optional[str] = None, category: Optional[str] = None
    ) -> int:
        """Return total count for pagination."""
        query = self.db.query(func.count(Ticket.id))
        if status:
            query = query.filter(Ticket.status == status)
        if category:
            query = query.filter(Ticket.category == category)
        return query.scalar()

    def update(self, ticket: Ticket) -> Ticket:
        """Commit changes to an already-fetched ticket object."""
        self.db.commit()
        self.db.refresh(ticket)
        return ticket

    def delete(self, ticket: Ticket) -> None:
        """Hard delete a ticket record."""
        self.db.delete(ticket)
        self.db.commit()

    def get_status_counts(self) -> dict:
        """Return a dict of {status: count} for all tickets."""
        rows = (
            self.db.query(Ticket.status, func.count(Ticket.id))
            .group_by(Ticket.status)
            .all()
        )
        return {status: count for status, count in rows}

    def get_category_counts(self) -> dict:
        """Return a dict of {category: count}."""
        rows = (
            self.db.query(Ticket.category, func.count(Ticket.id))
            .group_by(Ticket.category)
            .all()
        )
        return {cat or "uncategorised": count for cat, count in rows}

    def get_average_confidence(self) -> float:
        """Return average confidence score across all resolved tickets."""
        result = self.db.query(func.avg(Ticket.confidence)).scalar()
        return round(result or 0.0, 4)
