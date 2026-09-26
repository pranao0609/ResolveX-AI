"""
ticket_model.py — SQLAlchemy ORM model for support tickets.
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Enum
from sqlalchemy.sql import func
from app.database import Base
import enum


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    AUTO_RESOLVED = "auto_resolved"
    ESCALATED = "escalated"  # sent to human-in-the-loop
    CLOSED = "closed"


class TicketPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Ticket(Base):
    """Represents a single support ticket in the system."""

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Core fields
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)  # classified category
    priority = Column(String(50), default=TicketPriority.MEDIUM)
    status = Column(String(50), default=TicketStatus.OPEN)

    # AI-generated fields
    solution = Column(Text, nullable=True)  # LLM-generated resolution
    confidence = Column(Float, nullable=True)  # AI confidence score 0–1
    explanation = Column(Text, nullable=True)  # explainability output
    embedding_id = Column(String(255), nullable=True)  # reference in FAISS index

    # Metadata
    submitted_by = Column(String(255), nullable=True)  # user email / id
    assigned_to = Column(String(255), nullable=True)  # human agent (HITL)

    # Expert Resolver Assignment fields
    assigned_resolver_id = Column(String(50), nullable=True)
    assigned_resolver_name = Column(String(100), nullable=True)
    assigned_resolver_category = Column(String(100), nullable=True)
    assigned_at = Column(DateTime(timezone=True), nullable=True)

    attachment_paths = Column(Text, nullable=True)  # comma-separated file paths

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<Ticket id={self.id} title='{self.title}' status={self.status}>"
