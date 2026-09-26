"""
resolution_schema.py — Pydantic schemas for AI resolution responses.
"""

from pydantic import BaseModel
from typing import Optional


class ResolutionRequest(BaseModel):
    """Trigger AI pipeline resolution for a ticket."""

    ticket_id: int
    force: bool = False  # force re-run even if already resolved


class ResolutionResult(BaseModel):
    """Result of an AI resolution attempt."""

    ticket_id: int
    category: Optional[str] = None
    solution: Optional[str] = None
    confidence: float
    auto_resolved: bool
    escalated_to_human: bool
    explanation: Optional[str] = None
    assigned_resolver_id: Optional[str] = None
    assigned_resolver_name: Optional[str] = None
    assigned_resolver_category: Optional[str] = None


class HITLReviewRequest(BaseModel):
    """Payload when a human agent reviews an escalated ticket."""

    ticket_id: int
    agent_id: int
    agent_solution: str
    close_ticket: bool = True
