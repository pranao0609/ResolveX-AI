"""
analytics_schema.py — Pydantic schemas for analytics/stats responses.
"""

from pydantic import BaseModel
from typing import Dict


class TicketStats(BaseModel):
    """High-level ticket volume statistics."""

    total_tickets: int
    open_tickets: int
    auto_resolved: int
    escalated: int
    closed: int


class ConfidenceStats(BaseModel):
    """Confidence distribution across resolved tickets."""

    avg_confidence: float
    high_confidence_count: int  # >= 0.75
    low_confidence_count: int  # < 0.50


class CategoryBreakdown(BaseModel):
    """Ticket count per category."""

    category_counts: Dict[str, int]


class AnalyticsSummary(BaseModel):
    """Combined analytics summary response."""

    ticket_stats: TicketStats
    confidence_stats: ConfidenceStats
    category_breakdown: CategoryBreakdown
