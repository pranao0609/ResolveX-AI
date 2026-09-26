"""
analytics_service.py — Aggregated analytics and reporting logic.
"""

from sqlalchemy.orm import Session

from app.repositories.ticket_repo import TicketRepository
from app.schemas.analytics_schema import (
    AnalyticsSummary,
    TicketStats,
    ConfidenceStats,
    CategoryBreakdown,
)
from app.core.constants import CONFIDENCE_HIGH, CONFIDENCE_LOW
from app.core.logger import logger


class AnalyticsService:
    def __init__(self, db: Session):
        self.repo = TicketRepository(db)

    def get_summary(self) -> AnalyticsSummary:
        """Combine all analytics into a single response."""
        return AnalyticsSummary(
            ticket_stats=self.get_ticket_stats(),
            confidence_stats=self.get_confidence_stats(),
            category_breakdown=self.get_category_breakdown(),
        )

    def get_ticket_stats(self) -> TicketStats:
        """Return ticket counts grouped by status."""
        counts = self.repo.get_status_counts()
        return TicketStats(
            total_tickets=sum(counts.values()),
            open_tickets=counts.get("open", 0),
            auto_resolved=counts.get("auto_resolved", 0),
            escalated=counts.get("escalated", 0),
            closed=counts.get("closed", 0),
        )

    def get_confidence_stats(self) -> ConfidenceStats:
        """Return average confidence and high/low bucket counts."""
        avg = self.repo.get_average_confidence()

        # Count tickets per bucket using list_all (placeholder — extend with SQL aggregation)
        all_tickets = self.repo.list_all(limit=10_000)
        high = sum(
            1 for t in all_tickets if t.confidence and t.confidence >= CONFIDENCE_HIGH
        )
        low = sum(
            1 for t in all_tickets if t.confidence and t.confidence < CONFIDENCE_LOW
        )

        return ConfidenceStats(
            avg_confidence=avg,
            high_confidence_count=high,
            low_confidence_count=low,
        )

    def get_category_breakdown(self) -> CategoryBreakdown:
        """Return ticket count per category."""
        counts = self.repo.get_category_counts()
        return CategoryBreakdown(category_counts=counts)
