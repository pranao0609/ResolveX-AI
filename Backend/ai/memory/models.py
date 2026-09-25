"""
Memory data models for ResolveX.

These models define the contract between agents and the memory layer.
They intentionally do not depend on SQLAlchemy ORM models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class MemoryRecord:
    """
    Normalized representation of a historical ticket used by agents.

    This deliberately contains only information useful for reasoning.
    Database-specific ORM objects must not leak into the agent layer.
    """

    ticket_id: int
    title: str
    description: str

    category: str | None = None
    status: str | None = None

    solution: str | None = None
    confidence: float | None = None
    explanation: str | None = None

    submitted_by: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """Return whether the ticket represents a completed resolution."""

        return self.status in {
            "auto_resolved",
            "closed",
        }

    @property
    def has_solution(self) -> bool:
        """Return whether a usable historical solution exists."""

        return bool(self.solution and self.solution.strip())

    def to_dict(self) -> dict[str, Any]:
        """Convert the memory record into an agent-safe dictionary."""

        return {
            "ticket_id": self.ticket_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "status": self.status,
            "solution": self.solution,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "submitted_by": self.submitted_by,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at is not None
                else None
            ),
            "updated_at": (
                self.updated_at.isoformat()
                if self.updated_at is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class MemorySearchQuery:
    """
    Query contract for historical memory retrieval.

    The initial implementation uses structured filters.
    Semantic similarity can be added later without changing the
    agent-facing memory interface.
    """

    query: str | None = None

    category: str | None = None
    submitted_by: str | None = None

    exclude_ticket_id: int | None = None

    resolved_only: bool = True

    limit: int = 5

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("Memory search limit must be at least 1.")

        if self.limit > 50:
            raise ValueError("Memory search limit cannot exceed 50.")

        if self.query is not None:
            self.query = self.query.strip() or None

        if self.category is not None:
            self.category = self.category.strip().lower() or None

        if self.submitted_by is not None:
            self.submitted_by = self.submitted_by.strip() or None


@dataclass(slots=True)
class MemorySearchResult:
    """Result returned by the memory layer."""

    records: list[MemoryRecord] = field(default_factory=list)

    total: int = 0

    query: MemorySearchQuery | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of memory records returned."""

        return len(self.records)

    def to_dict(self) -> dict[str, Any]:
        """Convert the search result to an agent-safe dictionary."""

        return {
            "records": [record.to_dict() for record in self.records],
            "total": self.total,
            "count": self.count,
            "query": (
                {
                    "query": self.query.query,
                    "category": self.query.category,
                    "submitted_by": self.query.submitted_by,
                    "exclude_ticket_id": self.query.exclude_ticket_id,
                    "resolved_only": self.query.resolved_only,
                    "limit": self.query.limit,
                }
                if self.query is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }