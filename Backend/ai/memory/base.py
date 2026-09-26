"""
Abstract memory interface for ResolveX.

Agents depend on this interface rather than directly depending on
PostgreSQL, SQLAlchemy, or repository implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai.memory.models import (
    MemoryRecord,
    MemorySearchQuery,
    MemorySearchResult,
)


class MemoryStore(ABC):
    """
    Abstract interface for ResolveX memory.

    Implementations may use PostgreSQL, Redis, vector stores, or
    another backend in the future without changing agent code.
    """

    @abstractmethod
    def search(
        self,
        query: MemorySearchQuery,
    ) -> MemorySearchResult:
        """
        Search historical memory.

        Args:
            query: Structured memory search query.

        Returns:
            Normalized memory search result.
        """

        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        ticket_id: int,
    ) -> MemoryRecord | None:
        """
        Retrieve one historical memory record by ticket ID.
        """

        raise NotImplementedError
