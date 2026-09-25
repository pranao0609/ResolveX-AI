"""
ResolveX memory tools.

These functions are thin tool handlers. They do not perform tool
telemetry themselves; execute_tool() remains responsible for the
canonical tool-call record.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ai.memory.models import MemorySearchQuery
from ai.memory.ticket_memory import TicketMemoryStore


def search_memory(
    db: Session,
    query: str | None = None,
    category: str | None = None,
    submitted_by: str | None = None,
    exclude_ticket_id: int | None = None,
    resolved_only: bool = True,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Search historical ticket memory.

    Returns normalized dictionaries suitable for agent consumption.
    """

    memory = TicketMemoryStore(db)

    search_query = MemorySearchQuery(
        query=query,
        category=category,
        submitted_by=submitted_by,
        exclude_ticket_id=exclude_ticket_id,
        resolved_only=resolved_only,
        limit=limit,
    )

    result = memory.search(search_query)

    return [
        record.to_dict()
        for record in result.records
    ]