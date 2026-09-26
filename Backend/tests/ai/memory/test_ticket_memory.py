from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from ai.memory.models import MemorySearchQuery
from ai.memory.ticket_memory import TicketMemoryStore


def _ticket(
    ticket_id: int = 1,
    *,
    title: str = "VPN connection failed",
    description: str = "VPN does not connect.",
    category: str = "network",
    status: str = "closed",
    solution: str = "Reset VPN credentials.",
    confidence: float = 0.95,
    submitted_by: str = "user@example.com",
):
    ticket = MagicMock()

    ticket.id = ticket_id
    ticket.title = title
    ticket.description = description
    ticket.category = category
    ticket.status = status
    ticket.solution = solution
    ticket.confidence = confidence
    ticket.explanation = "Historical successful resolution."
    ticket.submitted_by = submitted_by
    ticket.created_at = datetime(2026, 9, 25, 10, 0, 0)
    ticket.updated_at = datetime(2026, 9, 25, 10, 30, 0)

    return ticket


def test_get_returns_memory_record() -> None:
    db = MagicMock()

    store = TicketMemoryStore(db)

    ticket = _ticket(ticket_id=42)

    store.repo.get_by_id = MagicMock(return_value=ticket)

    result = store.get(42)

    assert result is not None
    assert result.ticket_id == 42
    assert result.title == "VPN connection failed"
    assert result.solution == "Reset VPN credentials."
    assert result.confidence == 0.95


def test_get_returns_none_when_ticket_missing() -> None:
    db = MagicMock()

    store = TicketMemoryStore(db)

    store.repo.get_by_id = MagicMock(return_value=None)

    result = store.get(999)

    assert result is None


def test_search_returns_normalized_memory_records() -> None:
    db = MagicMock()

    store = TicketMemoryStore(db)

    ticket = _ticket(ticket_id=10)

    query = MemorySearchQuery(
        query="VPN",
        category="network",
        limit=5,
    )

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.all.return_value = [ticket]

    db.query.return_value = mock_query

    result = store.search(query)

    assert result.count == 1
    assert result.records[0].ticket_id == 10
    assert result.records[0].category == "network"
    assert result.records[0].solution == "Reset VPN credentials."

    assert result.metadata["backend"] == "postgresql"
    assert result.metadata["source"] == "tickets"
    assert result.metadata["semantic_search"] is False


def test_search_uses_resolved_only_by_default() -> None:
    db = MagicMock()

    store = TicketMemoryStore(db)

    query = MemorySearchQuery(
        query="VPN",
        resolved_only=True,
    )

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.all.return_value = []

    db.query.return_value = mock_query

    store.search(query)

    assert mock_query.filter.called
    assert mock_query.order_by.called
    assert mock_query.limit.called
    assert mock_query.all.called
