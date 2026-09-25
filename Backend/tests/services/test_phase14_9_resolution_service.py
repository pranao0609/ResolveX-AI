from types import SimpleNamespace
from unittest.mock import patch


def _make_ticket(status="open"):
    return SimpleNamespace(
        id=1,
        title="VPN issue",
        description="VPN disconnects after login.",
        attachment_paths=None,
        status=status,
        solution=None,
        confidence=None,
        category=None,
        explanation=None,
        assigned_resolver_id=None,
        assigned_resolver_name=None,
        assigned_resolver_category=None,
        assigned_at=None,
    )


def test_resolution_service_uses_graph():
    from app.services.resolution_service import ResolutionService

    ticket = _make_ticket()

    db = object()

    graph_output = {
        "solution": "Restart the VPN client.",
        "confidence": 0.91,
        "category": "network",
        "explanation": "Supported by VPN documentation.",
        "decision": "auto_resolve",
        "requires_human": False,
        "fallback_used": False,
        "errors": [],
    }

    with patch(
        "app.services.resolution_service.TicketRepository"
    ) as repo_cls:

        repo = repo_cls.return_value
        repo.get_by_id.return_value = ticket

        with patch(
            "app.services.resolution_service.execute_resolvex_graph",
            return_value=graph_output,
        ) as execute_graph:

            service = ResolutionService(db)

            result = __import__(
                "asyncio"
            ).run(
                service.resolve(
                    ticket_id=1
                )
            )

    execute_graph.assert_called_once_with(
        ticket=ticket
    )

    assert result.ticket_id == 1
    assert result.category == "network"
    assert result.confidence == 0.91
    assert result.auto_resolved is True
    assert result.escalated_to_human is False

    assert ticket.solution == (
        "Restart the VPN client."
    )
    assert ticket.status is not None


def test_resolution_service_escalates_graph_human_review():
    from app.services.resolution_service import ResolutionService

    ticket = _make_ticket()

    graph_output = {
        "solution": "Review the VPN configuration.",
        "confidence": 0.60,
        "category": "network",
        "explanation": "Human review required.",
        "decision": "human_review",
        "requires_human": True,
        "fallback_used": False,
        "errors": [],
    }

    with patch(
        "app.services.resolution_service.TicketRepository"
    ) as repo_cls:

        repo = repo_cls.return_value
        repo.get_by_id.return_value = ticket

        with patch(
            "app.services.resolution_service.execute_resolvex_graph",
            return_value=graph_output,
        ):

            with patch(
                "app.services.resolution_service.get_best_expert_resolver"
            ) as resolver:

                resolver.return_value = {
                    "id": 5,
                    "name": "Network Expert",
                    "category": "network",
                }

                service = ResolutionService(
                    object()
                )

                result = __import__(
                    "asyncio"
                ).run(
                    service.resolve(
                        ticket_id=1
                    )
                )

    assert result.auto_resolved is False
    assert result.escalated_to_human is True

    assert ticket.assigned_resolver_id == 5
    assert ticket.assigned_resolver_name == (
        "Network Expert"
    )