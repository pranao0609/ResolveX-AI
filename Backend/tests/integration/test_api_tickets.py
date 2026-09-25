"""
test_api_tickets.py — Integration tests for the ticket CRUD API endpoints.

Uses the shared `client` fixture (TestClient + in-memory SQLite).
The AI resolution pipeline is mocked so tests don't need Groq or FAISS.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.schemas.resolution_schema import ResolutionResult


# ── Shared mock resolution result ────────────────────────────────────────────

def _fake_resolution(ticket_id: int = 1, confidence: float = 0.82) -> ResolutionResult:
    return ResolutionResult(
        ticket_id=ticket_id,
        category="network",
        solution="Restart VPN client and check firewall rules.",
        confidence=confidence,
        explanation="VPN keywords matched with high confidence.",
        auto_resolved=confidence >= 0.75,
        escalated_to_human=confidence < 0.75,
        assigned_resolver_id=None,
        assigned_resolver_name=None,
        assigned_resolver_category=None,
    )


# ── List tickets ──────────────────────────────────────────────────────────────

class TestListTickets:
    """GET /api/v1/tickets"""

    def test_empty_ticket_list_returns_200(self, client):
        response = client.get("/api/v1/tickets")
        assert response.status_code == 200

    def test_empty_ticket_list_returns_list_structure(self, client):
        data = client.get("/api/v1/tickets").json()
        assert "tickets" in data or isinstance(data, list) or "items" in data or "data" in data

    def test_pagination_params_accepted(self, client):
        response = client.get("/api/v1/tickets?page=1&page_size=10")
        assert response.status_code == 200

    def test_invalid_page_zero_rejected(self, client):
        response = client.get("/api/v1/tickets?page=0")
        assert response.status_code == 422

    def test_page_size_too_large_rejected(self, client):
        response = client.get("/api/v1/tickets?page_size=999")
        assert response.status_code == 422

    def test_status_filter_accepted(self, client):
        response = client.get("/api/v1/tickets?status=open")
        assert response.status_code == 200

    def test_category_filter_accepted(self, client):
        response = client.get("/api/v1/tickets?category=network")
        assert response.status_code == 200


# ── Get single ticket ─────────────────────────────────────────────────────────

class TestGetTicket:
    """GET /api/v1/tickets/{id}"""

    def test_nonexistent_ticket_returns_404(self, client):
        response = client.get("/api/v1/tickets/99999")
        assert response.status_code == 404

    def test_404_response_contains_detail(self, client):
        response = client.get("/api/v1/tickets/99999")
        assert "detail" in response.json()


# ── Create ticket ─────────────────────────────────────────────────────────────

class TestCreateTicket:
    """POST /api/v1/tickets"""

    def _post_ticket(self, client, payload: dict):
        """Helper: POST multipart form-data."""
        return client.post("/api/v1/tickets", data=payload)

    def test_create_ticket_succeeds_with_valid_payload(self, client, sample_ticket_payload):
        with patch(
            "app.routes.ticket_routes.ResolutionService.resolve",
            new_callable=AsyncMock,
            return_value=_fake_resolution(1),
        ):
            response = self._post_ticket(client, sample_ticket_payload)

        assert response.status_code == 201

    def test_create_ticket_returns_ticket_id(self, client, sample_ticket_payload):
        with patch(
            "app.routes.ticket_routes.ResolutionService.resolve",
            new_callable=AsyncMock,
            return_value=_fake_resolution(1),
        ):
            data = self._post_ticket(client, sample_ticket_payload).json()

        assert "id" in data
        assert isinstance(data["id"], int)

    def test_create_ticket_returns_correct_category(self, client, sample_ticket_payload):
        with patch(
            "app.routes.ticket_routes.ResolutionService.resolve",
            new_callable=AsyncMock,
            return_value=_fake_resolution(1),
        ):
            data = self._post_ticket(client, sample_ticket_payload).json()

        assert data.get("category") in (
            "network", "software", "hardware", "access_permission", "security", "other"
        )

    def test_create_ticket_includes_confidence(self, client, sample_ticket_payload):
        with patch(
            "app.routes.ticket_routes.ResolutionService.resolve",
            new_callable=AsyncMock,
            return_value=_fake_resolution(1, confidence=0.82),
        ):
            data = self._post_ticket(client, sample_ticket_payload).json()

        assert "confidence" in data
        # Confidence is stored as a percentage int on the TicketResponse
        assert isinstance(data["confidence"], (int, float))

    def test_missing_title_returns_422(self, client):
        bad_payload = {
            "description": "some description",
            "category": "network",
        }
        response = self._post_ticket(client, bad_payload)
        assert response.status_code == 422

    def test_missing_description_returns_422(self, client):
        bad_payload = {
            "title": "Some title",
            "category": "network",
        }
        response = self._post_ticket(client, bad_payload)
        assert response.status_code == 422

    def test_missing_category_returns_422(self, client):
        bad_payload = {
            "title": "Some title",
            "description": "some description",
        }
        response = self._post_ticket(client, bad_payload)
        assert response.status_code == 422

    def test_create_returns_solution_field(self, client, sample_ticket_payload):
        with patch(
            "app.routes.ticket_routes.ResolutionService.resolve",
            new_callable=AsyncMock,
            return_value=_fake_resolution(1),
        ):
            data = self._post_ticket(client, sample_ticket_payload).json()

        assert "solution" in data

    def test_create_returns_decision_via_status(self, client, sample_ticket_payload):
        """
        NOTE: TicketResponse schema does NOT expose a 'decision' field.
        The route handler computes decision internally and maps it to the
        ticket 'status' field (auto_resolved → 'auto_resolved', else 'escalated').
        This test documents that behaviour and verifies 'status' is present.
        Schema gap tracked: TicketResponse should expose 'decision' explicitly.
        """
        with patch(
            "app.routes.ticket_routes.ResolutionService.resolve",
            new_callable=AsyncMock,
            return_value=_fake_resolution(1, confidence=0.82),
        ):
            data = self._post_ticket(client, sample_ticket_payload).json()

        # 'decision' is NOT in TicketResponse — 'status' reflects the outcome
        assert "status" in data
        assert data["status"] in ("open", "auto_resolved", "escalated", "in_progress", "closed")


# ── Update ticket ─────────────────────────────────────────────────────────────

class TestUpdateTicket:
    """PATCH /api/v1/tickets/{id}"""

    def test_patch_nonexistent_ticket_returns_404(self, client):
        response = client.patch(
            "/api/v1/tickets/99999",
            json={"status": "closed"},
        )
        assert response.status_code == 404


# ── Delete ticket ─────────────────────────────────────────────────────────────

class TestDeleteTicket:
    """DELETE /api/v1/tickets/{id}"""

    def test_delete_nonexistent_ticket_returns_404(self, client):
        response = client.delete("/api/v1/tickets/99999")
        assert response.status_code == 404
