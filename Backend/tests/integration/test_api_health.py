"""
test_api_health.py — Integration tests for the health check endpoints.

Uses the FastAPI TestClient (no external dependencies required).
These are the simplest possible smoke tests that verify the API
is alive and returning the correct structure.

The FastAPI lifespan calls init_db() which normally connects to AWS RDS.
We patch it inside the client fixture so the mock is active when the
TestClient enters its context (i.e., when the lifespan fires).
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def api_client():
    """Module-scoped TestClient; health endpoints have no DB dependency.
    Patches init_db so we don't attempt a real AWS RDS connection.
    """
    with patch("app.main.init_db", return_value=None):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestLivenessProbe:
    """GET /api/v1/health — basic liveness check."""

    def test_health_returns_200(self, api_client):
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, api_client):
        data = api_client.get("/api/v1/health").json()
        assert data["status"] == "ok"

    def test_health_includes_app_name(self, api_client):
        data = api_client.get("/api/v1/health").json()
        assert "app" in data
        assert len(data["app"]) > 0

    def test_health_includes_version(self, api_client):
        data = api_client.get("/api/v1/health").json()
        assert "version" in data
        assert len(data["version"]) > 0

    def test_health_content_type_is_json(self, api_client):
        response = api_client.get("/api/v1/health")
        assert "application/json" in response.headers.get("content-type", "")


class TestReadinessProbe:
    """GET /api/v1/health/ready — readiness check."""

    def test_ready_returns_200(self, api_client):
        response = api_client.get("/api/v1/health/ready")
        assert response.status_code == 200

    def test_ready_returns_ready_status(self, api_client):
        data = api_client.get("/api/v1/health/ready").json()
        assert data["status"] == "ready"

    def test_ready_includes_database_field(self, api_client):
        data = api_client.get("/api/v1/health/ready").json()
        assert "database" in data

    def test_ready_includes_vector_store_field(self, api_client):
        data = api_client.get("/api/v1/health/ready").json()
        assert "vector_store" in data


class TestRequestCorrelationId:
    """Verify the X-Request-ID middleware adds a correlation header."""

    def test_response_contains_request_id_header(self, api_client):
        response = api_client.get("/api/v1/health")
        assert "x-request-id" in response.headers

    def test_custom_request_id_is_echoed_back(self, api_client):
        custom_id = "test-correlation-abc123"
        response = api_client.get(
            "/api/v1/health",
            headers={"X-Request-ID": custom_id},
        )
        assert response.headers.get("x-request-id") == custom_id


class TestNotFoundBehaviour:
    """Verify 404 on unknown endpoints."""

    def test_unknown_endpoint_returns_404(self, api_client):
        response = api_client.get("/api/v1/nonexistent-route")
        assert response.status_code == 404
