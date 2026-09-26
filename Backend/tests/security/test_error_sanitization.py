"""
test_error_sanitization.py — Security tests for API error response sanitization.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    # Pass raise_server_exceptions=False so custom exception handlers process 500s
    return TestClient(app, raise_server_exceptions=False)


def test_uncaught_exception_does_not_leak_stack_trace(client):
    """Verify that unhandled server exceptions return clean 500 JSON without stack traces."""

    def dummy_error_route():
        raise RuntimeError(
            "Database connection string leaked: postgresql://usr:secret@host/db"
        )

    app.add_api_route("/api/v1/health/trigger-unhandled-error", dummy_error_route)

    res = client.get("/api/v1/health/trigger-unhandled-error")
    assert res.status_code == 500

    data = res.json()
    assert "error" in data
    assert data["error"] == "Internal server error"
    assert "request_id" in data
    assert "Traceback" not in res.text
    assert "postgresql://" not in res.text
    assert "RuntimeError" not in res.text
