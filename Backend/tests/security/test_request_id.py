"""
test_request_id.py — Security tests for HTTP correlation IDs.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_request_id_generated_if_missing(client):
    """Verify X-Request-ID header is generated and returned if missing in request."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    req_id = res.headers.get("X-Request-ID")
    assert req_id is not None
    assert len(req_id) > 10


def test_request_id_propagated_if_supplied(client):
    """Verify valid incoming X-Request-ID is preserved in response headers."""
    custom_id = "test-req-id-12345"
    res = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID") == custom_id
