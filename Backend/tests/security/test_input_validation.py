"""
test_input_validation.py — Security tests for API input payload validation.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_invalid_category_rejected(client):
    """Verify ticket creation with invalid category raises HTTP 422 validation error."""
    payload = {
        "title": "Valid Ticket Title",
        "description": "Valid ticket description text meeting min length",
        "category": "malicious_unsupported_category",
    }
    res = client.post("/api/v1/tickets", json=payload)
    assert res.status_code == 422


def test_oversized_title_rejected(client):
    """Verify title exceeding max_length (255 chars) raises HTTP 422 validation error."""
    payload = {
        "title": "A" * 300,
        "description": "Valid ticket description text meeting min length",
        "category": "software",
    }
    res = client.post("/api/v1/tickets", json=payload)
    assert res.status_code == 422


def test_oversized_description_rejected(client):
    """Verify description exceeding max_length (10000 chars) raises HTTP 422 validation error."""
    payload = {
        "title": "Valid Title",
        "description": "B" * 12000,
        "category": "software",
    }
    res = client.post("/api/v1/tickets", json=payload)
    assert res.status_code == 422
