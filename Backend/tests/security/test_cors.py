"""
test_cors.py — Security tests for CORS behavior and origin validation.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_cors_accepts_configured_origin(client):
    """Verify configured CORS origin receives Access-Control-Allow-Origin header."""
    allowed_origin = settings.parsed_cors_origins[0]
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": allowed_origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == allowed_origin


def test_cors_rejects_unconfigured_origin(client):
    """Verify unconfigured unauthorized origin does NOT receive allow-origin header."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://evil-hacker-site.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # FastAPI CORSMiddleware omits access-control-allow-origin header for untrusted origins
    assert (
        response.headers.get("access-control-allow-origin")
        != "http://evil-hacker-site.com"
    )
