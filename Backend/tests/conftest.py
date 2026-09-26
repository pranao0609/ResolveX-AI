"""
conftest.py — Shared pytest fixtures for the ResolveX-AI Backend test suite.

Provides:
  - in-memory SQLite database (isolated per test session)
  - FastAPI TestClient with DB dependency override
  - reusable test data factories (ticket dict, resolution dict)
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db

# ── In-memory SQLite engine (no external dependency) ─────────────────────────

SQLALCHEMY_TEST_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Session-scoped DB setup ───────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create all tables once per test session, then drop them."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


# ── Per-test DB session ───────────────────────────────────────────────────────


@pytest.fixture()
def db_session():
    """Provide a transactional DB session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ── Override FastAPI DB dependency ────────────────────────────────────────────


@pytest.fixture()
def client(db_session):
    """Return a FastAPI TestClient with the DB overridden to the test session.
    Also patches init_db so the lifespan doesn't attempt AWS RDS connection.
    """
    from unittest.mock import patch

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with patch("app.main.init_db", return_value=None):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    app.dependency_overrides.clear()


# ── Test data factories ───────────────────────────────────────────────────────


@pytest.fixture()
def sample_ticket_payload():
    """Minimal valid ticket form payload."""
    return {
        "title": "VPN connection drops every 10 minutes",
        "description": "The corporate VPN disconnects automatically when idle for 10 minutes. "
        "This started after the latest Windows update.",
        "category": "network",
        "submitted_by": "test_user@example.com",
    }


@pytest.fixture()
def sample_resolution_data():
    """Fake pipeline output for mocking the resolution service."""
    return {
        "category": "network",
        "solution": "Try restarting the VPN client and checking your firewall rules.",
        "confidence": 0.82,
        "explanation": "VPN keywords detected with high similarity match.",
        "auto_resolved": True,
        "escalated_to_human": False,
        "assigned_resolver_id": None,
        "assigned_resolver_name": None,
        "assigned_resolver_category": None,
    }
