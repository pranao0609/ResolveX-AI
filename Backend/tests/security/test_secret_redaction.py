"""
test_secret_redaction.py — Security tests for logging secret redaction.
"""

import pytest
from app.core.logger import redact_secrets


def test_redact_secrets_filters_groq_api_key():
    """Verify Groq API key patterns (gsk_...) are redacted as [REDACTED_SECRET]."""
    raw_msg = "Error connecting with key gsk_abc123456789xyz"
    clean = redact_secrets(raw_msg)
    assert "gsk_abc123456789xyz" not in clean
    assert "[REDACTED_SECRET]" in clean


def test_redact_secrets_filters_langsmith_api_key():
    """Verify LangSmith API key patterns (lsv2_...) are redacted as [REDACTED_SECRET]."""
    raw_msg = "LangSmith connecting with lsv2_secret_key_999"
    clean = redact_secrets(raw_msg)
    assert "lsv2_secret_key_999" not in clean
    assert "[REDACTED_SECRET]" in clean


def test_redact_secrets_filters_bearer_tokens():
    """Verify Bearer authorization header patterns are redacted."""
    raw_msg = "Header Authorization: Bearer eyJhbGciOiJIUzI1Ni..."
    clean = redact_secrets(raw_msg)
    assert "eyJhbGciOiJIUzI1Ni" not in clean
    assert "[REDACTED_SECRET]" in clean
