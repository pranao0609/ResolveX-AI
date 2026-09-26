"""
test_metadata_safety.py — Unit tests for metadata sanitization & secret protection.
"""

from ai.observability.metadata import sanitize_metadata


def test_sanitize_metadata_removes_sensitive_keys():
    raw = {
        "ticket_id": "12345",
        "api_key": "gsk_secret_key_should_be_removed",
        "secret_token": "bearer_12345",
        "user_password": "super_secret_password",
        "nested": {
            "auth_credentials": "admin:pass",
            "safe_field": "hello",
        },
    }

    sanitized = sanitize_metadata(raw)

    assert sanitized["ticket_id"] == "12345"
    assert "api_key" not in sanitized
    assert "secret_token" not in sanitized
    assert "user_password" not in sanitized
    assert sanitized["nested"]["safe_field"] == "hello"
    assert "auth_credentials" not in sanitized["nested"]


def test_sanitize_metadata_handles_none_and_primitives():
    assert sanitize_metadata(None) == {}
    assert sanitize_metadata({}) == {}
    assert sanitize_metadata({"key": None}) == {}
    assert sanitize_metadata({"count": 42, "ratio": 3.14, "flag": True}) == {
        "count": 42,
        "ratio": 3.14,
        "flag": True,
    }


def test_sanitize_metadata_truncates_large_strings():
    long_str = "A" * 2000
    sanitized = sanitize_metadata({"description": long_str})
    assert sanitized["description"].startswith("A" * 1000)
    assert sanitized["description"].endswith("... [truncated]")


def test_sanitize_metadata_handles_list_of_dicts():
    raw = {
        "items": [
            {"name": "item1", "secret": "shh"},
            {"name": "item2", "secret_token": "abc"},
        ]
    }
    sanitized = sanitize_metadata(raw)
    assert len(sanitized["items"]) == 2
    assert sanitized["items"][0] == {"name": "item1"}
    assert sanitized["items"][1] == {"name": "item2"}
