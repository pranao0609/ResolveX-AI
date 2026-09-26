"""
test_langsmith_config.py — Unit tests for LangSmith configuration & environment setup.
"""

import os
from unittest.mock import patch

from ai.observability.langsmith_config import (
    configure_langsmith_environment,
    get_langsmith_api_key,
    get_langsmith_endpoint,
    get_langsmith_project,
    is_langsmith_enabled,
)


def test_default_tracing_is_disabled(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "LANGSMITH_TRACING", False)
    with patch.dict(os.environ, {}, clear=True):
        assert is_langsmith_enabled() is False


def test_is_langsmith_enabled_truthy():
    truthy_vals = ["true", "True", "1", "yes", "ON"]
    for val in truthy_vals:
        with patch.dict(os.environ, {"LANGSMITH_TRACING": val}, clear=True):
            assert is_langsmith_enabled() is True, f"Failed for LANGSMITH_TRACING={val}"

    for val in truthy_vals:
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": val}, clear=True):
            assert (
                is_langsmith_enabled() is True
            ), f"Failed for LANGCHAIN_TRACING_V2={val}"


def test_is_langsmith_enabled_falsy():
    falsy_vals = ["false", "False", "0", "no", "OFF", "random_string"]
    for val in falsy_vals:
        with patch.dict(os.environ, {"LANGSMITH_TRACING": val}, clear=True):
            assert (
                is_langsmith_enabled() is False
            ), f"Failed for LANGSMITH_TRACING={val}"


def test_get_langsmith_api_key():
    with patch.dict(os.environ, {"LANGSMITH_API_KEY": "ls_test_key_123"}, clear=True):
        assert get_langsmith_api_key() == "ls_test_key_123"

    with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "lc_test_key_456"}, clear=True):
        assert get_langsmith_api_key() == "lc_test_key_456"


def test_get_langsmith_project():
    with patch.dict(os.environ, {"LANGSMITH_PROJECT": "TestProject"}, clear=True):
        assert get_langsmith_project() == "TestProject"


def test_get_langsmith_endpoint():
    with patch.dict(
        os.environ, {"LANGSMITH_ENDPOINT": "https://custom.endpoint.com"}, clear=True
    ):
        assert get_langsmith_endpoint() == "https://custom.endpoint.com"


def test_configure_langsmith_environment_disabled():
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}, clear=True):
        active = configure_langsmith_environment()
        assert active is False
        assert os.environ.get("LANGSMITH_TRACING") == "false"
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"


def test_configure_langsmith_environment_enabled_with_key():
    env_vars = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "valid_mock_key_abc",
        "LANGSMITH_PROJECT": "ResolveX-Test",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        active = configure_langsmith_environment()
        assert active is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "valid_mock_key_abc"
        assert os.environ.get("LANGCHAIN_PROJECT") == "ResolveX-Test"


def test_configure_langsmith_environment_missing_key():
    env_vars = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        active = configure_langsmith_environment()
        assert active is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"
        assert os.environ.get("LANGSMITH_TRACING") == "false"
