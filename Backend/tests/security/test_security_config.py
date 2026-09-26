"""
test_security_config.py — Security tests for production configuration bounds.
"""

import pytest
from app.config import Settings


def test_production_environment_rejects_debug():
    """Verify that DEBUG=True in production environment raises ValueError during startup."""
    s = Settings(
        ENVIRONMENT="production",
        DEBUG=True,
        GROQ_API_KEY="valid_key",
        CORS_ORIGINS="http://company.com",
    )
    with pytest.raises(ValueError, match="DEBUG mode must be False in production"):
        s.validate_critical_settings()


def test_production_environment_rejects_wildcard_cors():
    """Verify that CORS_ORIGINS='*' in production environment raises ValueError."""
    s = Settings(
        ENVIRONMENT="production",
        DEBUG=False,
        GROQ_API_KEY="valid_key",
        CORS_ORIGINS="*",
    )
    with pytest.raises(
        ValueError, match=r"Wildcard CORS origin '\*' is forbidden in production"
    ):
        s.validate_critical_settings()


def test_production_environment_requires_valid_api_key():
    """Verify production fails if GROQ_API_KEY is missing or template default."""
    s = Settings(
        ENVIRONMENT="production",
        DEBUG=False,
        GROQ_API_KEY="your-groq-api-key-here",
        CORS_ORIGINS="http://company.com",
    )
    with pytest.raises(
        ValueError, match="GROQ_API_KEY must be configured in production"
    ):
        s.validate_critical_settings()


def test_valid_production_config():
    """Verify valid production settings pass validation cleanly."""
    s = Settings(
        ENVIRONMENT="production",
        DEBUG=False,
        GROQ_API_KEY="gsk_valid_prod_key",
        CORS_ORIGINS="https://app.resolvex.com",
    )
    s.validate_critical_settings()
    assert s.ENVIRONMENT == "production"
    assert s.DEBUG is False
