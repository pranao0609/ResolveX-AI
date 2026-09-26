"""
logger.py — Centralized structured logger for ResolveX-AI with automatic secret redaction.
"""

import logging
import re
import sys

SECRET_PATTERNS = [
    re.compile(r"(gsk_[a-zA-Z0-9_\-]+)", re.IGNORECASE),
    re.compile(r"(sk-[a-zA-Z0-9_\-]+)", re.IGNORECASE),
    re.compile(r"(lsv2_[a-zA-Z0-9_\-]+)", re.IGNORECASE),
    re.compile(r"(Bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE),
    re.compile(r"(password\s*=\s*['\"]?[^\s'\"]+['\"]?)", re.IGNORECASE),
]


def redact_secrets(message: str) -> str:
    """Redact known API key patterns and credentials from log messages."""
    if not isinstance(message, str):
        return message
    clean_msg = message
    for pattern in SECRET_PATTERNS:
        clean_msg = pattern.sub("[REDACTED_SECRET]", clean_msg)
    return clean_msg


class SecretRedactingFormatter(logging.Formatter):
    """Logging Formatter that automatically redacts API keys and secrets."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return redact_secrets(formatted)


def _build_logger(name: str) -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        SecretRedactingFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _logger = logging.getLogger(name)
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False
    return _logger


# Single shared logger instance — import this everywhere
logger = _build_logger("resolvex")
