"""
metadata.py — Security & Metadata Sanitization Helper for ResolveX Observability.

Guarantees no sensitive credentials, API keys, passwords, authentication tokens,
or excessive PII are sent to telemetry or tracing metadata.
"""

from typing import Dict, Any

ALLOWED_SAFE_KEYS = {
    "token_usage",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
}

SENSITIVE_KEY_PATTERNS = {
    "key",
    "secret",
    "password",
    "auth",
    "credential",
    "bearer",
    "private",
}

# Specific secret token patterns (to avoid matching performance token metrics)
SENSITIVE_TOKEN_PATTERNS = {
    "access_token",
    "refresh_token",
    "auth_token",
    "secret_token",
    "api_token",
    "user_token",
    "session_token",
}


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter and sanitize dictionary metadata.
    Removes sensitive keys and truncates long strings to prevent PII/secret leaks.
    """
    if not isinstance(meta, dict):
        return {}

    sanitized = {}
    for key, value in meta.items():
        key_str = str(key).lower()

        # Check allowed safe keys first
        if key_str not in ALLOWED_SAFE_KEYS:
            if any(pat in key_str for pat in SENSITIVE_KEY_PATTERNS) or any(
                pat in key_str for pat in SENSITIVE_TOKEN_PATTERNS
            ):
                continue  # Omit sensitive keys completely

        if isinstance(value, dict):
            sanitized[key] = sanitize_metadata(value)
        elif isinstance(value, (list, tuple)):
            cleaned_list = []
            for item in value:
                if isinstance(item, dict):
                    cleaned_list.append(sanitize_metadata(item))
                elif isinstance(item, str) and len(item) > 500:
                    cleaned_list.append(item[:500] + "... [truncated]")
                else:
                    cleaned_list.append(item)
            sanitized[key] = cleaned_list
        elif isinstance(value, str):
            val_lower = value.lower()
            if any(
                pat in val_lower for pat in ("gsk_", "sk-", "bearer ", "authorization")
            ):
                continue
            if len(value) > 1000:
                sanitized[key] = value[:1000] + "... [truncated]"
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value

    return sanitized
