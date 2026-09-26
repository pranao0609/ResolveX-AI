"""
test_classifier.py — Unit tests for the ticket classification logic.

Tests the classify_ticket() function in isolation from the ML model
by verifying that the pattern-matching fast-path works correctly
without needing the transformers zero-shot model or Groq API.
"""

import pytest
from unittest.mock import patch, MagicMock

# ── Helper: import classifier with transformers pipeline mocked ───────────────
# The module-level `pipeline(...)` call downloads a model; we mock it early.


@pytest.fixture(autouse=True)
def mock_transformers_pipeline():
    """Prevent transformers from loading the bart-large-mnli model in tests."""
    with patch("transformers.pipeline") as mock_pipe:
        # Return a callable that returns a canned zero-shot result
        fake_result = {"labels": ["network_error"], "scores": [0.85]}
        mock_pipe.return_value = MagicMock(return_value=fake_result)
        yield mock_pipe


# ── Import after mocking so module-level code is intercepted ─────────────────

from ai.classification.classifier import (
    classify_ticket,
    llm_classify_fallback,
    CATEGORY_MAP,
    IT_CATEGORIES,
)


class TestPatternMatching:
    """Verify the regex pattern-match fast path (no model call needed)."""

    def test_vpn_classified_as_network(self):
        result, confidence = classify_ticket("VPN connection keeps dropping")
        assert result == "network"
        assert confidence >= 0.90

    def test_database_error_classified_as_software(self):
        result, confidence = classify_ticket("postgres database connection refused")
        assert result == "software"
        assert confidence >= 0.90

    def test_phishing_classified_as_security(self):
        # Use pure security vocabulary that doesn't overlap with other patterns.
        # 'phishing' and 'ransomware' only match the security_error pattern.
        result, confidence = classify_ticket(
            "ransomware detected on the system, possible security breach"
        )
        assert result == "security"
        assert confidence >= 0.90

    def test_login_classified_as_access_permission(self):
        result, confidence = classify_ticket(
            "password reset not working, access denied"
        )
        assert result == "access_permission"
        assert confidence >= 0.90

    def test_billing_classified_as_other(self):
        result, confidence = classify_ticket(
            "incorrect charge on my invoice this month"
        )
        assert result == "other"
        assert confidence >= 0.90

    def test_hardware_mouse_classified_correctly(self):
        result, confidence = classify_ticket("my mouse is not working at all")
        assert result == "hardware"
        assert confidence >= 0.90

    def test_network_wifi_classified_correctly(self):
        result, confidence = classify_ticket(
            "wifi drops out constantly, internet not working"
        )
        assert result == "network"
        assert confidence >= 0.90

    def test_bsod_classified_as_software(self):
        result, confidence = classify_ticket("blue screen BSOD windows error occurred")
        assert result == "software"
        assert confidence >= 0.90

    def test_printer_classified_as_hardware(self):
        result, confidence = classify_ticket("printer is offline and not working")
        assert result == "hardware"
        assert confidence >= 0.90


class TestEdgeCases:
    """Edge cases and boundary conditions for the classifier."""

    def test_empty_string_returns_software_default(self):
        result, confidence = classify_ticket("")
        assert result == "software"
        assert confidence == 0.50

    def test_very_short_text_below_threshold(self):
        # len("hi") == 2 < 5 → falls to default
        result, confidence = classify_ticket("hi")
        assert result == "software"
        assert confidence == 0.50

    def test_whitespace_only_text(self):
        result, confidence = classify_ticket("   ")
        assert result == "software"
        assert confidence == 0.50

    def test_confidence_is_float(self):
        _, confidence = classify_ticket("VPN not connecting")
        assert isinstance(confidence, float)

    def test_category_is_string(self):
        category, _ = classify_ticket("VPN not connecting")
        assert isinstance(category, str)

    def test_result_is_valid_final_category(self):
        """All returned categories must be mapped CATEGORY_MAP values."""
        valid_final_cats = set(CATEGORY_MAP.values())
        category, _ = classify_ticket("VPN not connecting")
        assert category in valid_final_cats


class TestCategoryMap:
    """Validate that CATEGORY_MAP covers all IT_CATEGORIES."""

    def test_all_it_categories_are_in_category_map(self):
        for cat in IT_CATEGORIES:
            assert cat in CATEGORY_MAP, f"IT_CATEGORY '{cat}' not found in CATEGORY_MAP"

    def test_category_map_values_are_non_empty_strings(self):
        for internal, final in CATEGORY_MAP.items():
            assert (
                isinstance(final, str) and len(final) > 0
            ), f"CATEGORY_MAP['{internal}'] must be a non-empty string"


class TestLLMFallback:
    """Test the Groq LLM fallback path without making real API calls."""

    def test_fallback_returns_default_when_no_api_key(self):
        """When Groq client cannot be created, fallback returns safe default."""
        with patch("ai.classification.classifier._get_groq_client", return_value=None):
            result, confidence = llm_classify_fallback("something unknown")
        assert result == "software"
        assert confidence == 0.75

    def test_fallback_parses_valid_groq_response(self):
        """Fallback correctly parses a well-formed Groq response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "login_error,0.88"
        )
        with patch(
            "ai.classification.classifier._get_groq_client", return_value=mock_client
        ):
            result, confidence = llm_classify_fallback("I cannot log in to my account")
        assert result == "access_permission"
        assert abs(confidence - 0.88) < 1e-6

    def test_fallback_handles_malformed_response(self):
        """Fallback returns safe default when response format is unexpected."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "this is not valid format"
        )
        with patch(
            "ai.classification.classifier._get_groq_client", return_value=mock_client
        ):
            result, confidence = llm_classify_fallback("some ticket text")
        assert result == "software"
        assert confidence == 0.80

    def test_fallback_handles_groq_exception(self):
        """When Groq call raises, fallback returns safe default."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Groq timeout")
        with patch(
            "ai.classification.classifier._get_groq_client", return_value=mock_client
        ):
            result, confidence = llm_classify_fallback("some ticket text")
        assert result == "software"
        assert confidence == 0.75
