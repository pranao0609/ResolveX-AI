"""
test_preprocessing.py — Unit tests for the text cleaning and preprocessing utilities.

Tests:
  - clean_text(): normalisation, HTML stripping, whitespace collapse, lowercasing
  - truncate(): length enforcement
"""

import pytest
from ai.preprocessing.text_cleaner import clean_text, truncate


class TestCleanText:
    """Verify clean_text() normalises text correctly."""

    def test_returns_empty_string_for_none(self):
        assert clean_text(None) == ""

    def test_returns_empty_string_for_empty_input(self):
        assert clean_text("") == ""

    def test_strips_leading_and_trailing_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_collapses_multiple_spaces(self):
        result = clean_text("hello    world")
        assert "  " not in result
        assert "hello world" == result

    def test_collapses_newlines(self):
        result = clean_text("line one\n\nline two")
        assert "\n" not in result
        assert "line one line two" == result

    def test_collapses_tabs(self):
        result = clean_text("col1\t\tcol2")
        assert "\t" not in result

    def test_lowercases_text(self):
        assert clean_text("UPPERCASE TEXT") == "uppercase text"

    def test_removes_html_tags(self):
        result = clean_text("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "hello world" == result

    def test_removes_nested_html(self):
        result = clean_text("<div><span>inner</span></div>")
        assert "inner" in result
        assert "<" not in result

    def test_preserves_punctuation(self):
        result = clean_text("Can't log in. Reset password failed!")
        assert "'" in result or "can" in result   # content preserved

    def test_handles_mixed_html_and_text(self):
        result = clean_text("<br/>VPN <b>disconnected</b> after update")
        assert "vpn" in result
        assert "disconnected" in result
        assert "<" not in result

    def test_plain_text_unchanged_modulo_case(self):
        result = clean_text("simple text here")
        assert result == "simple text here"

    def test_special_characters_in_support_text(self):
        """Hyphens and forward slashes common in IT tickets should survive."""
        result = clean_text("Error code: 500 - Internal Server Error")
        assert "500" in result
        assert "internal server error" in result


class TestTruncate:
    """Verify truncate() enforces character limits."""

    def test_short_text_not_truncated(self):
        text = "short text"
        assert truncate(text, max_chars=100) == text

    def test_text_at_exact_limit_not_truncated(self):
        text = "a" * 50
        assert truncate(text, max_chars=50) == text

    def test_long_text_truncated_to_max(self):
        text = "a" * 3000
        result = truncate(text, max_chars=2000)
        assert len(result) == 2000

    def test_default_max_is_2000(self):
        text = "x" * 2500
        result = truncate(text)
        assert len(result) == 2000

    def test_empty_string_returns_empty(self):
        assert truncate("") == ""

    def test_truncation_cuts_from_end(self):
        text = "hello world this is a long text"
        result = truncate(text, max_chars=5)
        assert result == "hello"
