from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryAnalysis:
    """
    Structured analysis of a ticket before knowledge retrieval.
    """

    original_query: str
    normalized_query: str
    retrieval_query: str
    rewrite_needed: bool
    rewrite_strategy: str
    rewrite_reason: str


def _normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _remove_ticket_noise(text: str) -> str:
    """
    Remove common ticket formatting noise while preserving
    technical information.
    """

    text = re.sub(
        r"\b(ticket|incident|case)\s*(id|number)?\s*[:#-]?\s*\d+\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(urgent|urgently|asap|please help|please assist)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return _normalize_whitespace(text)


def _looks_like_error_query(text: str) -> bool:
    """
    Detect common support/error language.
    """

    error_patterns = (
        r"\berror\b",
        r"\bfail(?:s|ed|ure)?\b",
        r"\bcrash(?:es|ed)?\b",
        r"\bexception\b",
        r"\bunable\b",
        r"\bcan't\b",
        r"\bcannot\b",
        r"\bnot working\b",
        r"\bdoesn't work\b",
        r"\bdisconnect(?:s|ed)?\b",
        r"\btimeout\b",
        r"\bdenied\b",
        r"\bissue\b",
        r"\bproblem\b",
    )

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in error_patterns
    )


def _looks_too_short(text: str) -> bool:
    """
    Determine whether the query contains too little information
    for reliable retrieval.
    """

    tokens = text.split()

    return len(tokens) < 4


def _build_keyword_query(text: str) -> str:
    """
    Build a retrieval-oriented query by removing low-value
    conversational words while preserving technical terms.
    """

    stop_words = {
        "i",
        "am",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "my",
        "me",
        "to",
        "of",
        "for",
        "and",
        "on",
        "in",
        "with",
        "please",
        "help",
        "can",
        "you",
        "we",
        "our",
    }

    tokens = re.findall(
        r"[A-Za-z0-9_.:/-]+",
        text,
    )

    filtered = [token for token in tokens if token.lower() not in stop_words]

    return " ".join(filtered).strip()


def analyze_query(query: str) -> QueryAnalysis:
    """
    Analyze a support-ticket query and determine whether
    retrieval-oriented rewriting is necessary.

    This function does not call an LLM.

    Rewrite is triggered when:
        - the query is very short, or
        - the query contains conversational/error phrasing
          that can be converted into a retrieval-oriented form.
    """

    original_query = query or ""

    normalized_query = _normalize_whitespace(original_query)

    cleaned_query = _remove_ticket_noise(normalized_query)

    if not cleaned_query:
        return QueryAnalysis(
            original_query=original_query,
            normalized_query="",
            retrieval_query="",
            rewrite_needed=False,
            rewrite_strategy="none",
            rewrite_reason="Query contains no retrievable text.",
        )

    too_short = _looks_too_short(cleaned_query)

    error_query = _looks_like_error_query(cleaned_query)

    keyword_query = _build_keyword_query(cleaned_query)

    if too_short and keyword_query:
        return QueryAnalysis(
            original_query=original_query,
            normalized_query=cleaned_query,
            retrieval_query=keyword_query,
            rewrite_needed=True,
            rewrite_strategy="keyword_extraction",
            rewrite_reason=(
                "The query is short and was converted "
                "into a retrieval-oriented keyword query."
            ),
        )

    if error_query and keyword_query:
        return QueryAnalysis(
            original_query=original_query,
            normalized_query=cleaned_query,
            retrieval_query=keyword_query,
            rewrite_needed=True,
            rewrite_strategy="technical_keyword_extraction",
            rewrite_reason=(
                "The query contains support/error language "
                "and was converted into a technical retrieval query."
            ),
        )

    return QueryAnalysis(
        original_query=original_query,
        normalized_query=cleaned_query,
        retrieval_query=cleaned_query,
        rewrite_needed=False,
        rewrite_strategy="none",
        rewrite_reason=("The normalized ticket text is suitable for retrieval."),
    )
