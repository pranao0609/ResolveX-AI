"""
explainer.py — Explainability module for AI-generated resolutions.
Produces a human-readable explanation of how the system arrived at its solution.
"""

from typing import List
from app.core.logger import logger
from app.core.constants import CONFIDENCE_HIGH, CONFIDENCE_LOW


def explain(
    ticket_text: str,
    category: str,
    context_docs: List[str],
    solution: str,
    confidence: float,
) -> str:
    """
    Generate a human-readable explanation of the AI resolution.

    Args:
        ticket_text:   Cleaned ticket text.
        category:      Predicted ticket category.
        context_docs:  Retrieved KB documents used for RAG context.
        solution:      LLM-generated solution text.
        confidence:    Final composite confidence score.

    Returns:
        Explanation string shown to the end user / admin.
    """
    lines = [
        "## ResolveX-AI Decision Explanation",
        "",
        f"**Ticket Category**: `{category}`",
        f"**Confidence Score**: `{confidence:.2%}`",
    ]

    # ── Confidence interpretation ─────────────────────────────────────────────
    if confidence >= CONFIDENCE_HIGH:
        lines.append("**Decision**: ✅ Auto-resolved (high confidence)")
    elif confidence < CONFIDENCE_LOW:
        lines.append("**Decision**: 🔴 Escalated to human agent (low confidence)")
    else:
        lines.append("**Decision**: 🟡 Escalated to human agent (moderate confidence)")

    # ── RAG context summary ───────────────────────────────────────────────────
    lines.append("")
    if context_docs:
        lines.append(
            f"**Supporting Context**: {len(context_docs)} similar past ticket(s) were used."
        )
    else:
        lines.append(
            "**Supporting Context**: No similar past tickets found in the knowledge base."
        )

    # ── Ticket keywords (placeholder for LIME/SHAP in production) ────────────
    keywords = _extract_keywords(ticket_text)
    if keywords:
        lines.append(f"**Key Terms Identified**: {', '.join(keywords)}")

    # ── Solution preview ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("**AI-Generated Solution Preview**:")
    lines.append(solution[:300] + "..." if len(solution) > 300 else solution)

    explanation = "\n".join(lines)
    logger.debug(f"Explanation generated ({len(explanation)} chars)")
    return explanation


def _extract_keywords(text: str) -> List[str]:
    """
    Extract simple top keywords from the ticket text.
    Placeholder: replace with TF-IDF or KeyBERT in production.
    """
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "it",
        "i",
        "we",
        "to",
        "of",
        "and",
        "for",
        "in",
        "on",
    }
    words = [w.strip(".,!?;:\"'") for w in text.split()]
    keywords = [w for w in words if len(w) > 4 and w not in stop_words]
    # Return top 5 most frequent (simple frequency count)
    from collections import Counter

    counter = Counter(keywords)
    return [word for word, _ in counter.most_common(5)]
