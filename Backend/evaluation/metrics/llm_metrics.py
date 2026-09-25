from __future__ import annotations

import re
from typing import Any

import numpy as np

from ai.llm.schemas import ResolutionResult


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize text for lexical comparison."""
    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def _tokenize(text: str) -> set[str]:
    """Return normalized content-word tokens."""
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "may",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "when",
        "with",
    }

    return {
        token
        for token in re.findall(
            r"\b[a-zA-Z0-9_]+\b",
            _normalize_text(text),
        )
        if token not in stopwords
    }


def _jaccard_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """Compute lexical Jaccard similarity."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def correctness(
    generated_answer: str,
    expected_resolution: str,
) -> float:
    """
    Measure similarity between the generated resolution
    and the expected resolution.

    This is intentionally a lightweight deterministic baseline.
    Phase 12 can later introduce an LLM-as-a-judge evaluator.
    """
    return _jaccard_similarity(
        generated_answer,
        expected_resolution,
    )


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------


def relevance(
    ticket: str,
    generated_answer: str,
) -> float:
    """
    Measure whether the generated answer addresses
    the original ticket.
    """
    return _jaccard_similarity(
        ticket,
        generated_answer,
    )


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def faithfulness(
    generated_answer: str,
    retrieved_context: str,
) -> float:
    """
    Estimate how strongly the generated answer is supported
    by retrieved context.

    The answer is evaluated sentence-by-sentence.
    """
    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?]+",
            generated_answer,
        )
        if sentence.strip()
    ]

    if not sentences:
        return 0.0

    scores = [
        _jaccard_similarity(
            sentence,
            retrieved_context,
        )
        for sentence in sentences
    ]

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Hallucination
# ---------------------------------------------------------------------------


def hallucination(
    generated_answer: str,
    retrieved_context: str,
) -> float:
    """
    Estimate unsupported content.

    Returns:
        0.0 -> no detected unsupported content
        1.0 -> completely unsupported according to this
                deterministic lexical baseline
    """
    return 1.0 - faithfulness(
        generated_answer,
        retrieved_context,
    )


# ---------------------------------------------------------------------------
# Instruction adherence
# ---------------------------------------------------------------------------


def instruction_adherence(
    result: ResolutionResult | dict[str, Any],
) -> float:
    """
    Check whether the structured response satisfies the
    required resolution schema and basic content requirements.

    Accepts either a ResolutionResult instance or a
    dictionary representation of the structured output.
    """

    try:
        if isinstance(result, dict):
            result = ResolutionResult.model_validate(result)

        elif not isinstance(result, ResolutionResult):
            return 0.0

    except Exception:
        return 0.0

    checks = [
        bool(result.diagnosis.strip()),
        bool(result.root_cause.strip()),
        bool(result.resolution_steps),
        all(
            isinstance(step, str) and step.strip()
            for step in result.resolution_steps
        ),
        all(
            isinstance(item, str) and item.strip()
            for item in result.evidence
        ),
        0.0 <= result.confidence <= 1.0,
        isinstance(result.requires_human, bool),
    ]

    return sum(checks) / len(checks)


# ---------------------------------------------------------------------------
# Structured-output validity
# ---------------------------------------------------------------------------


def structured_output_validity(
    output: Any,
) -> float:
    """
    Validate an LLM output against ResolutionResult.

    Returns:
        1.0 if valid
        0.0 if invalid
    """

    try:
        if isinstance(output, ResolutionResult):
            return 1.0

        ResolutionResult.model_validate(output)

        return 1.0

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def evaluate_llm_output(
    *,
    ticket: str,
    generated_answer: str,
    expected_resolution: str,
    retrieved_context: str,
    structured_output: Any,
) -> dict[str, float]:
    """
    Compute all Phase 12 deterministic LLM metrics.
    """

    faithfulness_score = faithfulness(
        generated_answer,
        retrieved_context,
    )

    return {
        "correctness": correctness(
            generated_answer,
            expected_resolution,
        ),
        "faithfulness": faithfulness_score,
        "relevance": relevance(
            ticket,
            generated_answer,
        ),
        "hallucination": hallucination(
            generated_answer,
            retrieved_context,
        ),
        "instruction_adherence": instruction_adherence(
        structured_output
        ),
        "structured_output_validity": (
            structured_output_validity(
                structured_output
            )
        ),
    }