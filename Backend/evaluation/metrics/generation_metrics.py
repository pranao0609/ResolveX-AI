"""
ResolveX generation evaluation metrics.

Uses semantic similarity from the same sentence-transformer
family already used by the ResolveX RAG system.

Metrics:
    - Faithfulness
    - Answer Relevance
    - Context Relevance
    - Answer Correctness
    - Evidence Support
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    """Load the evaluation embedding model lazily."""

    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def _normalize(text: str) -> str:
    """Normalize text."""

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _sentences(text: str) -> list[str]:
    """Split text into sentences."""

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def _cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Calculate cosine similarity."""

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    score = np.dot(
        vector_a,
        vector_b,
    ) / denominator

    return float(
        np.clip(score, 0.0, 1.0)
    )


def _semantic_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """Calculate semantic similarity between two texts."""

    text_a = _normalize(text_a)
    text_b = _normalize(text_b)

    if not text_a or not text_b:
        return 0.0

    model = _get_model()

    embeddings = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
    )

    return float(
        np.clip(
            np.dot(
                embeddings[0],
                embeddings[1],
            ),
            0.0,
            1.0,
        )
    )


def answer_relevance(
    generated_answer: str,
    ticket: str,
) -> float:
    """
    Measure semantic relevance between the generated answer
    and the original ticket.
    """

    return _semantic_similarity(
        generated_answer,
        ticket,
    )


def answer_correctness(
    generated_answer: str,
    expected_answer: str,
) -> float:
    """
    Measure semantic similarity between the generated answer
    and the manually curated reference answer.
    """

    return _semantic_similarity(
        generated_answer,
        expected_answer,
    )


def context_relevance(
    generated_answer: str,
    retrieved_documents: list[dict],
) -> float:
    """
    Measure semantic relevance between the generated answer
    and retrieved knowledge-base context.
    """

    if not generated_answer:
        return 0.0

    if not retrieved_documents:
        return 0.0

    context = " ".join(
        str(
            document.get(
                "content",
                "",
            )
        )
        for document in retrieved_documents
    )

    return _semantic_similarity(
        generated_answer,
        context,
    )


def faithfulness(
    generated_answer: str,
    retrieved_documents: list[dict],
) -> float:
    """
    Estimate whether generated statements are grounded
    in retrieved context.

    Each generated sentence is compared semantically
    against the retrieved context. The average sentence
    grounding score is returned.
    """

    if not generated_answer:
        return 0.0

    if not retrieved_documents:
        return 0.0

    context = " ".join(
        str(
            document.get(
                "content",
                "",
            )
        )
        for document in retrieved_documents
    )

    generated_sentences = _sentences(
        generated_answer
    )

    if not generated_sentences:
        return 0.0

    model = _get_model()

    context_embedding = model.encode(
        context,
        normalize_embeddings=True,
    )

    sentence_embeddings = model.encode(
        generated_sentences,
        normalize_embeddings=True,
    )

    scores = []

    for sentence_embedding in sentence_embeddings:
        score = np.dot(
            sentence_embedding,
            context_embedding,
        )

        scores.append(
            float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )
        )

    return float(
        np.mean(scores)
    )


def evidence_support(
    generated_answer: str,
    required_evidence: list[str],
    retrieved_documents: list[dict],
) -> float:
    """
    Measure whether required evidence is represented by
    the retrieved documents or generated answer.

    Evidence is matched semantically rather than requiring
    the exact KB title to appear in the answer.
    """

    if not required_evidence:
        return 1.0

    if not retrieved_documents:
        return 0.0

    retrieved_titles = [
        document.get(
            "title",
            "",
        )
        for document in retrieved_documents
    ]

    supported = 0

    for evidence in required_evidence:

        best_score = 0.0

        for title in retrieved_titles:
            score = _semantic_similarity(
                evidence,
                title,
            )

            best_score = max(
                best_score,
                score,
            )

        answer_score = _semantic_similarity(
            evidence,
            generated_answer,
        )

        best_score = max(
            best_score,
            answer_score,
        )

        if best_score >= 0.55:
            supported += 1

    return supported / len(
        required_evidence
    )


def evaluate_generation_case(
    ticket: str,
    generated_answer: str,
    expected_answer: str,
    retrieved_documents: list[dict],
    required_evidence: list[str],
) -> dict:
    """Calculate all generation metrics."""

    return {
        "faithfulness": faithfulness(
            generated_answer,
            retrieved_documents,
        ),
        "answer_relevance": answer_relevance(
            generated_answer,
            ticket,
        ),
        "context_relevance": context_relevance(
            generated_answer,
            retrieved_documents,
        ),
        "answer_correctness": answer_correctness(
            generated_answer,
            expected_answer,
        ),
        "evidence_support": evidence_support(
            generated_answer,
            required_evidence,
            retrieved_documents,
        ),
    }