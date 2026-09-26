"""
ticket_pipeline.py — Master orchestrator of the ResolveX-AI processing pipeline.

Pipeline flow:
    Ticket Input
    → preprocess  (clean text + extract from attachments)
    → classify    (assign category)
    → embed       (generate vector embedding)
    → retrieve    (RAG: fetch similar KB entries)
    → generate    (Groq LLM produces structured resolution)
    → confidence  (compute composite score)
    → explain     (generate human-readable reasoning)
"""

import time

from app.core.logger import logger

from ai.preprocessing.text_cleaner import clean_text
from ai.preprocessing.file_parser import parse_attachments
from ai.classification.classifier import classify_ticket
from ai.embedding.embedding_model import generate_embedding

from ai.config.ai_config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_STRATEGY,
    RETRIEVAL_TOP_K,
)

from ai.rag.doc_store import get_doc_store
from ai.rag.hybrid_retriever import HybridRetriever

from ai.llm.solution_generator import generate_solution

from ai.confidence.confidence_engine import compute_confidence
from ai.explainability.explainer import explain

hybrid_retriever = HybridRetriever(
    bm25_weight=BM25_WEIGHT,
    dense_weight=DENSE_WEIGHT,
)

doc_store = get_doc_store()


async def run_pipeline(
    ticket,
    include_evaluation_details: bool = False,
    prompt_version: int = 1,
) -> dict:
    """
    Run the full AI pipeline on a Ticket ORM object.

    Args:
        ticket:
            Ticket ORM instance or compatible object.

        include_evaluation_details:
            When True, include cleaned text and retrieved
            documents in the returned result. This is used
            by the evaluation framework and does not affect
            normal production responses.

    Returns:
        Dictionary containing:

            solution
            diagnosis
            root_cause
            resolution_steps
            evidence
            confidence
            requires_human
            category
            explanation
            fallback_used

        When include_evaluation_details=True, also contains:

            evaluation
    """

    start_time = time.time()

    logger.info(f"[Pipeline] Starting for ticket_id={ticket.id}")

    # ── Step 1: Preprocessing ────────────────────────────────────────────────

    t0 = time.time()

    cleaned_text = clean_text(ticket.description)

    # Append any extracted text from attached files
    # (OCR / PDF parsing).
    if ticket.attachment_paths:
        paths = [p.strip() for p in ticket.attachment_paths.split(",") if p.strip()]

        extracted = parse_attachments(paths)

        if extracted:
            cleaned_text = f"{cleaned_text}\n\n" f"[Attachments]\n" f"{extracted}"

    preprocess_ms = (time.time() - t0) * 1000

    logger.info(
        f"ticket_id={ticket.id} "
        f"stage=preprocessing "
        f"status=success "
        f"text_length={len(cleaned_text)} "
        f"latency_ms={preprocess_ms:.2f}"
    )

    # ── Step 2: Classification ──────────────────────────────────────────────

    t0 = time.time()

    category, classification_score = classify_ticket(cleaned_text)

    classify_ms = (time.time() - t0) * 1000

    logger.info(
        f"ticket_id={ticket.id} "
        f"stage=classification "
        f"category={category} "
        f"score={classification_score:.3f} "
        f"latency_ms={classify_ms:.2f}"
    )

    # ── Step 3: Embedding ────────────────────────────────────────────────────

    t0 = time.time()

    embedding = generate_embedding(cleaned_text)

    embed_ms = (time.time() - t0) * 1000

    logger.info(
        f"ticket_id={ticket.id} "
        f"stage=embedding "
        f"status=success "
        f"latency_ms={embed_ms:.2f}"
    )

    # ── Step 4: RAG Retrieval ────────────────────────────────────────────────

    t0 = time.time()

    retrieval_candidates = None
    context_docs = []

    if RETRIEVAL_STRATEGY == "hybrid":

        retrieval_candidates = hybrid_retriever.retrieve(
            query=cleaned_text,
            top_k=RETRIEVAL_TOP_K,
            candidate_k=RETRIEVAL_CANDIDATE_K,
        )

    elif RETRIEVAL_STRATEGY == "bm25":

        retrieval_candidates = [
            {
                "index_id": result["index_id"],
                "score": result["score"],
                "retriever": "bm25",
            }
            for result in (
                hybrid_retriever.bm25_store.search(
                    query=cleaned_text,
                    top_k=RETRIEVAL_TOP_K,
                )
            )
        ]

    elif RETRIEVAL_STRATEGY == "dense":

        from ai.rag.retriever import retrieve_context

        context_docs = retrieve_context(
            embedding,
            top_k=RETRIEVAL_TOP_K,
            score_threshold=-1.0,
        )

    else:

        raise ValueError(f"Unsupported RETRIEVAL_STRATEGY: " f"{RETRIEVAL_STRATEGY}")

    # Convert retrieval candidates into actual
    # knowledge-base documents.
    if retrieval_candidates is not None:

        for candidate in retrieval_candidates:

            document = doc_store.get_chunk(candidate.index_id)

            if document is None:
                continue

            doc_data = document.model_dump()

            doc_data["score"] = float(candidate.score)

            doc_data["retriever"] = candidate.retriever

            # Preserve the internal index ID so the
            # evaluation system can map it back to
            # the KB document ID.
            doc_data["index_id"] = candidate.index_id

            context_docs.append(doc_data)

    context_text = "\n\n---\n\n".join(
        (
            f"[{d.get('source', '').upper()}] "
            f"{d.get('title', '')}\n"
            f"{d.get('content', '')}"
        )
        for d in context_docs
    )

    retrieve_ms = (time.time() - t0) * 1000

    logger.info(
        f"ticket_id={ticket.id} "
        f"stage=retrieval "
        f"strategy={RETRIEVAL_STRATEGY} "
        f"result_count={len(context_docs)} "
        f"latency_ms={retrieve_ms:.2f}"
    )

    # ── Step 5: Structured LLM Solution Generation ──────────────────────────

    t0 = time.time()

    resolution, fallback_used = generate_solution(
        cleaned_text,
        context_text,
        prompt_version=prompt_version,
    )

    llm_ms = (time.time() - t0) * 1000

    # The LLM confidence is now supplied by the
    # validated ResolutionResult instead of the
    # previous response-length heuristic.
    llm_score = resolution.confidence

    logger.info(
        f"ticket_id={ticket.id} "
        f"stage=llm_generation "
        f"prompt_version={prompt_version} "
        f"llm_confidence={llm_score:.3f} "
        f"requires_human={resolution.requires_human} "
        f"fallback={fallback_used} "
        f"latency_ms={llm_ms:.2f}"
    )

    # ── Step 6: Confidence Scoring ───────────────────────────────────────────

    t0 = time.time()

    similarity_score = _compute_similarity_score(context_docs)

    confidence = compute_confidence(
        similarity_score=similarity_score,
        llm_score=llm_score,
        classification_score=classification_score,
    )

    confidence_ms = (time.time() - t0) * 1000

    logger.info(
        f"ticket_id={ticket.id} "
        f"stage=confidence "
        f"score={confidence:.4f} "
        f"sim={similarity_score:.3f} "
        f"latency_ms={confidence_ms:.2f}"
    )

    # ── Step 7: Explainability ───────────────────────────────────────────────

    # Preserve the existing explainability interface by
    # passing a readable solution representation.
    solution_text = _format_solution(resolution)

    explanation = explain(
        ticket_text=cleaned_text,
        category=category,
        context_docs=context_docs,
        solution=solution_text,
        confidence=confidence,
    )

    # ── Pipeline Completion ──────────────────────────────────────────────────

    elapsed_ms = (time.time() - start_time) * 1000

    logger.info(
        f"ticket_id={ticket.id} "
        f"stage=pipeline_completion "
        f"confidence={confidence:.4f} "
        f"fallback={fallback_used} "
        f"total_latency_ms={elapsed_ms:.2f}"
    )

    # ── Build Result ─────────────────────────────────────────────────────────

    result = {
        # Backward-compatible human-readable solution.
        "solution": solution_text,
        # Structured LLM output.
        "diagnosis": resolution.diagnosis,
        "root_cause": resolution.root_cause,
        "resolution_steps": resolution.resolution_steps,
        "evidence": resolution.evidence,
        "llm_confidence": resolution.confidence,
        "requires_human": resolution.requires_human,
        # ResolveX composite confidence.
        "confidence": confidence,
        "category": category,
        "explanation": explanation,
        "fallback_used": fallback_used,
    }

    # Evaluation-only information.
    #
    # This is deliberately excluded from normal production
    # responses unless explicitly requested.
    if include_evaluation_details:
        result["structured_output"] = resolution.model_dump()

        result["evaluation"] = {
            "cleaned_text": cleaned_text,
            "context_docs": context_docs,
            "retrieval_strategy": RETRIEVAL_STRATEGY,
            "prompt_version": prompt_version,
        }

        logger.info(
            f"[Evaluation] "
            f"ticket_id={ticket.id} "
            f"context_docs={len(context_docs)}"
        )

    return result


def _format_solution(
    resolution,
) -> str:
    """
    Convert the structured ResolutionResult into a
    human-readable solution string.

    This preserves compatibility with the existing
    explainability layer and frontend.
    """

    steps = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(
            resolution.resolution_steps,
            start=1,
        )
    )

    evidence = (
        "\n".join(f"- {item}" for item in resolution.evidence)
        if resolution.evidence
        else "No specific evidence cited."
    )

    return (
        f"Diagnosis:\n"
        f"{resolution.diagnosis}\n\n"
        f"Root Cause:\n"
        f"{resolution.root_cause}\n\n"
        f"Resolution Steps:\n"
        f"{steps}\n\n"
        f"Evidence:\n"
        f"{evidence}"
    )


def _compute_similarity_score(
    context_docs: list,
) -> float:
    """
    Derive similarity score from real retrieval scores.

    Each document in context_docs contains a 'score'
    generated by the active retrieval strategy.

    The maximum score is used so that one highly relevant
    document can raise the similarity signal.
    """

    if not context_docs:
        return 0.0

    scores = [
        d.get(
            "score",
            0.0,
        )
        for d in context_docs
        if isinstance(d, dict)
    ]

    return float(max(scores)) if scores else 0.0
