"""
retrieval_tools.py

Explicit retrieval tools used by the ResolveX Retrieval Agent.

Phase 16.3:
    - search_knowledge_base()
    - search_previous_tickets()
    - rerank_documents()

The tools provide a stable interface between the retrieval agent
and the underlying retrieval infrastructure.
"""

from __future__ import annotations

from typing import Any, Sequence

from ai.config.ai_config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_STRATEGY,
    RETRIEVAL_TOP_K,
)
from ai.rag.doc_store import get_doc_store
from ai.rag.hybrid_retriever import HybridRetriever
from ai.rag.retrieval_models import RetrievalCandidate
from ai.rag.reranker import get_reranker
from app.core.logger import logger


# =====================================================================
# Shared Retrieval Infrastructure
# =====================================================================

_hybrid_retriever = HybridRetriever(
    bm25_weight=BM25_WEIGHT,
    dense_weight=DENSE_WEIGHT,
)

_doc_store = get_doc_store()


# =====================================================================
# Internal Helpers
# =====================================================================


def _candidate_to_dict(
    candidate: RetrievalCandidate,
) -> dict[str, Any]:
    """
    Convert a RetrievalCandidate into a serializable dictionary.
    """

    return {
        "index_id": int(
            candidate.index_id
        ),
        "score": float(
            candidate.score
        ),
        "retriever": (
            candidate.retriever
        ),
    }


def _resolve_candidates(
    candidates: Sequence[
        RetrievalCandidate
    ],
) -> list[dict[str, Any]]:
    """
    Resolve retrieval candidates against DocumentStore.

    Candidates without a corresponding document are skipped.
    """

    documents: list[dict[str, Any]] = []

    for candidate in candidates:

        document = _doc_store.get_chunk(
            candidate.index_id
        )

        if document is None:
            logger.warning(
                "Retrieval tool could not resolve "
                "index_id=%s",
                candidate.index_id,
            )
            continue

        document_data = (
            document.model_dump()
        )

        document_data[
            "index_id"
        ] = int(
            candidate.index_id
        )

        document_data[
            "score"
        ] = float(
            candidate.score
        )

        document_data[
            "retriever"
        ] = candidate.retriever

        documents.append(
            document_data
        )

    return documents


# =====================================================================
# Tool 1 — Search Knowledge Base
# =====================================================================


def search_knowledge_base(
    query: str,
    *,
    strategy: str | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search the ResolveX knowledge base.

    Supported strategies:
        - dense
        - bm25
        - hybrid

    Args:
        query:
            Search query.

        strategy:
            Retrieval strategy. Defaults to configured
            RETRIEVAL_STRATEGY.

        top_k:
            Number of final candidates.

        candidate_k:
            Number of candidates used by hybrid retrieval.

    Returns:
        List of serializable document dictionaries.
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "Knowledge-base query must be a string"
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Knowledge-base query must not be empty"
        )

    selected_strategy = (
        strategy
        or RETRIEVAL_STRATEGY
    )

    selected_top_k = (
        top_k
        if top_k is not None
        else RETRIEVAL_TOP_K
    )

    selected_candidate_k = (
        candidate_k
        if candidate_k is not None
        else RETRIEVAL_CANDIDATE_K
    )

    if selected_top_k <= 0:
        return []

    logger.info(
        "retrieval_tool=search_knowledge_base "
        "strategy=%s top_k=%s candidate_k=%s",
        selected_strategy,
        selected_top_k,
        selected_candidate_k,
    )

    # ---------------------------------------------------------------
    # Hybrid retrieval
    # ---------------------------------------------------------------

    if selected_strategy == "hybrid":

        candidates = (
            _hybrid_retriever.retrieve(
                query=query,
                top_k=selected_top_k,
                candidate_k=selected_candidate_k,
            )
        )

        return _resolve_candidates(
            candidates
        )

    # ---------------------------------------------------------------
    # BM25 retrieval
    # ---------------------------------------------------------------

    if selected_strategy == "bm25":

        results = (
            _hybrid_retriever.bm25_store.search(
                query=query,
                top_k=selected_top_k,
            )
        )

        candidates = [
            RetrievalCandidate(
                index_id=int(
                    result["index_id"]
                ),
                score=float(
                    result["score"]
                ),
                retriever="bm25",
            )
            for result in results
        ]

        return _resolve_candidates(
            candidates
        )

    # ---------------------------------------------------------------
    # Dense retrieval
    # ---------------------------------------------------------------

    if selected_strategy == "dense":

        from ai.rag.retriever import (
            retrieve_context,
        )

        documents = retrieve_context(
            query,
            top_k=selected_top_k,
            score_threshold=-1.0,
        )

        return [
            {
                **document,
                "retriever": document.get(
                    "retriever",
                    "dense",
                ),
            }
            for document in documents
        ]

    raise ValueError(
        "Unsupported RETRIEVAL_STRATEGY: "
        f"{selected_strategy}"
    )


# =====================================================================
# Tool 2 — Search Previous Tickets
# =====================================================================


def search_previous_tickets(
    query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Search previously resolved support tickets.

    Phase 16.3 establishes the explicit tool contract.

    The current ResolveX retrieval layer does not yet expose a
    dedicated previous-ticket index. Therefore this function
    intentionally returns an empty result rather than treating
    knowledge-base documents as historical tickets.

    A dedicated historical-ticket store can be connected later
    without changing the Retrieval Agent's tool interface.
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "Previous-ticket query must be a string"
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Previous-ticket query must not be empty"
        )

    if top_k <= 0:
        return []

    logger.info(
        "retrieval_tool=search_previous_tickets "
        "status=unavailable "
        "query_length=%s "
        "top_k=%s",
        len(query),
        top_k,
    )

    return []


# =====================================================================
# Tool 3 — Rerank Documents
# =====================================================================


def rerank_documents(
    query: str,
    documents: Sequence[
        dict[str, Any]
    ],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved documents using the ResolveX
    cross-encoder reranker.

    Args:
        query:
            Query used for relevance scoring.

        documents:
            Documents produced by retrieval tools.

        top_k:
            Number of documents to return.

    Returns:
        Documents ordered by cross-encoder relevance.
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "Reranking query must be a string"
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Reranking query must not be empty"
        )

    if top_k <= 0:
        return []

    if not documents:
        return []

    # ---------------------------------------------------------------
    # Convert tool documents into RetrievalCandidate objects
    # ---------------------------------------------------------------

    candidates: list[
        RetrievalCandidate
    ] = []

    for document in documents:

        if "index_id" not in document:
            continue

        try:
            index_id = int(
                document["index_id"]
            )

            score = float(
                document.get(
                    "score",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            logger.warning(
                "Skipping invalid reranking document"
            )
            continue

        candidates.append(
            RetrievalCandidate(
                index_id=index_id,
                score=score,
                retriever=document.get(
                    "retriever",
                    "retrieval",
                ),
            )
        )

    if not candidates:
        return []

    # ---------------------------------------------------------------
    # Cross-encoder reranking
    # ---------------------------------------------------------------

    reranker = get_reranker()

    reranked_candidates = (
        reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
    )

    # ---------------------------------------------------------------
    # Resolve reranked candidates back to documents
    # ---------------------------------------------------------------

    reranked_documents = (
        _resolve_candidates(
            reranked_candidates
        )
    )

    # ---------------------------------------------------------------
    # Preserve explicit reranker score
    # ---------------------------------------------------------------

    return reranked_documents


# =====================================================================
# Tool Registry
# =====================================================================


RETRIEVAL_TOOLS = {
    "search_knowledge_base": (
        search_knowledge_base
    ),
    "search_previous_tickets": (
        search_previous_tickets
    ),
    "rerank_documents": (
        rerank_documents
    ),
}


def get_retrieval_tools() -> dict[
    str,
    Any,
]:
    """
    Return the explicit retrieval-tool registry.
    """

    return dict(
        RETRIEVAL_TOOLS
    )