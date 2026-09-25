"""
retriever.py — RAG retrieval layer.

Uses FAISS vector store to find similar RAG chunks and returns
structured retrieval results.

FAISS index N must always correspond to DocumentStore
metadata entry N.
"""

from typing import Dict, List, Optional

import numpy as np

from app.core.logger import logger
from ai.config.ai_config import (
    FAISS_SCORE_THRESHOLD,
    FAISS_TOP_K,
)
from ai.embedding.embedding_model import generate_embedding
from ai.rag.doc_store import get_doc_store
from ai.rag.models import ChunkMetadata
from ai.rag.vector_store import get_vector_store


def register_chunk(
    metadata: ChunkMetadata,
) -> int:
    """
    Register chunk metadata in the persistent document store.

    Returns:
        FAISS-compatible index ID.

    The returned index corresponds to the position that the caller
    must use when adding the chunk's embedding to FAISS.
    """

    store = get_doc_store()

    index_id = store.add_chunk(metadata)

    logger.debug(
        "Registered RAG chunk index=%s chunk_id=%s "
        "document_id=%s",
        index_id,
        metadata.chunk_id,
        metadata.document_id,
    )

    return index_id


def register_chunks(
    metadata_list: List[ChunkMetadata],
) -> List[int]:
    """Register multiple chunk metadata records."""

    if not metadata_list:
        return []

    store = get_doc_store()

    index_ids = store.add_chunks(metadata_list)

    logger.debug(
        "Registered %d RAG chunks",
        len(index_ids),
    )

    return index_ids


def register_document(
    source: str,
    title: str,
    category: str,
    content: str,
    doc_id: Optional[int] = None,
    extra_metadata: Optional[Dict] = None,
) -> int:
    """
    Legacy document registration API.

    Kept temporarily for compatibility with the existing pipeline.

    New RAG ingestion should use register_chunk().
    """

    store = get_doc_store()

    index_id = len(store.documents)

    document = {
        "index_id": index_id,
        "doc_id": doc_id,
        "source": source,
        "title": title,
        "category": category,
        "content": content,
    }

    if extra_metadata:
        document.update(extra_metadata)

    store.documents.append(document)

    return index_id


def save_doc_store() -> None:
    """Persist the document store to disk."""

    get_doc_store().save()


def validate_store_alignment() -> bool:
    """
    Ensure FAISS vector count matches document-store count.
    """

    vector_store = get_vector_store()
    doc_store = get_doc_store()

    if vector_store.total_vectors != doc_store.total_documents:
        logger.warning(
            "FAISS/doc store mismatch: vectors=%s, docs=%s",
            vector_store.total_vectors,
            doc_store.total_documents,
        )

        return False

    logger.info(
        "FAISS/doc store aligned: vectors=%s, docs=%s",
        vector_store.total_vectors,
        doc_store.total_documents,
    )

    return True


def _embed_query(
    query: str | np.ndarray,
) -> np.ndarray:
    """
    Convert a natural-language query into an embedding.

    Supports:
        - str: generate the embedding internally
        - np.ndarray: use an already-generated embedding
    """

    if isinstance(query, str):
        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty")

        embedding = generate_embedding(query)

    elif isinstance(query, np.ndarray):
        embedding = query

    else:
        raise TypeError(
            "Query must be either a string or a numpy.ndarray"
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if embedding.ndim != 1:
        raise ValueError(
            f"Query embedding must be 1D, got shape={embedding.shape}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            "Query embedding contains NaN or infinite values"
        )

    return embedding


def retrieve_context(
    query: str | np.ndarray,
    top_k: int = FAISS_TOP_K,
    score_threshold: float = FAISS_SCORE_THRESHOLD,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant chunks for a natural-language query.

    The query is converted into an embedding before FAISS search.

    Returns:
        Structured metadata associated with each FAISS vector.
    """

    vector_store = get_vector_store()
    doc_store = get_doc_store()

    if vector_store.total_vectors == 0:
        logger.warning(
            "FAISS index is empty — no context retrieved"
        )
        return []

    if not validate_store_alignment():
        logger.warning(
            "Skipping retrieval due to FAISS/doc store mismatch"
        )
        return []

    try:
        query_embedding = _embed_query(query)

        scores, indices = vector_store.search(
            query_embedding,
            top_k=top_k,
        )

    except Exception as exc:
        logger.error(
            "Vector search failed: %s",
            exc,
        )
        return []

    results: List[Dict] = []

    for score, idx in zip(scores, indices):
        if idx < 0:
            continue

        if float(score) < score_threshold:
            logger.debug(
                "Rejected chunk idx=%s score=%.4f "
                "below threshold=%.4f",
                idx,
                score,
                score_threshold,
            )
            continue

        metadata = doc_store.get_document(
            int(idx)
        )

        if not metadata:
            continue

        result = {
            **metadata,
            "score": float(score),
        }

        results.append(result)

        logger.debug(
            "Retrieved chunk idx=%s score=%.4f "
            "chunk_id=%s",
            idx,
            score,
            metadata.get("chunk_id", ""),
        )

    return results