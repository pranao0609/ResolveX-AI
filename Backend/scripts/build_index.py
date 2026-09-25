"""
build_index.py — Canonical RAG index builder.

Builds the FAISS vector index and chunk metadata store from
Knowledge Base entries stored in PostgreSQL.

Usage:

    python scripts/build_index.py

    python scripts/build_index.py --no-reset
"""

import argparse
import os
import sys

# Ensure Backend directory is on Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from app.core.logger import logger
from app.database import SessionLocal, init_db

from ai.rag.doc_store import get_doc_store
from ai.rag.retriever import validate_store_alignment
from ai.rag.service import RAGIngestionService
from ai.rag.vector_store import get_vector_store


def build_index(reset: bool = True):
    """
    Build the complete RAG index from PostgreSQL KB data.

    Args:
        reset: Whether to rebuild the index from scratch.

    Returns:
        IngestionResult
    """

    logger.info(
        "Starting canonical ResolveX RAG index build"
    )

    init_db()

    vector_store = get_vector_store()
    doc_store = get_doc_store()

    if reset:
        logger.info(
            "Resetting existing FAISS index and document store"
        )

        vector_store.reset()
        doc_store.clear()

    # ---------------------------------------------------------
    # Database-backed canonical ingestion
    # ---------------------------------------------------------

    db = SessionLocal()

    try:
        service = RAGIngestionService(db)

        result = service.ingest_knowledge_base()

    except Exception:
        logger.exception(
            "Canonical RAG index build failed"
        )
        raise

    finally:
        db.close()

    # ---------------------------------------------------------
    # Final consistency check
    # ---------------------------------------------------------

    aligned = validate_store_alignment()

    if not aligned:
        logger.error(
            "RAG index build failed: "
            "FAISS and DocumentStore are misaligned"
        )

        raise RuntimeError(
            "FAISS and DocumentStore are misaligned"
        )

    logger.info(
        "Canonical RAG index build completed: "
        "documents=%d valid=%d invalid=%d chunks=%d indexed=%d",
        result.documents_received,
        result.documents_valid,
        result.documents_invalid,
        result.chunks_created,
        result.chunks_indexed,
    )

    logger.info(
        "Final RAG store state: %d vectors",
        vector_store.total_vectors,
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build ResolveX RAG index from PostgreSQL "
            "knowledge-base entries."
        )
    )

    parser.add_argument(
        "--no-reset",
        action="store_true",
        help=(
            "Keep the existing FAISS and document stores "
            "instead of rebuilding from scratch."
        ),
    )

    args = parser.parse_args()

    build_index(
        reset=not args.no_reset
    )