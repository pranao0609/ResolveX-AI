"""
ingestion.py — Canonical RAG document ingestion pipeline.

Pipeline:

Knowledge Base documents
    ↓
Clean
    ↓
Hash
    ↓
Validate
    ↓
Chunk
    ↓
Metadata
    ↓
Embedding
    ↓
FAISS + DocumentStore
    ↓
Persist
"""

from dataclasses import dataclass
from typing import List

from app.core.logger import logger
from ai.rag.chunker import DocumentChunker
from ai.rag.index_writer import RAGIndexWriter
from ai.rag.models import CanonicalDocument
from ai.rag.pipeline import RAGDocumentPipeline
from ai.rag.vector_store import get_vector_store
from ai.rag.doc_store import get_doc_store


@dataclass
class IngestionResult:
    """
    Result of a canonical RAG ingestion operation.
    """

    documents_received: int
    documents_valid: int
    documents_invalid: int
    chunks_created: int
    chunks_indexed: int

    @property
    def success(self) -> bool:
        return (
            self.documents_received > 0
            and self.documents_valid > 0
            and self.chunks_created > 0
            and self.chunks_indexed == self.chunks_created
        )


class RAGIngestionPipeline:
    """
    Orchestrates the complete canonical RAG ingestion pipeline.
    """

    def __init__(self):
        self.chunker = DocumentChunker()

    def ingest(
        self,
        documents: List[CanonicalDocument],
    ) -> IngestionResult:
        """
        Ingest canonical documents into the RAG system.

        Steps:
            1. Clean
            2. Hash
            3. Validate
            4. Chunk
            5. Embed
            6. Index
            7. Persist
        """

        if not documents:
            logger.info("No documents supplied for RAG ingestion")

            return IngestionResult(
                documents_received=0,
                documents_valid=0,
                documents_invalid=0,
                chunks_created=0,
                chunks_indexed=0,
            )

        documents_received = len(documents)

        logger.info(
            "Starting RAG ingestion for %d documents",
            documents_received,
        )

        # ---------------------------------------------------------
        # 1. Prepare documents
        # ---------------------------------------------------------
        prepared = RAGDocumentPipeline.prepare(documents)

        logger.info(
            "Document preparation completed: %d valid, %d invalid",
            prepared.valid_count,
            prepared.invalid_count,
        )

        if not prepared.valid_documents:
            logger.warning("No valid documents available after preparation")

            return IngestionResult(
                documents_received=documents_received,
                documents_valid=0,
                documents_invalid=prepared.invalid_count,
                chunks_created=0,
                chunks_indexed=0,
            )

        # ---------------------------------------------------------
        # 2. Chunk documents
        # ---------------------------------------------------------
        chunks = []

        for document in prepared.valid_documents:
            document_chunks = self.chunker.chunk(document)
            chunks.extend(document_chunks)

        logger.info(
            "Created %d chunks from %d valid documents",
            len(chunks),
            prepared.valid_count,
        )

        if not chunks:
            logger.warning("No chunks were produced from valid documents")

            return IngestionResult(
                documents_received=documents_received,
                documents_valid=prepared.valid_count,
                documents_invalid=prepared.invalid_count,
                chunks_created=0,
                chunks_indexed=0,
            )

        # ---------------------------------------------------------
        # 3. Embed + index + metadata registration
        # ---------------------------------------------------------
        index_ids = RAGIndexWriter.index_chunks(chunks)

        chunks_indexed = len(index_ids)

        # ---------------------------------------------------------
        # 4. Persist
        # ---------------------------------------------------------
        get_vector_store().save()
        get_doc_store().save()

        logger.info(
            "RAG ingestion completed: " "%d documents → %d chunks → %d indexed",
            prepared.valid_count,
            len(chunks),
            chunks_indexed,
        )

        return IngestionResult(
            documents_received=documents_received,
            documents_valid=prepared.valid_count,
            documents_invalid=prepared.invalid_count,
            chunks_created=len(chunks),
            chunks_indexed=chunks_indexed,
        )


def ingest_documents(
    documents: List[CanonicalDocument],
) -> IngestionResult:
    """
    Backward-compatible entry point for the canonical ingestion pipeline.

    The input must now contain CanonicalDocument objects.
    """

    pipeline = RAGIngestionPipeline()

    return pipeline.ingest(documents)
