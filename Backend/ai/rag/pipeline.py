"""
pipeline.py — RAG document preparation pipeline.

Responsible for orchestrating:

    loader → cleaner → hasher → validator

This module does not perform chunking, embedding, or vector indexing.
Those are handled by later stages.
"""

from dataclasses import dataclass
from typing import List

from app.core.logger import logger

from ai.rag.cleaner import DocumentCleaner
from ai.rag.hasher import DocumentHasher
from ai.rag.models import CanonicalDocument
from ai.rag.validator import DocumentValidator


@dataclass
class PreparedDocuments:
    """Result of the document preparation pipeline."""

    valid_documents: List[CanonicalDocument]
    invalid_documents: List[CanonicalDocument]

    @property
    def valid_count(self) -> int:
        return len(self.valid_documents)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_documents)


class RAGDocumentPipeline:
    """
    Prepare canonical documents for downstream RAG processing.

    Processing order:

        clean
          ↓
        hash
          ↓
        validate
    """

    @staticmethod
    def prepare(
        documents: List[CanonicalDocument],
    ) -> PreparedDocuments:
        if not documents:
            logger.warning("RAG document pipeline received no documents")

            return PreparedDocuments(
                valid_documents=[],
                invalid_documents=[],
            )

        cleaned_documents = [DocumentCleaner.clean(document) for document in documents]

        hashed_documents = DocumentHasher.attach_hashes(cleaned_documents)

        valid_documents, invalid_documents = DocumentValidator.validate_all(
            hashed_documents
        )

        logger.info(
            "RAG document preparation completed: " "%d valid, %d invalid",
            len(valid_documents),
            len(invalid_documents),
        )

        return PreparedDocuments(
            valid_documents=valid_documents,
            invalid_documents=invalid_documents,
        )
