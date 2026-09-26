"""
loader.py — Knowledge Base document loader for RAG ingestion.

Loads knowledge-base entries from PostgreSQL and converts them into
CanonicalDocument objects.

This module is intentionally responsible only for loading data.
Cleaning, validation, chunking, embedding, and indexing are handled
by later stages of the ingestion pipeline.
"""

from typing import List

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.kb_model import KnowledgeBaseEntry

from ai.rag.models import CanonicalDocument


class KnowledgeBaseLoader:
    """Load knowledge-base records from PostgreSQL."""

    def __init__(self, db: Session):
        self.db = db

    def load_all(self) -> List[CanonicalDocument]:
        """
        Load all knowledge-base entries and convert them into
        canonical RAG documents.

        Returns:
            List of CanonicalDocument objects.
        """

        entries = (
            self.db.query(KnowledgeBaseEntry)
            .order_by(KnowledgeBaseEntry.id.asc())
            .all()
        )

        if not entries:
            logger.warning("No knowledge-base entries found")
            return []

        documents: List[CanonicalDocument] = []

        for entry in entries:
            try:
                document = self._convert_entry(entry)
                documents.append(document)

            except Exception as exc:
                logger.error(
                    "Failed to convert KB entry id=%s: %s",
                    entry.id,
                    exc,
                )

        logger.info(
            "Loaded %d/%d knowledge-base entries for RAG ingestion",
            len(documents),
            len(entries),
        )

        return documents

    @staticmethod
    def _convert_entry(entry: KnowledgeBaseEntry) -> CanonicalDocument:
        """
        Convert a SQLAlchemy KB entry into a CanonicalDocument.
        """

        source_type = entry.source or "knowledge_base"

        return CanonicalDocument(
            document_id=f"kb:{entry.id}",
            source_id=f"{source_type}:{entry.id}",
            title=entry.title.strip(),
            content=entry.content.strip(),
            category=entry.category.strip() if entry.category else None,
            source_type=source_type,
            created_at=entry.created_at,
            version="1.0",
        )
