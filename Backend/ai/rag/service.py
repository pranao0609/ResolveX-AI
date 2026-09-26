from sqlalchemy.orm import Session

from app.core.logger import logger
from ai.rag.ingestion import IngestionResult, RAGIngestionPipeline
from ai.rag.loader import KnowledgeBaseLoader


class RAGIngestionService:
    """
    Database-backed entry point for canonical RAG ingestion.

    Responsibility:
        PostgreSQL Knowledge Base
            ↓
        KnowledgeBaseLoader
            ↓
        RAGIngestionPipeline
    """

    def __init__(self, db: Session):
        self.db = db
        self.loader = KnowledgeBaseLoader(db)
        self.pipeline = RAGIngestionPipeline()

    def ingest_knowledge_base(self) -> IngestionResult:
        """
        Load all knowledge-base entries from PostgreSQL and
        ingest them through the canonical RAG pipeline.
        """

        logger.info("Starting database-backed RAG knowledge-base ingestion")

        documents = self.loader.load_all()

        if not documents:
            logger.warning("Knowledge-base ingestion aborted: no documents loaded")

            return IngestionResult(
                documents_received=0,
                documents_valid=0,
                documents_invalid=0,
                chunks_created=0,
                chunks_indexed=0,
            )

        result = self.pipeline.ingest(documents)

        logger.info(
            "Database-backed RAG ingestion completed: "
            "received=%d valid=%d invalid=%d chunks=%d indexed=%d",
            result.documents_received,
            result.documents_valid,
            result.documents_invalid,
            result.chunks_created,
            result.chunks_indexed,
        )

        return result
