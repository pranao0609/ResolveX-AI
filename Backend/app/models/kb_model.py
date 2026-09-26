"""
kb_model.py — SQLAlchemy ORM model for the Knowledge Base.
Stores past resolved tickets and curated articles used for RAG retrieval.
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base


class KnowledgeBaseEntry(Base):
    """A resolved ticket or knowledge article used for RAG context."""

    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)  # the solution/article body
    category = Column(String(100), nullable=True)
    source = Column(
        String(100),
        nullable=False,
        default="knowledge_base",
    )  # "ticket" | "article" | "manual"
    relevance_score = Column(Float, nullable=True)  # cached similarity score

    # FAISS vector reference
    faiss_index_id = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<KBEntry id={self.id} title='{self.title}'>"
