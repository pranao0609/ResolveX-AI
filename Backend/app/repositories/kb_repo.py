"""
kb_repo.py — Repository layer for Knowledge Base CRUD operations.
"""

from sqlalchemy.orm import Session
from typing import Optional, List

from app.models.kb_model import KnowledgeBaseEntry


class KBRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, entry: KnowledgeBaseEntry) -> KnowledgeBaseEntry:
        """Add a new KB entry."""
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def get_by_id(self, entry_id: int) -> Optional[KnowledgeBaseEntry]:
        return (
            self.db.query(KnowledgeBaseEntry)
            .filter(KnowledgeBaseEntry.id == entry_id)
            .first()
        )

    def list_by_category(self, category: str) -> List[KnowledgeBaseEntry]:
        return (
            self.db.query(KnowledgeBaseEntry)
            .filter(KnowledgeBaseEntry.category == category)
            .all()
        )

    def list_all(self, skip: int = 0, limit: int = 100) -> List[KnowledgeBaseEntry]:
        return self.db.query(KnowledgeBaseEntry).offset(skip).limit(limit).all()

    def count(self) -> int:
        from sqlalchemy import func

        return self.db.query(func.count(KnowledgeBaseEntry.id)).scalar()
