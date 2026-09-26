from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.models.kb_model import KnowledgeBaseEntry


def test_kb_entry_default_source():
    engine = create_engine("sqlite:///:memory:")

    KnowledgeBaseEntry.__table__.create(engine)

    with Session(engine) as session:
        entry = KnowledgeBaseEntry(
            title="Test Knowledge Article",
            content="Test knowledge content.",
            category="software",
        )

        session.add(entry)
        session.flush()

        assert entry.source == "knowledge_base"


def test_kb_entry_explicit_source_is_preserved():
    entry = KnowledgeBaseEntry(
        title="Test Manual Article",
        content="Test manual content.",
        category="software",
        source="manual",
    )

    assert entry.source == "manual"
