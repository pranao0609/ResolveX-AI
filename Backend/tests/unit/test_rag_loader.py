from datetime import datetime, timezone
from unittest.mock import MagicMock

from ai.rag.loader import KnowledgeBaseLoader


def make_kb_entry(
    entry_id=1,
    title="VPN Troubleshooting",
    content="Restart the VPN client and reconnect.",
    category="network",
    source="article",
):
    entry = MagicMock()

    entry.id = entry_id
    entry.title = title
    entry.content = content
    entry.category = category
    entry.source = source
    entry.created_at = datetime.now(timezone.utc)

    return entry


def test_load_all_converts_kb_entries_to_documents():
    db = MagicMock()

    entry = make_kb_entry()

    db.query.return_value.order_by.return_value.all.return_value = [
        entry
    ]

    loader = KnowledgeBaseLoader(db)

    documents = loader.load_all()

    assert len(documents) == 1

    document = documents[0]

    assert document.document_id == "kb:1"
    assert document.source_id == "article:1"
    assert document.title == "VPN Troubleshooting"
    assert document.content == "Restart the VPN client and reconnect."
    assert document.category == "network"
    assert document.source_type == "article"
    assert document.version == "1.0"


def test_load_all_preserves_database_order():
    db = MagicMock()

    entries = [
        make_kb_entry(entry_id=1, title="First"),
        make_kb_entry(entry_id=2, title="Second"),
        make_kb_entry(entry_id=3, title="Third"),
    ]

    db.query.return_value.order_by.return_value.all.return_value = entries

    loader = KnowledgeBaseLoader(db)

    documents = loader.load_all()

    assert [doc.document_id for doc in documents] == [
        "kb:1",
        "kb:2",
        "kb:3",
    ]


def test_load_all_returns_empty_list_when_database_is_empty():
    db = MagicMock()

    db.query.return_value.order_by.return_value.all.return_value = []

    loader = KnowledgeBaseLoader(db)

    documents = loader.load_all()

    assert documents == []


def test_loader_strips_title_and_content():
    db = MagicMock()

    entry = make_kb_entry(
        title="   VPN Troubleshooting   ",
        content="   Restart VPN.   ",
    )

    db.query.return_value.order_by.return_value.all.return_value = [entry]

    loader = KnowledgeBaseLoader(db)

    documents = loader.load_all()

    assert documents[0].title == "VPN Troubleshooting"
    assert documents[0].content == "Restart VPN."


def test_loader_handles_missing_source():
    db = MagicMock()

    entry = make_kb_entry(source=None)

    db.query.return_value.order_by.return_value.all.return_value = [entry]

    loader = KnowledgeBaseLoader(db)

    documents = loader.load_all()

    assert documents[0].source_type == "knowledge_base"
    assert documents[0].source_id == "knowledge_base:1"