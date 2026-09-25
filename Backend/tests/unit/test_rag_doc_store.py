from datetime import datetime, timezone

from ai.rag.doc_store import DocumentStore
from ai.rag.models import ChunkMetadata


def make_metadata(
    chunk_id="kb:42:chunk:0",
    chunk_index=0,
):
    return ChunkMetadata(
        chunk_id=chunk_id,
        document_id="kb:42",
        source_id="knowledge_base:42",
        chunk_index=chunk_index,
        title="Network Troubleshooting",
        category="network",
        source_type="knowledge_base",
        created_at=datetime(
            2026,
            9,
            24,
            tzinfo=timezone.utc,
        ),
        version="1.0",
        content_hash="a" * 64,
        content="Restart the router.",
    )


def make_empty_store(tmp_path, monkeypatch):
    """
    Create an isolated DocumentStore for testing.

    Tests must never read or modify the real FAISS docstore.
    """

    docstore_path = tmp_path / "docstore.json"

    monkeypatch.setattr(
        "ai.rag.doc_store.FAISS_DOCSTORE_PATH",
        str(docstore_path),
    )

    return DocumentStore()


def test_add_chunk_returns_sequential_index(
    tmp_path,
    monkeypatch,
):
    store = make_empty_store(
        tmp_path,
        monkeypatch,
    )

    first_index = store.add_chunk(
        make_metadata(
            chunk_id="kb:42:chunk:0",
            chunk_index=0,
        )
    )

    second_index = store.add_chunk(
        make_metadata(
            chunk_id="kb:42:chunk:1",
            chunk_index=1,
        )
    )

    assert first_index == 0
    assert second_index == 1
    assert store.total_chunks == 2


def test_get_chunk_returns_chunk_metadata(
    tmp_path,
    monkeypatch,
):
    store = make_empty_store(
        tmp_path,
        monkeypatch,
    )

    metadata = make_metadata()

    index_id = store.add_chunk(metadata)

    retrieved = store.get_chunk(index_id)

    assert retrieved is not None
    assert retrieved.chunk_id == metadata.chunk_id
    assert retrieved.document_id == metadata.document_id
    assert retrieved.source_id == metadata.source_id
    assert retrieved.chunk_index == metadata.chunk_index
    assert retrieved.title == metadata.title
    assert retrieved.category == metadata.category
    assert retrieved.source_type == metadata.source_type
    assert retrieved.version == metadata.version
    assert retrieved.content_hash == metadata.content_hash
    assert retrieved.content == metadata.content


def test_get_document_preserves_index_id(
    tmp_path,
    monkeypatch,
):
    store = make_empty_store(
        tmp_path,
        monkeypatch,
    )

    index_id = store.add_chunk(make_metadata())

    document = store.get_document(index_id)

    assert document is not None
    assert document["index_id"] == index_id
    assert document["chunk_id"] == "kb:42:chunk:0"


def test_get_chunk_invalid_index_returns_none(
    tmp_path,
    monkeypatch,
):
    store = make_empty_store(
        tmp_path,
        monkeypatch,
    )

    assert store.get_chunk(0) is None
    assert store.get_chunk(-1) is None


def test_add_chunks_returns_sequential_indices(
    tmp_path,
    monkeypatch,
):
    store = make_empty_store(
        tmp_path,
        monkeypatch,
    )

    metadata_list = [
        make_metadata(
            chunk_id="kb:42:chunk:0",
            chunk_index=0,
        ),
        make_metadata(
            chunk_id="kb:42:chunk:1",
            chunk_index=1,
        ),
        make_metadata(
            chunk_id="kb:42:chunk:2",
            chunk_index=2,
        ),
    ]

    indices = store.add_chunks(metadata_list)

    assert indices == [0, 1, 2]
    assert store.total_chunks == 3


def test_clear_removes_all_chunks(
    tmp_path,
    monkeypatch,
):
    store = make_empty_store(
        tmp_path,
        monkeypatch,
    )

    store.add_chunk(make_metadata())

    assert store.total_chunks == 1

    store.clear()

    assert store.total_chunks == 0
    assert store.get_document(0) is None


def test_save_and_reload_preserves_metadata(
    tmp_path,
    monkeypatch,
):
    docstore_path = tmp_path / "docstore.json"

    monkeypatch.setattr(
        "ai.rag.doc_store.FAISS_DOCSTORE_PATH",
        str(docstore_path),
    )

    store = DocumentStore()

    metadata = make_metadata()

    index_id = store.add_chunk(metadata)

    store.save()

    assert docstore_path.exists()

    reloaded_store = DocumentStore()

    retrieved = reloaded_store.get_chunk(index_id)

    assert retrieved is not None
    assert retrieved.chunk_id == metadata.chunk_id
    assert retrieved.document_id == metadata.document_id
    assert retrieved.content_hash == metadata.content_hash
    assert retrieved.content == metadata.content


def test_saved_json_contains_index_id_and_chunk_metadata(
    tmp_path,
    monkeypatch,
):
    docstore_path = tmp_path / "docstore.json"

    monkeypatch.setattr(
        "ai.rag.doc_store.FAISS_DOCSTORE_PATH",
        str(docstore_path),
    )

    store = DocumentStore()

    store.add_chunk(make_metadata())
    store.save()

    with open(
        docstore_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = __import__("json").load(file)

    assert len(data) == 1

    entry = data[0]

    assert entry["index_id"] == 0
    assert entry["chunk_id"] == "kb:42:chunk:0"
    assert entry["document_id"] == "kb:42"
    assert entry["source_id"] == "knowledge_base:42"
    assert entry["content_hash"] == "a" * 64