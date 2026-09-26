from datetime import datetime, timezone

from ai.rag import doc_store, retriever
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
        content="Restart the router.",
        category="network",
        source_type="knowledge_base",
        created_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
        version="1.0",
        content_hash="a" * 64,
    )


def reset_doc_store(tmp_path, monkeypatch):
    docstore_path = tmp_path / "docstore.json"

    monkeypatch.setattr(
        doc_store,
        "FAISS_DOCSTORE_PATH",
        str(docstore_path),
    )

    monkeypatch.setattr(
        doc_store,
        "_doc_store",
        None,
    )

    return docstore_path


def test_register_chunk_returns_store_index(tmp_path, monkeypatch):
    reset_doc_store(tmp_path, monkeypatch)

    metadata = make_metadata()

    index_id = retriever.register_chunk(metadata)

    assert index_id == 0

    store = retriever.get_doc_store()

    assert store.total_chunks == 1

    stored = store.get_chunk(index_id)

    assert stored is not None
    assert stored.chunk_id == metadata.chunk_id
    assert stored.document_id == metadata.document_id
    assert stored.content == metadata.content


def test_register_chunks_returns_sequential_indices(tmp_path, monkeypatch):
    reset_doc_store(tmp_path, monkeypatch)

    metadata_list = [
        make_metadata(
            chunk_id=f"kb:42:chunk:{index}",
            chunk_index=index,
        )
        for index in range(3)
    ]

    indices = retriever.register_chunks(metadata_list)

    assert indices == [0, 1, 2]

    store = retriever.get_doc_store()

    assert store.total_chunks == 3

    for index in range(3):
        stored = store.get_chunk(index)

        assert stored is not None
        assert stored.chunk_index == index


def test_register_chunks_handles_empty_input(tmp_path, monkeypatch):
    reset_doc_store(tmp_path, monkeypatch)

    assert retriever.register_chunks([]) == []

    store = retriever.get_doc_store()

    assert store.total_chunks == 0
