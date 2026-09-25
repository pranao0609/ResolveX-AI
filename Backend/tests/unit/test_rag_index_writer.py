from datetime import datetime, timezone

import numpy as np

from ai.rag import doc_store, index_writer
from ai.rag.models import DocumentChunk


def make_chunk(index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"kb:42:chunk:{index}",
        document_id="kb:42",
        source_id="knowledge_base:42",
        chunk_index=index,
        title="Network Troubleshooting",
        content=f"Restart the router step {index}.",
        category="network",
        source_type="knowledge_base",
        created_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
        version="1.0",
        content_hash="a" * 64,
    )


class FakeVectorStore:
    def __init__(self):
        self.total_vectors = 0
        self.added = []

    def add(self, embeddings):
        self.added.append(embeddings.copy())
        self.total_vectors += len(embeddings)

    def rollback_to(self, vector_count):
        if vector_count < 0:
            raise ValueError("vector_count cannot be negative")

        if vector_count > self.total_vectors:
            raise ValueError(
                f"Cannot rollback to {vector_count}; "
                f"current count is {self.total_vectors}"
            )

        self.total_vectors = vector_count


class FakeDocumentStore:
    def __init__(self):
        self.documents = []

    def add_chunk(self, metadata):
        index_id = len(self.documents)

        self.documents.append(
            {
                "index_id": index_id,
                **metadata.model_dump(mode="json"),
            }
        )

        return index_id

    @property
    def total_chunks(self):
        return len(self.documents)


def test_index_chunks_registers_vectors_and_metadata(monkeypatch):
    vector_store = FakeVectorStore()
    document_store = FakeDocumentStore()

    monkeypatch.setattr(
        index_writer,
        "get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        index_writer,
        "get_doc_store",
        lambda: document_store,
    )

    embeddings = np.ones(
        (2, 384),
        dtype=np.float32,
    )

    monkeypatch.setattr(
        index_writer.RAGEmbedder,
        "embed",
        lambda chunks: embeddings,
    )

    chunks = [
        make_chunk(0),
        make_chunk(1),
    ]

    index_ids = index_writer.RAGIndexWriter.index_chunks(chunks)

    assert index_ids == [0, 1]

    assert vector_store.total_vectors == 2

    assert len(vector_store.added) == 1
    assert vector_store.added[0].shape == (2, 384)

    assert len(document_store.documents) == 2

    assert document_store.documents[0]["chunk_id"] == "kb:42:chunk:0"
    assert document_store.documents[1]["chunk_id"] == "kb:42:chunk:1"


def test_index_chunks_preserves_existing_index_offset(monkeypatch):
    vector_store = FakeVectorStore()
    vector_store.total_vectors = 5

    document_store = FakeDocumentStore()

    for index in range(5):
        document_store.documents.append(
            {
                "index_id": index,
            }
        )

    monkeypatch.setattr(
        index_writer,
        "get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        index_writer,
        "get_doc_store",
        lambda: document_store,
    )

    embeddings = np.ones(
        (2, 384),
        dtype=np.float32,
    )

    monkeypatch.setattr(
        index_writer.RAGEmbedder,
        "embed",
        lambda chunks: embeddings,
    )

    chunks = [
        make_chunk(0),
        make_chunk(1),
    ]

    index_ids = index_writer.RAGIndexWriter.index_chunks(chunks)

    assert index_ids == [5, 6]

    assert vector_store.total_vectors == 7

    assert document_store.documents[5]["chunk_id"] == "kb:42:chunk:0"
    assert document_store.documents[6]["chunk_id"] == "kb:42:chunk:1"


def test_index_chunks_empty_input(monkeypatch):
    called = False

    def fake_embed(chunks):
        nonlocal called
        called = True
        return np.empty((0, 384), dtype=np.float32)

    monkeypatch.setattr(
        index_writer.RAGEmbedder,
        "embed",
        fake_embed,
    )

    result = index_writer.RAGIndexWriter.index_chunks([])

    assert result == []
    assert called is False


def test_index_chunks_rejects_embedding_count_mismatch(monkeypatch):
    chunks = [
        make_chunk(0),
        make_chunk(1),
    ]

    embeddings = np.ones(
        (1, 384),
        dtype=np.float32,
    )

    monkeypatch.setattr(
        index_writer.RAGEmbedder,
        "embed",
        lambda chunks: embeddings,
    )

    try:
        index_writer.RAGIndexWriter.index_chunks(chunks)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Chunk/embedding count mismatch" in str(exc)

def test_index_chunks_rejects_preexisting_store_misalignment(
    monkeypatch,
):
    vector_store = FakeVectorStore()
    vector_store.total_vectors = 5

    document_store = FakeDocumentStore()

    for index in range(4):
        document_store.documents.append(
            {
                "index_id": index,
            }
        )

    monkeypatch.setattr(
        index_writer,
        "get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        index_writer,
        "get_doc_store",
        lambda: document_store,
    )

    chunks = [make_chunk(0)]

    try:
        index_writer.RAGIndexWriter.index_chunks(chunks)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "already misaligned" in str(exc)

    assert vector_store.total_vectors == 5
    assert document_store.total_chunks == 4


def test_index_chunks_rolls_back_metadata_on_registration_failure(
    monkeypatch,
):
    vector_store = FakeVectorStore()
    document_store = FakeDocumentStore()

    monkeypatch.setattr(
        index_writer,
        "get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        index_writer,
        "get_doc_store",
        lambda: document_store,
    )

    embeddings = np.ones(
        (2, 384),
        dtype=np.float32,
    )

    monkeypatch.setattr(
        index_writer.RAGEmbedder,
        "embed",
        lambda chunks: embeddings,
    )

    original_add_chunk = document_store.add_chunk

    call_count = 0

    def failing_add_chunk(metadata):
        nonlocal call_count

        call_count += 1

        if call_count == 2:
            raise RuntimeError("Simulated metadata failure")

        return original_add_chunk(metadata)

    document_store.add_chunk = failing_add_chunk

    chunks = [
        make_chunk(0),
        make_chunk(1),
    ]

    try:
        index_writer.RAGIndexWriter.index_chunks(chunks)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Simulated metadata failure" in str(exc)

    assert vector_store.total_vectors == 0
    assert document_store.total_chunks == 0