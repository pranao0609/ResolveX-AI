from datetime import datetime, timezone

import numpy as np

from ai.rag.ingestion import (
    RAGIngestionPipeline,
    ingest_documents,
)
from ai.rag.models import CanonicalDocument


def make_document(
    document_id="kb:1",
    title="Network Troubleshooting",
    content="Restart the router.\n\nCheck the network connection.",
):
    return CanonicalDocument(
        document_id=document_id,
        source_id=f"knowledge_base:{document_id}",
        title=title,
        content=content,
        category="network",
        source_type="knowledge_base",
        created_at=datetime(
            2026,
            9,
            24,
            tzinfo=timezone.utc,
        ),
        version="1.0",
    )


class FakeVectorStore:
    def __init__(self):
        self.total_vectors = 0
        self.saved = False

    def save(self):
        self.saved = True


class FakeDocumentStore:
    def __init__(self):
        self.documents = []
        self.saved = False

    def save(self):
        self.saved = True


def test_ingestion_pipeline_empty_documents():
    pipeline = RAGIngestionPipeline()

    result = pipeline.ingest([])

    assert result.documents_received == 0
    assert result.documents_valid == 0
    assert result.documents_invalid == 0
    assert result.chunks_created == 0
    assert result.chunks_indexed == 0
    assert result.success is False


def test_ingestion_pipeline_indexes_chunks(monkeypatch):
    vector_store = FakeVectorStore()
    document_store = FakeDocumentStore()

    monkeypatch.setattr(
        "ai.rag.ingestion.get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        "ai.rag.ingestion.get_doc_store",
        lambda: document_store,
    )

    monkeypatch.setattr(
        "ai.rag.ingestion.RAGIndexWriter.index_chunks",
        lambda chunks: list(range(len(chunks))),
    )

    document = make_document()

    result = RAGIngestionPipeline().ingest([document])

    assert result.documents_received == 1
    assert result.documents_valid == 1
    assert result.documents_invalid == 0
    assert result.chunks_created > 0
    assert result.chunks_indexed == result.chunks_created
    assert result.success is True

    assert vector_store.saved is True
    assert document_store.saved is True


def test_ingestion_pipeline_excludes_invalid_documents(monkeypatch):
    vector_store = FakeVectorStore()
    document_store = FakeDocumentStore()

    monkeypatch.setattr(
        "ai.rag.ingestion.get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        "ai.rag.ingestion.get_doc_store",
        lambda: document_store,
    )

    monkeypatch.setattr(
        "ai.rag.ingestion.RAGIndexWriter.index_chunks",
        lambda chunks: list(range(len(chunks))),
    )

    valid_document = make_document(
        document_id="kb:valid",
    )

    invalid_document = make_document(
        document_id="kb:invalid",
        title="   ",
        content="   ",
    )

    result = RAGIngestionPipeline().ingest(
        [
            valid_document,
            invalid_document,
        ]
    )

    assert result.documents_received == 2
    assert result.documents_valid == 1
    assert result.documents_invalid == 1
    assert result.chunks_created > 0
    assert result.chunks_indexed == result.chunks_created


def test_ingest_documents_uses_canonical_pipeline(monkeypatch):
    monkeypatch.setattr(
        RAGIngestionPipeline,
        "ingest",
        lambda self, documents: "pipeline-result",
    )

    result = ingest_documents(
        [
            make_document(),
        ]
    )

    assert result == "pipeline-result"
