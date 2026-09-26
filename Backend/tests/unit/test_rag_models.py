from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ai.rag.models import (
    CanonicalDocument,
    DocumentChunk,
    ChunkMetadata,
)


def test_canonical_document_creation():
    document = CanonicalDocument(
        document_id="kb-001",
        source_id="article-001",
        title="VPN Troubleshooting",
        content="Restart the VPN client and reconnect.",
        category="network",
    )

    assert document.document_id == "kb-001"
    assert document.source_id == "article-001"
    assert document.title == "VPN Troubleshooting"
    assert document.category == "network"
    assert document.version == "1.0"
    assert document.created_at.tzinfo is not None


def test_canonical_document_requires_content():
    with pytest.raises(ValidationError):
        CanonicalDocument(
            document_id="kb-001",
            source_id="article-001",
            title="VPN Troubleshooting",
            content="",
        )


def test_document_chunk_creation():
    created_at = datetime.now(timezone.utc)

    chunk = DocumentChunk(
        chunk_id="kb-001-chunk-000",
        document_id="kb-001",
        source_id="article-001",
        chunk_index=0,
        title="VPN Troubleshooting",
        content="Restart the VPN client.",
        category="network",
        created_at=created_at,
        version="1.0",
        content_hash="abc123",
    )

    assert chunk.chunk_id == "kb-001-chunk-000"
    assert chunk.document_id == "kb-001"
    assert chunk.chunk_index == 0
    assert chunk.content_hash == "abc123"


def test_chunk_index_cannot_be_negative():
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="kb-001-chunk-000",
            document_id="kb-001",
            source_id="article-001",
            chunk_index=-1,
            title="VPN Troubleshooting",
            content="Restart the VPN client.",
            created_at=datetime.now(timezone.utc),
        )


def test_chunk_metadata_creation():
    metadata = ChunkMetadata(
        chunk_id="kb-001-chunk-000",
        document_id="kb-001",
        source_id="article-001",
        chunk_index=0,
        title="VPN Troubleshooting",
        category="network",
        source_type="knowledge_base",
        created_at=datetime.now(timezone.utc),
        version="1.0",
        content_hash="abc123",
        content="Restart the VPN client.",
    )

    assert metadata.chunk_id == "kb-001-chunk-000"
    assert metadata.document_id == "kb-001"
    assert metadata.content_hash == "abc123"
