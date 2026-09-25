from datetime import datetime, timezone

import pytest

from ai.rag.metadata import MetadataBuilder
from ai.rag.models import ChunkMetadata, DocumentChunk


def make_chunk(
    chunk_id="kb:42:chunk:0",
    chunk_index=0,
    content_hash="a" * 64,
):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="kb:42",
        source_id="knowledge_base:42",
        chunk_index=chunk_index,
        title="Network Troubleshooting",
        content="Restart the router and check the connection.",
        category="network",
        source_type="knowledge_base",
        created_at=datetime(
            2026,
            9,
            24,
            tzinfo=timezone.utc,
        ),
        version="1.0",
        content_hash=content_hash,
    )


def test_build_creates_chunk_metadata():
    chunk = make_chunk()

    metadata = MetadataBuilder.build(chunk)

    assert isinstance(metadata, ChunkMetadata)


def test_metadata_preserves_chunk_identity():
    chunk = make_chunk()

    metadata = MetadataBuilder.build(chunk)

    assert metadata.chunk_id == chunk.chunk_id
    assert metadata.document_id == chunk.document_id
    assert metadata.source_id == chunk.source_id
    assert metadata.chunk_index == chunk.chunk_index


def test_metadata_preserves_document_information():
    chunk = make_chunk()

    metadata = MetadataBuilder.build(chunk)

    assert metadata.title == chunk.title
    assert metadata.category == chunk.category
    assert metadata.source_type == chunk.source_type
    assert metadata.created_at == chunk.created_at
    assert metadata.version == chunk.version


def test_metadata_preserves_content_information():
    chunk = make_chunk()

    metadata = MetadataBuilder.build(chunk)

    assert metadata.content == chunk.content
    assert metadata.content_hash == chunk.content_hash


def test_missing_content_hash_is_rejected():
    chunk = make_chunk(
        content_hash=None,
    )

    with pytest.raises(
        ValueError,
        match="content_hash",
    ):
        MetadataBuilder.build(chunk)


def test_build_many_creates_metadata_for_all_chunks():
    chunks = [
        make_chunk(
            chunk_id="kb:42:chunk:0",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="kb:42:chunk:1",
            chunk_index=1,
        ),
        make_chunk(
            chunk_id="kb:42:chunk:2",
            chunk_index=2,
        ),
    ]

    metadata_list = MetadataBuilder.build_many(
        chunks
    )

    assert len(metadata_list) == 3

    assert [
        metadata.chunk_id
        for metadata in metadata_list
    ] == [
        "kb:42:chunk:0",
        "kb:42:chunk:1",
        "kb:42:chunk:2",
    ]


def test_build_many_preserves_order():
    chunks = [
        make_chunk(
            chunk_id="kb:42:chunk:2",
            chunk_index=2,
        ),
        make_chunk(
            chunk_id="kb:42:chunk:0",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="kb:42:chunk:1",
            chunk_index=1,
        ),
    ]

    metadata_list = MetadataBuilder.build_many(
        chunks
    )

    assert [
        metadata.chunk_id
        for metadata in metadata_list
    ] == [
        "kb:42:chunk:2",
        "kb:42:chunk:0",
        "kb:42:chunk:1",
    ]