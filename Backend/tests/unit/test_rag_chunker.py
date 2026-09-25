from datetime import datetime, timezone

import pytest

from ai.rag.chunker import DocumentChunker
from ai.rag.models import CanonicalDocument


def make_document(content):
    return CanonicalDocument(
        document_id="kb:42",
        source_id="knowledge_base:42",
        title="Network Troubleshooting",
        content=content,
        category="network",
        source_type="knowledge_base",
        created_at=datetime.now(timezone.utc),
        version="1.0",
        content_hash="a" * 64,
    )


def test_short_document_produces_one_chunk():
    document = make_document(
        "Restart the router and check the network connection."
    )

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=20,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content


def test_multiple_paragraphs_are_preserved():
    document = make_document(
        "First paragraph contains network information.\n\n"
        "Second paragraph contains troubleshooting steps.\n\n"
        "Third paragraph contains resolution instructions."
    )

    chunker = DocumentChunker(
        chunk_size=70,
        chunk_overlap=10,
        min_chunk_size=20,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2

    combined = "\n\n".join(
        chunk.content for chunk in chunks
    )

    assert "First paragraph" in combined
    assert "Second paragraph" in combined
    assert "Third paragraph" in combined


def test_long_sentence_is_split():
    content = "A" * 250

    document = make_document(content)

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=20,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 3

    for chunk in chunks:
        assert len(chunk.content) <= 100


def test_chunk_ids_are_deterministic():
    document = make_document(
        "This is a test document with several words."
    )

    chunker = DocumentChunker(
        chunk_size=30,
        chunk_overlap=5,
        min_chunk_size=10,
    )

    chunks = chunker.chunk(document)

    assert chunks[0].chunk_id == "kb:42:chunk:0"

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == (
            f"kb:42:chunk:{index}"
        )


def test_chunk_indices_are_sequential():
    document = make_document(
        "First paragraph.\n\n"
        "Second paragraph.\n\n"
        "Third paragraph.\n\n"
        "Fourth paragraph."
    )

    chunker = DocumentChunker(
        chunk_size=30,
        chunk_overlap=5,
        min_chunk_size=10,
    )

    chunks = chunker.chunk(document)

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == list(range(len(chunks)))


def test_document_metadata_is_preserved():
    document = make_document(
        "Network troubleshooting information."
    )

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=20,
    )

    chunks = chunker.chunk(document)

    chunk = chunks[0]

    assert chunk.document_id == document.document_id
    assert chunk.source_id == document.source_id
    assert chunk.title == document.title
    assert chunk.category == document.category
    assert chunk.source_type == document.source_type
    assert chunk.created_at == document.created_at
    assert chunk.version == document.version
    assert chunk.content_hash == document.content_hash


def test_empty_content_produces_no_chunks():
    document = make_document(" ")

    # CanonicalDocument permits whitespace, while the chunker
    # should simply have nothing useful to chunk.
    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=20,
    )

    chunks = chunker.chunk(document)

    assert chunks == []


def test_invalid_chunk_configuration_is_rejected():
    with pytest.raises(ValueError):
        DocumentChunker(
            chunk_size=0,
            chunk_overlap=0,
            min_chunk_size=10,
        )

    with pytest.raises(ValueError):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=100,
            min_chunk_size=10,
        )

    with pytest.raises(ValueError):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=101,
        )


def test_document_shorter_than_minimum_is_not_discarded():
    document = make_document(
        "Short but valid content."
    )

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=50,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == "Short but valid content."


def test_chunker_does_not_modify_original_document():
    original_content = (
        "First paragraph.\n\n"
        "Second paragraph."
    )

    document = make_document(original_content)

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=20,
    )

    chunker.chunk(document)

    assert document.content == original_content