from datetime import datetime, timezone

import numpy as np
import pytest

import ai.rag.embedder as embedder_module
from ai.config.ai_config import EMBEDDING_DIMENSION
from ai.rag.embedder import RAGEmbedder
from ai.rag.models import DocumentChunk


def make_chunk(
    chunk_id="kb:42:chunk:0",
    title="Network Troubleshooting",
    category="network",
    content="Restart the router and check the connection.",
):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="kb:42",
        source_id="knowledge_base:42",
        chunk_index=0,
        title=title,
        content=content,
        category=category,
        source_type="knowledge_base",
        created_at=datetime(
            2026,
            9,
            24,
            tzinfo=timezone.utc,
        ),
        version="1.0",
        content_hash="a" * 64,
    )


def test_build_embedding_text_contains_title_category_and_content():
    chunk = make_chunk()

    text = RAGEmbedder.build_embedding_text(chunk)

    assert "Title: Network Troubleshooting" in text
    assert "Category: network" in text
    assert "Content: Restart the router and check the connection." in text


def test_build_embedding_text_without_category():
    chunk = make_chunk(category=None)

    text = RAGEmbedder.build_embedding_text(chunk)

    assert "Title: Network Troubleshooting" in text
    assert "Content: Restart the router and check the connection." in text
    assert "Category:" not in text


def test_build_embedding_texts_preserves_order():
    chunks = [
        make_chunk(
            chunk_id="kb:42:chunk:0",
            title="First",
        ),
        make_chunk(
            chunk_id="kb:42:chunk:1",
            title="Second",
        ),
        make_chunk(
            chunk_id="kb:42:chunk:2",
            title="Third",
        ),
    ]

    texts = RAGEmbedder.build_embedding_texts(chunks)

    assert len(texts) == 3
    assert texts[0].startswith("Title: First")
    assert texts[1].startswith("Title: Second")
    assert texts[2].startswith("Title: Third")


def test_empty_chunks_returns_empty_embedding_matrix():
    embeddings = RAGEmbedder.embed([])

    assert embeddings.shape == (
        0,
        EMBEDDING_DIMENSION,
    )

    assert embeddings.dtype == np.float32


def test_validate_embeddings_accepts_valid_embeddings():
    embeddings = np.zeros(
        (3, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    RAGEmbedder.validate_embeddings(
        embeddings,
        expected_count=3,
    )


def test_validate_embeddings_rejects_wrong_dimension():
    embeddings = np.zeros(
        (3, EMBEDDING_DIMENSION + 1),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="dimension mismatch",
    ):
        RAGEmbedder.validate_embeddings(
            embeddings,
            expected_count=3,
        )


def test_validate_embeddings_rejects_wrong_count():
    embeddings = np.zeros(
        (2, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="count mismatch",
    ):
        RAGEmbedder.validate_embeddings(
            embeddings,
            expected_count=3,
        )


def test_validate_embeddings_rejects_non_2d_array():
    embeddings = np.zeros(
        EMBEDDING_DIMENSION,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="2D",
    ):
        RAGEmbedder.validate_embeddings(
            embeddings,
            expected_count=1,
        )


def test_validate_embeddings_rejects_non_finite_values():
    embeddings = np.zeros(
        (1, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    embeddings[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        RAGEmbedder.validate_embeddings(
            embeddings,
            expected_count=1,
        )


def test_embed_uses_shared_embedding_service(monkeypatch):
    chunks = [
        make_chunk(
            chunk_id="kb:42:chunk:0",
        ),
        make_chunk(
            chunk_id="kb:42:chunk:1",
        ),
    ]

    expected_embeddings = np.ones(
        (2, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    calls = {}

    def fake_generate_batch_embeddings(texts):
        calls["texts"] = texts
        return expected_embeddings

    monkeypatch.setattr(
        embedder_module,
        "generate_batch_embeddings",
        fake_generate_batch_embeddings,
    )

    embeddings = RAGEmbedder.embed(chunks)

    assert np.array_equal(
        embeddings,
        expected_embeddings,
    )

    assert len(calls["texts"]) == 2
    assert "Title: Network Troubleshooting" in calls["texts"][0]


def test_embed_returns_float32(monkeypatch):
    chunks = [make_chunk()]

    embeddings = np.ones(
        (1, EMBEDDING_DIMENSION),
        dtype=np.float64,
    )

    monkeypatch.setattr(
        embedder_module,
        "generate_batch_embeddings",
        lambda texts: embeddings,
    )

    result = RAGEmbedder.embed(chunks)

    assert result.dtype == np.float32
