"""
test_retriever.py — Unit tests for the RAG retrieval layer.

Mocks FAISS vector store and doc store so no real embeddings or
disk I/O are required. Tests the retriever's filtering, alignment
validation, and result-building logic.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ai.rag.retriever import retrieve_context, validate_store_alignment

# ── Shared fake document fixture ─────────────────────────────────────────────

FAKE_DOC = {
    "index_id": 0,
    "doc_id": 42,
    "source": "kb",
    "title": "VPN Troubleshooting Guide",
    "category": "network",
    "content": "Restart your VPN client and check firewall settings.",
}


def _make_fake_vector_store(total=1):
    vs = MagicMock()
    vs.total_vectors = total
    vs.search.return_value = (np.array([0.85]), np.array([0]))
    return vs


def _make_fake_doc_store(total=1):
    ds = MagicMock()
    ds.total_documents = total
    ds.get_document.return_value = FAKE_DOC
    return ds


class TestRetrieveContext:
    """Core retrieval behaviour tests."""

    def test_returns_results_when_store_aligned(self):
        query_embedding = np.random.rand(384).astype("float32")

        with (
            patch(
                "ai.rag.retriever.get_vector_store",
                return_value=_make_fake_vector_store(1),
            ),
            patch(
                "ai.rag.retriever.get_doc_store", return_value=_make_fake_doc_store(1)
            ),
        ):
            results = retrieve_context(query_embedding, top_k=1, score_threshold=0.3)

        assert len(results) == 1
        assert results[0]["title"] == "VPN Troubleshooting Guide"
        assert results[0]["score"] == pytest.approx(0.85, abs=1e-3)

    def test_empty_index_returns_empty_list(self):
        query_embedding = np.random.rand(384).astype("float32")
        empty_vs = _make_fake_vector_store(0)

        with (
            patch("ai.rag.retriever.get_vector_store", return_value=empty_vs),
            patch(
                "ai.rag.retriever.get_doc_store", return_value=_make_fake_doc_store(0)
            ),
        ):
            results = retrieve_context(query_embedding)

        assert results == []

    def test_score_below_threshold_filtered_out(self):
        """Documents scoring below the threshold must not appear in results."""
        query_embedding = np.random.rand(384).astype("float32")
        vs = _make_fake_vector_store(1)
        vs.search.return_value = (np.array([0.10]), np.array([0]))  # very low score

        with (
            patch("ai.rag.retriever.get_vector_store", return_value=vs),
            patch(
                "ai.rag.retriever.get_doc_store", return_value=_make_fake_doc_store(1)
            ),
        ):
            results = retrieve_context(query_embedding, score_threshold=0.50)

        assert results == []

    def test_negative_faiss_index_skipped(self):
        """FAISS returns -1 for empty slots; they must be ignored."""
        query_embedding = np.random.rand(384).astype("float32")
        vs = _make_fake_vector_store(1)
        vs.search.return_value = (np.array([0.90]), np.array([-1]))  # invalid index

        with (
            patch("ai.rag.retriever.get_vector_store", return_value=vs),
            patch(
                "ai.rag.retriever.get_doc_store", return_value=_make_fake_doc_store(1)
            ),
        ):
            results = retrieve_context(query_embedding, score_threshold=0.0)

        assert results == []

    def test_result_contains_expected_keys(self):
        query_embedding = np.random.rand(384).astype("float32")

        with (
            patch(
                "ai.rag.retriever.get_vector_store",
                return_value=_make_fake_vector_store(1),
            ),
            patch(
                "ai.rag.retriever.get_doc_store", return_value=_make_fake_doc_store(1)
            ),
        ):
            results = retrieve_context(query_embedding, top_k=1, score_threshold=0.0)

        assert len(results) == 1
        result = results[0]
        for key in ("source", "title", "category", "content", "score"):
            assert key in result, f"Expected key '{key}' missing from result"

    def test_score_is_float_type(self):
        query_embedding = np.random.rand(384).astype("float32")

        with (
            patch(
                "ai.rag.retriever.get_vector_store",
                return_value=_make_fake_vector_store(1),
            ),
            patch(
                "ai.rag.retriever.get_doc_store", return_value=_make_fake_doc_store(1)
            ),
        ):
            results = retrieve_context(query_embedding, top_k=1, score_threshold=0.0)

        assert isinstance(results[0]["score"], float)

    def test_vector_search_exception_returns_empty(self):
        """If FAISS search raises, the function must return [] not crash."""
        query_embedding = np.random.rand(384).astype("float32")
        vs = _make_fake_vector_store(1)
        vs.search.side_effect = RuntimeError("FAISS internal error")

        with (
            patch("ai.rag.retriever.get_vector_store", return_value=vs),
            patch(
                "ai.rag.retriever.get_doc_store", return_value=_make_fake_doc_store(1)
            ),
        ):
            results = retrieve_context(query_embedding)

        assert results == []


class TestValidateStoreAlignment:
    """Tests for the FAISS / doc store alignment check."""

    def test_aligned_stores_return_true(self):
        vs = _make_fake_vector_store(5)
        ds = _make_fake_doc_store(5)

        with (
            patch("ai.rag.retriever.get_vector_store", return_value=vs),
            patch("ai.rag.retriever.get_doc_store", return_value=ds),
        ):
            assert validate_store_alignment() is True

    def test_misaligned_stores_return_false(self):
        vs = _make_fake_vector_store(5)
        ds = _make_fake_doc_store(3)  # mismatch

        with (
            patch("ai.rag.retriever.get_vector_store", return_value=vs),
            patch("ai.rag.retriever.get_doc_store", return_value=ds),
        ):
            assert validate_store_alignment() is False

    def test_both_empty_stores_are_aligned(self):
        vs = _make_fake_vector_store(0)
        ds = _make_fake_doc_store(0)

        with (
            patch("ai.rag.retriever.get_vector_store", return_value=vs),
            patch("ai.rag.retriever.get_doc_store", return_value=ds),
        ):
            assert validate_store_alignment() is True
