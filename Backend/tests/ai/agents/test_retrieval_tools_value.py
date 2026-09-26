from unittest.mock import MagicMock, patch

import pytest

from ai.agents.retrieval_tools import (
    get_retrieval_tools,
    rerank_documents,
    search_knowledge_base,
    search_previous_tickets,
)

# ============================================================================
# Helpers
# ============================================================================


def make_candidate(
    index_id=1,
    score=0.91,
    retriever="hybrid",
):
    candidate = MagicMock()
    candidate.index_id = index_id
    candidate.score = score
    candidate.retriever = retriever
    return candidate


def make_document(
    title="VPN Troubleshooting",
    content="Restart the VPN client.",
    source="knowledge_base",
):
    document = MagicMock()
    document.model_dump.return_value = {
        "title": title,
        "content": content,
        "source": source,
    }
    return document


# ============================================================================
# search_knowledge_base()
# ============================================================================


def test_search_knowledge_base_hybrid_success():
    candidate = make_candidate(
        index_id=1,
        score=0.91,
        retriever="hybrid",
    )

    document = make_document()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [candidate]

    mock_doc_store = MagicMock()
    mock_doc_store.get_chunk.return_value = document

    with (
        patch(
            "ai.agents.retrieval_tools._hybrid_retriever",
            mock_retriever,
        ),
        patch(
            "ai.agents.retrieval_tools._doc_store",
            mock_doc_store,
        ),
    ):
        result = search_knowledge_base(
            "VPN connection fails",
            strategy="hybrid",
            top_k=5,
            candidate_k=20,
        )

    assert len(result) == 1

    assert result[0]["index_id"] == 1
    assert result[0]["score"] == 0.91
    assert result[0]["retriever"] == "hybrid"
    assert result[0]["title"] == "VPN Troubleshooting"
    assert result[0]["content"] == "Restart the VPN client."
    assert result[0]["source"] == "knowledge_base"

    mock_retriever.retrieve.assert_called_once_with(
        query="VPN connection fails",
        top_k=5,
        candidate_k=20,
    )

    mock_doc_store.get_chunk.assert_called_once_with(1)


def test_search_knowledge_base_hybrid_skips_missing_documents():
    candidate_one = make_candidate(
        index_id=1,
        score=0.91,
        retriever="hybrid",
    )

    candidate_two = make_candidate(
        index_id=2,
        score=0.80,
        retriever="hybrid",
    )

    document = make_document()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        candidate_one,
        candidate_two,
    ]

    mock_doc_store = MagicMock()
    mock_doc_store.get_chunk.side_effect = [
        document,
        None,
    ]

    with (
        patch(
            "ai.agents.retrieval_tools._hybrid_retriever",
            mock_retriever,
        ),
        patch(
            "ai.agents.retrieval_tools._doc_store",
            mock_doc_store,
        ),
    ):
        result = search_knowledge_base(
            "VPN connection fails",
            strategy="hybrid",
        )

    assert len(result) == 1
    assert result[0]["index_id"] == 1

    assert mock_doc_store.get_chunk.call_count == 2


def test_search_knowledge_base_empty_result():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch(
        "ai.agents.retrieval_tools._hybrid_retriever",
        mock_retriever,
    ):
        result = search_knowledge_base(
            "unknown problem",
            strategy="hybrid",
        )

    assert result == []

    mock_retriever.retrieve.assert_called_once()


def test_search_knowledge_base_empty_query():
    with pytest.raises(
        ValueError,
        match="Knowledge-base query must not be empty",
    ):
        search_knowledge_base(
            "",
            strategy="hybrid",
        )


def test_search_knowledge_base_whitespace_query():
    with pytest.raises(
        ValueError,
        match="Knowledge-base query must not be empty",
    ):
        search_knowledge_base(
            "   ",
            strategy="hybrid",
        )


def test_search_knowledge_base_invalid_query_type():
    with pytest.raises(
        TypeError,
        match="Knowledge-base query must be a string",
    ):
        search_knowledge_base(
            None,
            strategy="hybrid",
        )


def test_search_knowledge_base_unsupported_strategy():
    with pytest.raises(
        ValueError,
        match="Unsupported RETRIEVAL_STRATEGY",
    ):
        search_knowledge_base(
            "VPN connection fails",
            strategy="unsupported",
        )


def test_search_knowledge_base_zero_top_k():
    mock_retriever = MagicMock()

    with patch(
        "ai.agents.retrieval_tools._hybrid_retriever",
        mock_retriever,
    ):
        result = search_knowledge_base(
            "VPN connection fails",
            strategy="hybrid",
            top_k=0,
        )

    assert result == []
    mock_retriever.retrieve.assert_not_called()


def test_search_knowledge_base_negative_top_k():
    mock_retriever = MagicMock()

    with patch(
        "ai.agents.retrieval_tools._hybrid_retriever",
        mock_retriever,
    ):
        result = search_knowledge_base(
            "VPN connection fails",
            strategy="hybrid",
            top_k=-1,
        )

    assert result == []
    mock_retriever.retrieve.assert_not_called()


# ============================================================================
# BM25 strategy
# ============================================================================


def test_search_knowledge_base_bm25_success():
    bm25_results = [
        {
            "index_id": 10,
            "score": 3.5,
        },
        {
            "index_id": 20,
            "score": 2.8,
        },
    ]

    document_one = make_document(
        title="VPN Guide",
        content="Restart VPN.",
    )

    document_two = make_document(
        title="Network Guide",
        content="Check network settings.",
    )

    mock_retriever = MagicMock()
    mock_retriever.bm25_store.search.return_value = bm25_results

    mock_doc_store = MagicMock()
    mock_doc_store.get_chunk.side_effect = [
        document_one,
        document_two,
    ]

    with (
        patch(
            "ai.agents.retrieval_tools._hybrid_retriever",
            mock_retriever,
        ),
        patch(
            "ai.agents.retrieval_tools._doc_store",
            mock_doc_store,
        ),
    ):
        result = search_knowledge_base(
            "VPN connection fails",
            strategy="bm25",
            top_k=2,
        )

    assert len(result) == 2

    assert result[0]["index_id"] == 10
    assert result[0]["score"] == 3.5
    assert result[0]["retriever"] == "bm25"

    assert result[1]["index_id"] == 20
    assert result[1]["score"] == 2.8
    assert result[1]["retriever"] == "bm25"

    mock_retriever.bm25_store.search.assert_called_once_with(
        query="VPN connection fails",
        top_k=2,
    )


# ============================================================================
# Dense strategy
# ============================================================================


def test_search_knowledge_base_dense_success():
    dense_documents = [
        {
            "index_id": 5,
            "score": 0.88,
            "title": "VPN Guide",
            "content": "Restart the VPN client.",
            "source": "knowledge_base",
        }
    ]

    with patch(
        "ai.rag.retriever.retrieve_context",
        return_value=dense_documents,
    ) as mock_retrieve:
        result = search_knowledge_base(
            "VPN connection fails",
            strategy="dense",
            top_k=5,
        )

    assert len(result) == 1
    assert result[0]["index_id"] == 5
    assert result[0]["score"] == 0.88
    assert result[0]["retriever"] == "dense"

    mock_retrieve.assert_called_once_with(
        "VPN connection fails",
        top_k=5,
        score_threshold=-1.0,
    )


# ============================================================================
# search_previous_tickets()
# ============================================================================


def test_search_previous_tickets_returns_empty_result():
    result = search_previous_tickets(
        "VPN connection fails",
    )

    assert result == []


def test_search_previous_tickets_respects_current_contract():
    result = search_previous_tickets(
        "printer not working",
        top_k=10,
    )

    assert isinstance(result, list)
    assert result == []


def test_search_previous_tickets_empty_query():
    with pytest.raises(
        ValueError,
        match="Previous-ticket query must not be empty",
    ):
        search_previous_tickets("")


def test_search_previous_tickets_invalid_query_type():
    with pytest.raises(
        TypeError,
        match="Previous-ticket query must be a string",
    ):
        search_previous_tickets(None)


def test_search_previous_tickets_zero_top_k():
    result = search_previous_tickets(
        "VPN connection fails",
        top_k=0,
    )

    assert result == []


# ============================================================================
# rerank_documents()
# ============================================================================


def test_rerank_documents_success():
    documents = [
        {
            "index_id": 1,
            "score": 0.80,
            "retriever": "hybrid",
            "title": "VPN Guide",
            "content": "Restart VPN.",
            "source": "knowledge_base",
        },
        {
            "index_id": 2,
            "score": 0.70,
            "retriever": "hybrid",
            "title": "Network Guide",
            "content": "Check network settings.",
            "source": "knowledge_base",
        },
    ]

    reranked_candidate_one = make_candidate(
        index_id=2,
        score=0.95,
        retriever="hybrid",
    )

    reranked_candidate_two = make_candidate(
        index_id=1,
        score=0.85,
        retriever="hybrid",
    )

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        reranked_candidate_one,
        reranked_candidate_two,
    ]

    document_one = make_document(
        title="Network Guide",
        content="Check network settings.",
    )

    document_two = make_document(
        title="VPN Guide",
        content="Restart VPN.",
    )

    mock_doc_store = MagicMock()
    mock_doc_store.get_chunk.side_effect = [
        document_one,
        document_two,
    ]

    with (
        patch(
            "ai.agents.retrieval_tools.get_reranker",
            return_value=mock_reranker,
        ),
        patch(
            "ai.agents.retrieval_tools._doc_store",
            mock_doc_store,
        ),
    ):
        result = rerank_documents(
            "VPN connection fails",
            documents,
            top_k=2,
        )

    assert len(result) == 2

    assert result[0]["index_id"] == 2
    assert result[0]["score"] == 0.95

    assert result[1]["index_id"] == 1
    assert result[1]["score"] == 0.85

    mock_reranker.rerank.assert_called_once()

    call_kwargs = mock_reranker.rerank.call_args.kwargs

    assert call_kwargs["query"] == "VPN connection fails"
    assert call_kwargs["top_k"] == 2
    assert len(call_kwargs["candidates"]) == 2


def test_rerank_documents_empty_documents():
    mock_reranker = MagicMock()

    with patch(
        "ai.agents.retrieval_tools.get_reranker",
        return_value=mock_reranker,
    ):
        result = rerank_documents(
            "VPN connection fails",
            [],
            top_k=5,
        )

    assert result == []
    mock_reranker.rerank.assert_not_called()


def test_rerank_documents_invalid_document_is_skipped():
    documents = [
        {
            "title": "Invalid document",
            "content": "Missing index ID.",
        },
        {
            "index_id": 2,
            "score": "invalid",
            "title": "Invalid score",
            "content": "Invalid score.",
        },
    ]

    mock_reranker = MagicMock()

    with patch(
        "ai.agents.retrieval_tools.get_reranker",
        return_value=mock_reranker,
    ):
        result = rerank_documents(
            "VPN connection fails",
            documents,
            top_k=5,
        )

    assert result == []
    mock_reranker.rerank.assert_not_called()


def test_rerank_documents_zero_top_k():
    mock_reranker = MagicMock()

    documents = [
        {
            "index_id": 1,
            "score": 0.8,
        }
    ]

    with patch(
        "ai.agents.retrieval_tools.get_reranker",
        return_value=mock_reranker,
    ):
        result = rerank_documents(
            "VPN connection fails",
            documents,
            top_k=0,
        )

    assert result == []
    mock_reranker.rerank.assert_not_called()


def test_rerank_documents_empty_query():
    with pytest.raises(
        ValueError,
        match="Reranking query must not be empty",
    ):
        rerank_documents(
            "",
            [],
        )


def test_rerank_documents_invalid_query_type():
    with pytest.raises(
        TypeError,
        match="Reranking query must be a string",
    ):
        rerank_documents(
            None,
            [],
        )


def test_rerank_documents_reranker_failure_propagates():
    documents = [
        {
            "index_id": 1,
            "score": 0.8,
            "retriever": "hybrid",
        }
    ]

    candidate = make_candidate(
        index_id=1,
        score=0.8,
    )

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = RuntimeError("reranker unavailable")

    with (
        patch(
            "ai.agents.retrieval_tools.get_reranker",
            return_value=mock_reranker,
        ),
        pytest.raises(
            RuntimeError,
            match="reranker unavailable",
        ),
    ):
        # The candidate variable ensures this test remains explicit
        # about the expected reranker input contract.
        assert candidate.index_id == 1

        rerank_documents(
            "VPN connection fails",
            documents,
            top_k=5,
        )


# ============================================================================
# Tool Registry
# ============================================================================


def test_get_retrieval_tools_returns_expected_tools():
    tools = get_retrieval_tools()

    assert isinstance(tools, dict)

    assert set(tools.keys()) == {
        "search_knowledge_base",
        "search_previous_tickets",
        "rerank_documents",
    }

    assert tools["search_knowledge_base"] is search_knowledge_base
    assert tools["search_previous_tickets"] is search_previous_tickets
    assert tools["rerank_documents"] is rerank_documents
