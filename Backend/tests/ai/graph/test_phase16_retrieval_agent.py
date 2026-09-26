import pytest
from unittest.mock import patch
from ai.graph.nodes import retrieval_agent


def base_state():
    return {
        "ticket_id": 101,
        "cleaned_ticket": "VPN connection fails after login.",
        "request_id": "request-test",
        "graph_run_id": "graph-test",
        "fallback_used": False,
        "errors": [],
        "warnings": [],
        "metadata": {},
    }


def make_document(
    index_id: int,
    score: float,
    *,
    title: str = "VPN Troubleshooting",
    content: str = "Restart the VPN client.",
    source: str = "knowledge_base",
    retriever: str = "hybrid",
):
    return {
        "index_id": index_id,
        "score": score,
        "retriever": retriever,
        "title": title,
        "content": content,
        "source": source,
    }


def identity_rerank(query, documents, top_k=5):
    """Deterministic reranker mock for graph-agent tests."""
    return documents[:top_k]


def test_retrieval_agent_hybrid_success():
    documents = [
        make_document(
            index_id=1,
            score=0.91,
        )
    ]

    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=documents,
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            side_effect=identity_rerank,
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_STRATEGY",
            "hybrid",
        ),
    ):
        result = retrieval_agent(base_state())

    assert result["retrieved_documents"]
    assert result["retrieved_documents"][0]["index_id"] == 1
    assert result["retrieved_documents"][0]["score"] == 0.91

    metadata = result["retrieval_metadata"]

    assert metadata["strategy"] == "hybrid"
    assert metadata["status"] == "success"
    assert metadata["candidate_count"] == 1
    assert metadata["result_count"] == 1
    assert metadata["top_score"] == 0.91
    assert metadata["empty_result"] is False
    assert metadata["fallback_used"] is False


def test_retrieval_agent_empty_query():
    state = base_state()
    state["cleaned_ticket"] = ""

    result = retrieval_agent(state)

    assert result["retrieved_context"] == ""
    assert result["retrieved_documents"] == []

    metadata = result["retrieval_metadata"]

    assert metadata["status"] == "empty"
    assert metadata["result_count"] == 0
    assert metadata["candidate_count"] == 0
    assert metadata["empty_result"] is True

    assert "retrieval_agent: empty query" in result["warnings"]


def test_retrieval_agent_zero_results():
    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            side_effect=identity_rerank,
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_STRATEGY",
            "hybrid",
        ),
    ):
        result = retrieval_agent(base_state())

    assert result["retrieved_context"] == ""
    assert result["retrieved_documents"] == []

    metadata = result["retrieval_metadata"]

    assert metadata["status"] == "empty"
    assert metadata["candidate_count"] == 0
    assert metadata["result_count"] == 0
    assert metadata["empty_result"] is True
    assert metadata["fallback_used"] is False


def test_retrieval_agent_missing_document_is_skipped():
    """
    Tool-layer contract test.

    A retrieval tool is responsible for returning usable document
    dictionaries. If the tool returns no documents, the graph agent
    should safely produce an empty retrieval result.
    """

    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            side_effect=identity_rerank,
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_STRATEGY",
            "hybrid",
        ),
    ):
        result = retrieval_agent(base_state())

    assert result["retrieved_documents"] == []
    assert result["retrieved_context"] == ""

    metadata = result["retrieval_metadata"]

    assert metadata["result_count"] == 0
    assert metadata["status"] == "empty"
    assert metadata["empty_result"] is True


def test_retrieval_agent_failure_sets_fallback():
    def failing_search(*args, **kwargs):
        raise RuntimeError("retrieval backend unavailable")

    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            side_effect=failing_search,
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_STRATEGY",
            "hybrid",
        ),
    ):
        result = retrieval_agent(base_state())

    assert result["retrieved_context"] == ""
    assert result["retrieved_documents"] == []

    assert result["fallback_used"] is True
    assert result["errors"]

    assert "retrieval_agent" in result["errors"][0]

    metadata = result["retrieval_metadata"]

    assert metadata["status"] == "failure"
    assert metadata["fallback_used"] is True
    assert metadata["error"] is not None
    assert "RuntimeError" in metadata["error"]


def test_retrieval_agent_unsupported_strategy_fails_safely():
    with patch(
        "ai.graph.nodes.retrieval.RETRIEVAL_STRATEGY",
        "unsupported",
    ):
        result = retrieval_agent(base_state())

    assert result["retrieved_context"] == ""
    assert result["retrieved_documents"] == []

    assert result["fallback_used"] is True
    assert result["errors"]

    metadata = result["retrieval_metadata"]

    assert metadata["status"] == "failure"
    assert metadata["fallback_used"] is True
    assert "Unsupported RETRIEVAL_STRATEGY" in metadata["error"]


def test_retrieval_metadata_contains_score_statistics():
    documents = [
        make_document(
            index_id=1,
            score=0.90,
            title="VPN Guide",
            content="Restart VPN.",
        ),
        make_document(
            index_id=2,
            score=0.70,
            title="Network Guide",
            content="Check network settings.",
        ),
    ]

    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=documents,
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            side_effect=identity_rerank,
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_STRATEGY",
            "hybrid",
        ),
    ):
        result = retrieval_agent(base_state())

    metadata = result["retrieval_metadata"]

    assert metadata["top_score"] == 0.90
    assert metadata["second_score"] == 0.70
    assert metadata["score_gap"] == pytest.approx(0.20)
    assert metadata["min_score"] == 0.70
    assert metadata["max_score"] == 0.90
    assert metadata["mean_score"] == 0.80

    assert metadata["source_count"] == 1
    assert metadata["retrievers"] == [
        "hybrid",
        "hybrid",
    ]


def test_retrieval_agent_preserves_existing_fallback_on_success():
    documents = [
        make_document(
            index_id=1,
            score=0.88,
        )
    ]

    state = base_state()
    state["fallback_used"] = True

    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=documents,
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            side_effect=identity_rerank,
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_STRATEGY",
            "hybrid",
        ),
    ):
        result = retrieval_agent(state)

    # Retrieval itself succeeded, so retrieval metadata should
    # describe the retrieval operation rather than inheriting
    # the previous state-level fallback flag.
    assert result["retrieval_metadata"]["fallback_used"] is False
    assert result["retrieval_metadata"]["status"] == "success"


def test_retrieval_agent_metadata_contains_runtime_dimensions():
    with (
        patch(
            "ai.graph.nodes.retrieval.search_knowledge_base",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.search_previous_tickets",
            return_value=[],
        ),
        patch(
            "ai.graph.nodes.retrieval.rerank_documents",
            side_effect=identity_rerank,
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_STRATEGY",
            "hybrid",
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_TOP_K",
            5,
        ),
        patch(
            "ai.graph.nodes.retrieval.RETRIEVAL_CANDIDATE_K",
            20,
        ),
    ):
        result = retrieval_agent(base_state())

    metadata = result["retrieval_metadata"]

    assert metadata["strategy"] == "hybrid"
    assert metadata["top_k"] == 5
    assert metadata["candidate_k"] == 20
    assert metadata["query_length"] > 0
    assert "decision_reason" in metadata
