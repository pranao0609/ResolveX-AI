import ai.rag.hybrid_retriever as hybrid_module


class FakeBM25Store:
    def search(self, query, top_k):
        return [
            {"index_id": 0, "score": 8.0},
            {"index_id": 1, "score": 6.0},
            {"index_id": 2, "score": 4.0},
        ][:top_k]


class FakeDenseRetriever:
    def retrieve_context(
        self,
        query,
        top_k,
        score_threshold,
    ):
        return [
            {"index_id": 1, "score": 0.90},
            {"index_id": 2, "score": 0.80},
            {"index_id": 3, "score": 0.70},
        ][:top_k]


def create_retriever(monkeypatch):
    monkeypatch.setattr(
        hybrid_module,
        "get_bm25_store",
        lambda: FakeBM25Store(),
    )

    monkeypatch.setattr(
    hybrid_module,
    "retrieve_context",
    FakeDenseRetriever().retrieve_context,
)

    return hybrid_module.HybridRetriever()


def test_hybrid_retriever_returns_results(monkeypatch):
    retriever = create_retriever(monkeypatch)

    results = retriever.retrieve(
        query="password reset",
        top_k=3,
        candidate_k=3,
    )

    assert len(results) == 3

    assert all(
        isinstance(result, hybrid_module.RetrievalCandidate)
        for result in results
    )


def test_hybrid_retriever_merges_candidate_sets(monkeypatch):
    retriever = create_retriever(monkeypatch)

    results = retriever.retrieve(
        query="password reset",
        top_k=4,
        candidate_k=3,
    )

    result_ids = {result.index_id for result in results}

    assert result_ids == {0, 1, 2, 3}


def test_hybrid_retriever_uses_hybrid_label(monkeypatch):
    retriever = create_retriever(monkeypatch)

    results = retriever.retrieve(
        query="password reset",
        top_k=3,
        candidate_k=3,
    )

    assert all(
        result.retriever == "hybrid"
        for result in results
    )


def test_hybrid_retriever_results_are_sorted(monkeypatch):
    retriever = create_retriever(monkeypatch)

    results = retriever.retrieve(
        query="password reset",
        top_k=3,
        candidate_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_hybrid_retriever_empty_query(monkeypatch):
    retriever = create_retriever(monkeypatch)

    try:
        retriever.retrieve("")
        assert False
    except ValueError:
        assert True


def test_hybrid_retriever_invalid_query_type(monkeypatch):
    retriever = create_retriever(monkeypatch)

    try:
        retriever.retrieve(None)
        assert False
    except TypeError:
        assert True


def test_hybrid_retriever_zero_top_k(monkeypatch):
    retriever = create_retriever(monkeypatch)

    results = retriever.retrieve(
        query="password reset",
        top_k=0,
    )

    assert results == []


def test_hybrid_retriever_zero_candidate_k(monkeypatch):
    retriever = create_retriever(monkeypatch)

    results = retriever.retrieve(
        query="password reset",
        top_k=5,
        candidate_k=0,
    )

    assert results == []