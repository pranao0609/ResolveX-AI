import ai.rag.bm25_store as bm25_module


def reset_bm25_store():
    bm25_module._bm25_store = None


class FakeDocumentStore:
    def __init__(self, documents):
        self.documents = documents
        self.total_documents = len(documents)


def test_bm25_builds_from_document_store(monkeypatch):
    reset_bm25_store()

    documents = [
        {
            "index_id": 0,
            "title": "Password Reset",
            "category": "account",
            "content": "Reset your password from the account settings.",
        },
        {
            "index_id": 1,
            "title": "VPN Connection",
            "category": "network",
            "content": "Troubleshoot VPN connection failures.",
        },
    ]

    fake_store = FakeDocumentStore(documents)

    monkeypatch.setattr(
        bm25_module,
        "get_doc_store",
        lambda: fake_store,
    )

    store = bm25_module.BM25Store()

    store.build()

    assert store.is_built is True
    assert store.total_documents == 2
    assert store.validate_alignment() is True


def test_bm25_returns_ranked_results(monkeypatch):
    reset_bm25_store()

    documents = [
        {
            "index_id": 0,
            "title": "Password Reset",
            "category": "account",
            "content": "Reset your password from account settings.",
        },
        {
            "index_id": 1,
            "title": "VPN Connection",
            "category": "network",
            "content": "Troubleshoot VPN connection failures.",
        },
    ]

    fake_store = FakeDocumentStore(documents)

    monkeypatch.setattr(
        bm25_module,
        "get_doc_store",
        lambda: fake_store,
    )

    store = bm25_module.BM25Store()
    store.build()

    results = store.search(
        "password reset",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["index_id"] == 0
    assert isinstance(results[0]["score"], float)


def test_bm25_empty_store(monkeypatch):
    reset_bm25_store()

    fake_store = FakeDocumentStore([])

    monkeypatch.setattr(
        bm25_module,
        "get_doc_store",
        lambda: fake_store,
    )

    store = bm25_module.BM25Store()

    store.build()

    assert store.is_built is False
    assert store.search("password reset") == []


def test_bm25_alignment_failure(monkeypatch):
    reset_bm25_store()

    documents = [
        {
            "index_id": 0,
            "title": "Password Reset",
            "category": "account",
            "content": "Reset your password.",
        },
    ]

    fake_store = FakeDocumentStore(documents)

    monkeypatch.setattr(
        bm25_module,
        "get_doc_store",
        lambda: fake_store,
    )

    store = bm25_module.BM25Store()

    store.build()

    # Simulate the DocumentStore changing after BM25 was built.
    fake_store.total_documents = 2

    assert store.validate_alignment() is False
