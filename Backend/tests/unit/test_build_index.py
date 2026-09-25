from unittest.mock import Mock

import pytest

from scripts import build_index as build_index_module


def make_result():
    result = Mock()

    result.documents_received = 10
    result.documents_valid = 10
    result.documents_invalid = 0
    result.chunks_created = 25
    result.chunks_indexed = 25

    return result


def test_build_index_resets_stores(monkeypatch):
    vector_store = Mock()
    vector_store.total_vectors = 25

    doc_store = Mock()

    service = Mock()
    service.ingest_knowledge_base.return_value = make_result()

    db = Mock()

    monkeypatch.setattr(
        build_index_module,
        "init_db",
        lambda: None,
    )

    monkeypatch.setattr(
        build_index_module,
        "get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        build_index_module,
        "get_doc_store",
        lambda: doc_store,
    )

    monkeypatch.setattr(
        build_index_module,
        "SessionLocal",
        lambda: db,
    )

    monkeypatch.setattr(
        build_index_module,
        "RAGIngestionService",
        lambda session: service,
    )

    monkeypatch.setattr(
        build_index_module,
        "validate_store_alignment",
        lambda: True,
    )

    result = build_index_module.build_index(
        reset=True
    )

    vector_store.reset.assert_called_once()
    doc_store.clear.assert_called_once()

    service.ingest_knowledge_base.assert_called_once()

    db.close.assert_called_once()

    assert result.documents_received == 10


def test_build_index_no_reset_preserves_stores(monkeypatch):
    vector_store = Mock()
    vector_store.total_vectors = 25

    doc_store = Mock()

    service = Mock()
    service.ingest_knowledge_base.return_value = make_result()

    db = Mock()

    monkeypatch.setattr(
        build_index_module,
        "init_db",
        lambda: None,
    )

    monkeypatch.setattr(
        build_index_module,
        "get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        build_index_module,
        "get_doc_store",
        lambda: doc_store,
    )

    monkeypatch.setattr(
        build_index_module,
        "SessionLocal",
        lambda: db,
    )

    monkeypatch.setattr(
        build_index_module,
        "RAGIngestionService",
        lambda session: service,
    )

    monkeypatch.setattr(
        build_index_module,
        "validate_store_alignment",
        lambda: True,
    )

    build_index_module.build_index(
        reset=False
    )

    vector_store.reset.assert_not_called()
    doc_store.clear.assert_not_called()

    service.ingest_knowledge_base.assert_called_once()


def test_build_index_fails_when_stores_are_misaligned(monkeypatch):
    vector_store = Mock()
    vector_store.total_vectors = 10

    doc_store = Mock()

    service = Mock()
    service.ingest_knowledge_base.return_value = make_result()

    db = Mock()

    monkeypatch.setattr(
        build_index_module,
        "init_db",
        lambda: None,
    )

    monkeypatch.setattr(
        build_index_module,
        "get_vector_store",
        lambda: vector_store,
    )

    monkeypatch.setattr(
        build_index_module,
        "get_doc_store",
        lambda: doc_store,
    )

    monkeypatch.setattr(
        build_index_module,
        "SessionLocal",
        lambda: db,
    )

    monkeypatch.setattr(
        build_index_module,
        "RAGIngestionService",
        lambda session: service,
    )

    monkeypatch.setattr(
        build_index_module,
        "validate_store_alignment",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="FAISS and DocumentStore are misaligned",
    ):
        build_index_module.build_index(
            reset=True
        )

    db.close.assert_called_once()