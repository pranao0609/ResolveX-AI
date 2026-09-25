from unittest.mock import Mock

from ai.rag.ingestion import IngestionResult
from ai.rag.service import RAGIngestionService


def test_ingest_knowledge_base_loads_documents_and_runs_pipeline():
    db = Mock()

    service = RAGIngestionService(db)

    documents = [
        Mock(),
        Mock(),
    ]

    expected_result = IngestionResult(
        documents_received=2,
        documents_valid=2,
        documents_invalid=0,
        chunks_created=4,
        chunks_indexed=4,
    )

    service.loader.load_all = Mock(
        return_value=documents
    )

    service.pipeline.ingest = Mock(
        return_value=expected_result
    )

    result = service.ingest_knowledge_base()

    service.loader.load_all.assert_called_once_with()

    service.pipeline.ingest.assert_called_once_with(
        documents
    )

    assert result == expected_result


def test_ingest_knowledge_base_handles_empty_database():
    db = Mock()

    service = RAGIngestionService(db)

    service.loader.load_all = Mock(
        return_value=[]
    )

    service.pipeline.ingest = Mock()

    result = service.ingest_knowledge_base()

    service.loader.load_all.assert_called_once_with()

    service.pipeline.ingest.assert_not_called()

    assert result.documents_received == 0
    assert result.documents_valid == 0
    assert result.documents_invalid == 0
    assert result.chunks_created == 0
    assert result.chunks_indexed == 0