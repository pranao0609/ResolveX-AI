from datetime import datetime, timezone

from ai.rag.models import CanonicalDocument
from ai.rag.pipeline import RAGDocumentPipeline


def make_document(
    document_id="kb:1",
    title="  Network   Troubleshooting  ",
    content="Restart   the router.\n\n\nCheck the connection.",
):
    return CanonicalDocument(
        document_id=document_id,
        source_id=f"knowledge_base:{document_id}",
        title=title,
        content=content,
        category="network",
        source_type="knowledge_base",
        created_at=datetime.now(timezone.utc),
        version="1.0",
    )


def test_pipeline_cleans_document():
    document = make_document()

    result = RAGDocumentPipeline.prepare([document])

    assert result.valid_count == 1
    assert result.invalid_count == 0

    prepared = result.valid_documents[0]

    assert prepared.title == "Network Troubleshooting"
    assert prepared.content == (
        "Restart the router.\n\nCheck the connection."
    )


def test_pipeline_attaches_content_hash():
    document = make_document()

    result = RAGDocumentPipeline.prepare([document])

    prepared = result.valid_documents[0]

    assert prepared.content_hash is not None
    assert len(prepared.content_hash) == 64


def test_pipeline_preserves_document_identity():
    document = make_document(
        document_id="kb:123"
    )

    result = RAGDocumentPipeline.prepare([document])

    prepared = result.valid_documents[0]

    assert prepared.document_id == "kb:123"
    assert prepared.source_id == "knowledge_base:kb:123"


def test_pipeline_separates_invalid_documents():
    valid_document = make_document(
        document_id="kb:1"
    )

    invalid_document = make_document(
        document_id="kb:2",
        content="   ",
    )

    result = RAGDocumentPipeline.prepare(
        [
            valid_document,
            invalid_document,
        ]
    )

    assert result.valid_count == 1
    assert result.invalid_count == 1

    assert result.valid_documents[0].document_id == "kb:1"
    assert result.invalid_documents[0].document_id == "kb:2"


def test_pipeline_handles_empty_input():
    result = RAGDocumentPipeline.prepare([])

    assert result.valid_documents == []
    assert result.invalid_documents == []
    assert result.valid_count == 0
    assert result.invalid_count == 0