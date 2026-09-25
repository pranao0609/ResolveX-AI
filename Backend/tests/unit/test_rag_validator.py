from datetime import datetime, timezone

from ai.rag.models import CanonicalDocument
from ai.rag.validator import DocumentValidator


def make_document(**overrides):
    data = {
        "document_id": "kb:1",
        "source_id": "knowledge_base:1",
        "title": "Network Troubleshooting",
        "content": "Follow these steps to troubleshoot network connectivity.",
        "category": "network",
        "source_type": "knowledge_base",
        "created_at": datetime.now(timezone.utc),
        "version": "1.0",
    }

    data.update(overrides)

    return CanonicalDocument(**data)


def test_valid_document_passes_validation():
    document = make_document()

    result = DocumentValidator.validate(document)

    assert result.valid is True
    assert result.errors == []
    assert result.error_count == 0


def test_empty_title_fails_validation():
    document = make_document(title=" ")

    result = DocumentValidator.validate(document)

    assert result.valid is False
    assert "title must not be empty" in result.errors


def test_empty_content_fails_validation():
    document = make_document(content=" ")

    result = DocumentValidator.validate(document)

    assert result.valid is False
    assert "content must not be empty" in result.errors


def test_empty_source_type_fails_validation():
    document = make_document(source_type=" ")

    result = DocumentValidator.validate(document)

    assert result.valid is False
    assert "source_type must not be empty" in result.errors


def test_empty_version_fails_validation():
    document = make_document(version=" ")

    result = DocumentValidator.validate(document)

    assert result.valid is False
    assert "version must not be empty" in result.errors


def test_multiple_validation_errors_are_returned():
    document = make_document(
        title=" ",
        content=" ",
        source_type=" ",
        version=" ",
    )

    result = DocumentValidator.validate(document)

    assert result.valid is False
    assert result.error_count == 4


def test_validate_all_separates_valid_and_invalid_documents():
    valid_document = make_document(
        document_id="kb:1",
        source_id="knowledge_base:1",
    )

    invalid_document = make_document(
        document_id="kb:2",
        source_id="knowledge_base:2",
        content=" ",
    )

    valid_documents, invalid_documents = DocumentValidator.validate_all(
        [valid_document, invalid_document]
    )

    assert len(valid_documents) == 1
    assert len(invalid_documents) == 1

    assert valid_documents[0].document_id == "kb:1"
    assert invalid_documents[0].document_id == "kb:2"