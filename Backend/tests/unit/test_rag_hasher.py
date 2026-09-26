from datetime import datetime, timezone

from ai.rag.hasher import DocumentDeduplicator, DocumentHasher
from ai.rag.models import CanonicalDocument


def make_document(
    document_id="kb:1",
    title="Network Troubleshooting",
    content="Restart the router and check the network connection.",
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


def test_hash_is_deterministic():
    document = make_document()

    hash_one = DocumentHasher.compute_hash(document)
    hash_two = DocumentHasher.compute_hash(document)

    assert hash_one == hash_two


def test_hash_is_sha256():
    document = make_document()

    content_hash = DocumentHasher.compute_hash(document)

    assert len(content_hash) == 64
    assert all(character in "0123456789abcdef" for character in content_hash)


def test_identical_documents_have_same_hash():
    document_one = make_document(document_id="kb:1")
    document_two = make_document(document_id="kb:2")

    hash_one = DocumentHasher.compute_hash(document_one)
    hash_two = DocumentHasher.compute_hash(document_two)

    assert hash_one == hash_two


def test_different_content_has_different_hash():
    document_one = make_document(content="Restart the router.")

    document_two = make_document(content="Restart the computer.")

    hash_one = DocumentHasher.compute_hash(document_one)
    hash_two = DocumentHasher.compute_hash(document_two)

    assert hash_one != hash_two


def test_different_title_has_different_hash():
    document_one = make_document(title="Network Troubleshooting")

    document_two = make_document(title="Network Connectivity")

    hash_one = DocumentHasher.compute_hash(document_one)
    hash_two = DocumentHasher.compute_hash(document_two)

    assert hash_one != hash_two


def test_attach_hash_populates_content_hash():
    document = make_document()

    assert document.content_hash is None

    hashed_document = DocumentHasher.attach_hash(document)

    assert hashed_document.content_hash is not None
    assert len(hashed_document.content_hash) == 64


def test_attach_hash_does_not_modify_original_document():
    document = make_document()

    hashed_document = DocumentHasher.attach_hash(document)

    assert document.content_hash is None
    assert hashed_document.content_hash is not None


def test_attach_hashes_processes_all_documents():
    documents = [
        make_document(document_id="kb:1"),
        make_document(
            document_id="kb:2",
            content="Check the Ethernet cable.",
        ),
        make_document(
            document_id="kb:3",
            content="Restart the network service.",
        ),
    ]

    hashed_documents = DocumentHasher.attach_hashes(documents)

    assert len(hashed_documents) == 3

    for document in hashed_documents:
        assert document.content_hash is not None
        assert len(document.content_hash) == 64


def test_duplicate_documents_are_detected():
    document_one = DocumentHasher.attach_hash(make_document(document_id="kb:1"))

    document_two = DocumentHasher.attach_hash(make_document(document_id="kb:2"))

    duplicates = DocumentDeduplicator.find_duplicates([document_one, document_two])

    assert len(duplicates) == 1

    duplicate_document_ids = next(iter(duplicates.values()))

    assert duplicate_document_ids == ["kb:1", "kb:2"]


def test_unique_documents_are_not_reported_as_duplicates():
    document_one = DocumentHasher.attach_hash(
        make_document(
            document_id="kb:1",
            content="Restart the router.",
        )
    )

    document_two = DocumentHasher.attach_hash(
        make_document(
            document_id="kb:2",
            content="Check the Ethernet cable.",
        )
    )

    duplicates = DocumentDeduplicator.find_duplicates([document_one, document_two])

    assert duplicates == {}


def test_remove_duplicates_preserves_first_document():
    document_one = DocumentHasher.attach_hash(make_document(document_id="kb:1"))

    document_two = DocumentHasher.attach_hash(make_document(document_id="kb:2"))

    unique_documents = DocumentDeduplicator.remove_duplicates(
        [document_one, document_two]
    )

    assert len(unique_documents) == 1
    assert unique_documents[0].document_id == "kb:1"


def test_remove_duplicates_preserves_unique_documents():
    document_one = DocumentHasher.attach_hash(
        make_document(
            document_id="kb:1",
            content="Restart the router.",
        )
    )

    document_two = DocumentHasher.attach_hash(
        make_document(
            document_id="kb:2",
            content="Check the Ethernet cable.",
        )
    )

    unique_documents = DocumentDeduplicator.remove_duplicates(
        [document_one, document_two]
    )

    assert len(unique_documents) == 2


def test_documents_without_hash_are_not_removed():
    document_one = make_document(document_id="kb:1")
    document_two = make_document(
        document_id="kb:2",
        content="Different content.",
    )

    unique_documents = DocumentDeduplicator.remove_duplicates(
        [document_one, document_two]
    )

    assert len(unique_documents) == 2
