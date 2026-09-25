"""
hasher.py — Content hashing and duplicate detection for RAG documents.

This module provides deterministic SHA-256 hashing for canonical
documents and utilities for detecting duplicate document content.

Document identity and content identity are intentionally separate:

    document_id  -> identifies the source record
    content_hash -> identifies the actual document content
"""

import hashlib
from collections import defaultdict
from typing import Dict, List

from ai.rag.models import CanonicalDocument


class DocumentHasher:
    """Generate deterministic content hashes for RAG documents."""

    @staticmethod
    def hash_content(
        title: str,
        content: str,
    ) -> str:
        """
        Generate a deterministic SHA-256 hash from document content.

        The title is included because changing the title is considered
        a content-level change for RAG indexing purposes.
        """

        normalized = f"{title}\n{content}"

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    @classmethod
    def compute_hash(
        cls,
        document: CanonicalDocument,
    ) -> str:
        """Compute the content hash for a canonical document."""

        return cls.hash_content(
            title=document.title,
            content=document.content,
        )

    @classmethod
    def attach_hash(
        cls,
        document: CanonicalDocument,
    ) -> CanonicalDocument:
        """
        Return a copy of the document with its content_hash populated.
        """

        content_hash = cls.compute_hash(document)

        return document.model_copy(
            update={"content_hash": content_hash}
        )

    @classmethod
    def attach_hashes(
        cls,
        documents: List[CanonicalDocument],
    ) -> List[CanonicalDocument]:
        """Attach content hashes to a collection of documents."""

        return [
            cls.attach_hash(document)
            for document in documents
        ]


class DocumentDeduplicator:
    """Detect documents containing identical content."""

    @staticmethod
    def find_duplicates(
        documents: List[CanonicalDocument],
    ) -> Dict[str, List[str]]:
        """
        Find duplicate documents based on content_hash.

        Returns:
            Dictionary mapping:

                content_hash -> [document_id, document_id, ...]

        Only hashes belonging to multiple documents are returned.
        """

        hash_to_documents: Dict[str, List[str]] = defaultdict(list)

        for document in documents:
            if not document.content_hash:
                continue

            hash_to_documents[
                document.content_hash
            ].append(document.document_id)

        return {
            content_hash: document_ids
            for content_hash, document_ids in hash_to_documents.items()
            if len(document_ids) > 1
        }

    @staticmethod
    def remove_duplicates(
        documents: List[CanonicalDocument],
    ) -> List[CanonicalDocument]:
        """
        Remove duplicate content while preserving the first document.

        Document order is preserved.

        The first document encountered for a given content hash is kept.
        """

        seen_hashes = set()
        unique_documents: List[CanonicalDocument] = []

        for document in documents:
            content_hash = document.content_hash

            if not content_hash:
                unique_documents.append(document)
                continue

            if content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)
            unique_documents.append(document)

        return unique_documents