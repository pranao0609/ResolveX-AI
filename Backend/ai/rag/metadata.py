"""
metadata.py — Chunk metadata construction for the RAG pipeline.

Converts DocumentChunk objects into ChunkMetadata objects.

The metadata builder does not perform:
    - chunking
    - embedding
    - vector indexing
    - persistence

Its only responsibility is constructing the metadata contract
that will accompany each vector.
"""

from typing import List

from ai.rag.models import ChunkMetadata, DocumentChunk


class MetadataBuilder:
    """Build persistent metadata from document chunks."""

    @staticmethod
    def build(
        chunk: DocumentChunk,
    ) -> ChunkMetadata:
        """
        Convert one DocumentChunk into ChunkMetadata.

        A chunk must already contain a content hash because the
        hash is part of the persistent metadata contract.
        """

        if not chunk.content_hash:
            raise ValueError(
                "DocumentChunk must have content_hash " "before metadata can be created"
            )

        return ChunkMetadata(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_id=chunk.source_id,
            chunk_index=chunk.chunk_index,
            title=chunk.title,
            category=chunk.category,
            source_type=chunk.source_type,
            created_at=chunk.created_at,
            version=chunk.version,
            content_hash=chunk.content_hash,
            content=chunk.content,
        )

    @classmethod
    def build_many(
        cls,
        chunks: List[DocumentChunk],
    ) -> List[ChunkMetadata]:
        """Convert multiple document chunks into metadata records."""

        return [cls.build(chunk) for chunk in chunks]
