"""
models.py — Canonical data models for the ResolveX RAG ingestion pipeline.

These models define the contract between:
    loader → cleaner → validator → chunker → metadata → embedding → index
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class CanonicalDocument(BaseModel):
    """
    Canonical representation of a knowledge-base document before chunking.
    """

    document_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)

    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)

    category: Optional[str] = None
    source_type: str = "knowledge_base"

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    version: str = "1.0"

    content_hash: Optional[str] = None


class DocumentChunk(BaseModel):
    """
    A chunk generated from a CanonicalDocument.

    Each chunk retains enough metadata to trace it back to its
    original source document.
    """

    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)

    chunk_index: int = Field(..., ge=0)

    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)

    category: Optional[str] = None
    source_type: str = "knowledge_base"

    created_at: datetime

    version: str = "1.0"

    content_hash: Optional[str] = None


class ChunkMetadata(BaseModel):
    """
    Metadata persisted alongside an embedding/vector.

    This is the metadata that will eventually be associated
    with a FAISS vector.
    """

    chunk_id: str
    document_id: str
    source_id: str

    chunk_index: int = Field(..., ge=0)

    title: str
    category: Optional[str] = None
    source_type: str

    created_at: datetime
    version: str

    content_hash: str

    content: str
