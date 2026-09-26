"""
chunker.py — Structure-aware document chunking for RAG.

The chunker attempts to preserve:
    1. Paragraph boundaries
    2. Sentence boundaries
    3. Configured chunk size
    4. Configured overlap

It never modifies the source CanonicalDocument.
"""

import re
from typing import List

from ai.config.ai_config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_MIN_CHUNK_SIZE,
)
from ai.rag.models import CanonicalDocument, DocumentChunk


class DocumentChunker:
    """Split canonical documents into retrieval-friendly chunks."""

    def __init__(
        self,
        chunk_size: int = RAG_CHUNK_SIZE,
        chunk_overlap: int = RAG_CHUNK_OVERLAP,
        min_chunk_size: int = RAG_MIN_CHUNK_SIZE,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")

        if min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size cannot exceed chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(
        self,
        document: CanonicalDocument,
    ) -> List[DocumentChunk]:
        """
        Split one canonical document into ordered chunks.
        """

        paragraphs = self._split_paragraphs(document.content)

        if not paragraphs:
            return []

        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            # Paragraph itself fits inside the target chunk.
            if len(paragraph) <= self.chunk_size:
                candidate = paragraph if not current else f"{current}\n\n{paragraph}"

                if len(candidate) <= self.chunk_size:
                    current = candidate
                    continue

                if current:
                    chunks.append(current)

                current = paragraph
                continue

            # Long paragraph requires sentence-aware splitting.
            if current:
                chunks.append(current)
                current = ""

            paragraph_chunks = self._split_long_text(paragraph)

            chunks.extend(paragraph_chunks)

        if current:
            chunks.append(current)

        chunks = self._merge_small_chunks(chunks)

        return [
            self._create_chunk(
                document=document,
                content=content,
                chunk_index=index,
            )
            for index, content in enumerate(chunks)
        ]

    @staticmethod
    def _split_paragraphs(content: str) -> List[str]:
        """Split content using blank lines as paragraph boundaries."""

        paragraphs = re.split(r"\n\s*\n", content)

        return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    def _split_long_text(self, text: str) -> List[str]:
        """
        Split text larger than chunk_size using sentence boundaries
        where possible.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        if len(sentences) == 1:
            return self._split_by_size(text)

        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            candidate = sentence if not current else f"{current} {sentence}"

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            # A single sentence can itself exceed the limit.
            if len(sentence) > self.chunk_size:
                chunks.extend(self._split_by_size(sentence))
                current = ""
            else:
                current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _split_by_size(self, text: str) -> List[str]:
        """
        Fallback splitter for text containing no usable sentence
        boundaries.

        Uses character windows with configured overlap.
        """

        text = text.strip()

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(
                start + self.chunk_size,
                text_length,
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _merge_small_chunks(
        self,
        chunks: List[str],
    ) -> List[str]:
        """
        Merge very small chunks with their neighboring chunk
        where doing so does not exceed chunk_size.
        """

        if len(chunks) <= 1:
            return chunks

        merged: List[str] = []

        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue

            previous = merged[-1]

            if (
                len(chunk) < self.min_chunk_size
                and len(previous) + 2 + len(chunk) <= self.chunk_size
            ):
                merged[-1] = f"{previous}\n\n{chunk}"
            else:
                merged.append(chunk)

        return merged

    @staticmethod
    def _create_chunk(
        document: CanonicalDocument,
        content: str,
        chunk_index: int,
    ) -> DocumentChunk:
        """Create a DocumentChunk while preserving source metadata."""

        chunk_id = f"{document.document_id}:chunk:{chunk_index}"

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            source_id=document.source_id,
            chunk_index=chunk_index,
            title=document.title,
            content=content,
            category=document.category,
            source_type=document.source_type,
            created_at=document.created_at,
            version=document.version,
            content_hash=document.content_hash,
        )
