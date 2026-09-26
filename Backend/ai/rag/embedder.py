"""
embedder.py — RAG chunk embedding service.

Builds embedding text from RAG chunks and delegates model inference
to the existing shared embedding service.

Responsibilities:
    - construct embedding text
    - generate batch embeddings
    - validate embedding dimensions
    - enforce float32 representation
    - preserve normalized embeddings
"""

from typing import List

import numpy as np

from ai.config.ai_config import EMBEDDING_DIMENSION
from ai.embedding.embedding_model import generate_batch_embeddings
from ai.rag.models import DocumentChunk


class RAGEmbedder:
    """Generate embeddings for RAG document chunks."""

    @staticmethod
    def build_embedding_text(
        chunk: DocumentChunk,
    ) -> str:
        """
        Build the text representation used for embedding.

        Title and category provide useful retrieval context while
        the chunk content remains the primary semantic signal.
        """

        parts = [
            f"Title: {chunk.title}",
        ]

        if chunk.category:
            parts.append(f"Category: {chunk.category}")

        parts.append(f"Content: {chunk.content}")

        return "\n".join(parts)

    @classmethod
    def build_embedding_texts(
        cls,
        chunks: List[DocumentChunk],
    ) -> List[str]:
        """Build embedding text for multiple chunks."""

        return [cls.build_embedding_text(chunk) for chunk in chunks]

    @staticmethod
    def validate_embeddings(
        embeddings: np.ndarray,
        expected_count: int,
    ) -> None:
        """
        Validate the shape and dtype of generated embeddings.
        """

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy.ndarray")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if embeddings.shape[0] != expected_count:
            raise ValueError(
                "Embedding count mismatch: "
                f"expected {expected_count}, "
                f"got {embeddings.shape[0]}"
            )

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {EMBEDDING_DIMENSION}, "
                f"got {embeddings.shape[1]}"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or infinite values")

    @classmethod
    def embed(
        cls,
        chunks: List[DocumentChunk],
    ) -> np.ndarray:
        """
        Generate normalized embeddings for RAG chunks.

        Returns:
            numpy array of shape:
                (number_of_chunks, EMBEDDING_DIMENSION)
        """

        if not chunks:
            return np.empty(
                (0, EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

        texts = cls.build_embedding_texts(chunks)

        embeddings = generate_batch_embeddings(texts)

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        cls.validate_embeddings(
            embeddings,
            expected_count=len(chunks),
        )

        return embeddings
