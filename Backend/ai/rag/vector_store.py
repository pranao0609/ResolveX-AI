"""
vector_store.py — FAISS vector index manager.

Uses cosine-like similarity via IndexFlatIP because embeddings are normalized.
Handles building, saving, loading, adding, and searching.
"""

import os
import numpy as np
from typing import Optional, Tuple
from app.core.logger import logger
from ai.config.ai_config import FAISS_INDEX_PATH, EMBEDDING_DIMENSION

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed — vector search unavailable")


class VectorStore:
    """Wraps a FAISS flat inner-product index with load/save/add/search helpers."""

    def __init__(self):
        self.index: Optional[object] = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing index from disk, or create a fresh one."""
        if not FAISS_AVAILABLE:
            return

        if os.path.exists(FAISS_INDEX_PATH):
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                logger.info(
                    f"FAISS index loaded from {FAISS_INDEX_PATH} ({self.index.ntotal} vectors)"
                )
            except Exception as exc:
                logger.error(f"Failed to load FAISS index: {exc}")
                self.index = self._create_index()
        else:
            logger.info("No existing FAISS index found — creating new flat IP index")
            self.index = self._create_index()

    def _create_index(self):
        """Create a new flat inner-product FAISS index."""
        if not FAISS_AVAILABLE:
            return None
        return faiss.IndexFlatIP(EMBEDDING_DIMENSION)

    def add(self, embeddings: np.ndarray) -> None:
        """Add vectors to the index."""
        if self.index is None:
            logger.warning("FAISS index not available — skipping add")
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Invalid embedding shape {embeddings.shape}; expected (N, {EMBEDDING_DIMENSION})"
            )

        self.index.add(embeddings)
        logger.debug(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k most similar vectors.

        Returns:
            Tuple of (scores, indices), each shape (top_k,)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index empty or unavailable")
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        query = np.asarray(query_vector, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2 or query.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Invalid query shape {query.shape}; expected (1, {EMBEDDING_DIMENSION})"
            )

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        return scores[0], indices[0]

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        if self.index is None:
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")

    def reset(self) -> None:
        """Reset the FAISS index."""
        self.index = self._create_index()
        logger.info("FAISS index reset")

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def rollback_to(self, vector_count: int) -> None:
        """
        Roll back the FAISS index to the specified number of vectors.

        This is used to restore index consistency when a coordinated
        RAG indexing operation fails after vectors have been added.
        """

        if self.index is None:
            return

        if vector_count < 0:
            raise ValueError("vector_count cannot be negative")

        current_count = self.index.ntotal

        if vector_count > current_count:
            raise ValueError(
                f"Cannot rollback to {vector_count}; "
                f"current index contains {current_count} vectors"
            )

        if vector_count == current_count:
            return

        if vector_count == 0:
            self.index = self._create_index()
            logger.info("FAISS index rolled back to 0 vectors")
            return

        if not hasattr(self.index, "reconstruct_n"):
            raise RuntimeError(
                "FAISS index does not support vector reconstruction"
            )

        existing_vectors = self.index.reconstruct_n(
            0,
            vector_count,
        )

        self.index = self._create_index()

        if self.index is None:
            raise RuntimeError(
                "FAISS index unavailable while restoring previous state"
            )

        self.index.add(
            np.asarray(
                existing_vectors,
                dtype=np.float32,
            )
        )

        logger.info(
            "FAISS index rolled back from %d to %d vectors",
            current_count,
            vector_count,
        )
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store