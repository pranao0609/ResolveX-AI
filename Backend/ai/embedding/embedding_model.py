"""
embedding_model.py — Sentence embedding using SentenceTransformers.

Model: all-MiniLM-L6-v2 (384-dim, fast and accurate).
Lazy-loads on first call to avoid startup delays.
"""

from typing import List
import numpy as np
from app.core.logger import logger
from ai.config.ai_config import EMBEDDING_MODEL_NAME

_model = None  # lazy singleton


def _get_model():
    """Load the embedding model once and cache it."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        try:
            from transformers import logging

            logging.set_verbosity_error()

            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            logger.info("Embedding model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load embedding model: {exc}")
            _model = None
    return _model


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a dense vector embedding for the given text.

    Args:
        text: Input string (pre-processed ticket text).

    Returns:
        numpy array of shape (384,), or a zero vector if model is unavailable.
    """
    from ai.config.ai_config import EMBEDDING_DIMENSION

    model = _get_model()
    if model is None:
        logger.warning("Embedding model unavailable, returning zero vector")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    try:
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)
    except Exception as exc:
        logger.error(f"Embedding generation failed: {exc}")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)


def generate_batch_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a batch of texts (used when building the FAISS index).

    Returns:
        numpy array of shape (N, 384).
    """
    model = _get_model()
    if model is None:
        from ai.config.ai_config import EMBEDDING_DIMENSION

        return np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)

    return model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)
