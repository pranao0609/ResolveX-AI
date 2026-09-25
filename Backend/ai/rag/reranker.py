"""
reranker.py — Cross-encoder reranking for ResolveX.

The reranker takes a query and a retrieved candidate set,
scores each query-document pair using a cross-encoder,
and returns the candidates ordered by the cross-encoder score.
"""

from typing import Dict, List, Sequence, Tuple

from sentence_transformers import CrossEncoder

from ai.rag.doc_store import get_doc_store
from ai.rag.retrieval_models import RetrievalCandidate
from app.core.logger import logger


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Cross-encoder based document reranker."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        batch_size: int = 16,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Reranker model_name must be a non-empty string")

        if batch_size <= 0:
            raise ValueError("Reranker batch_size must be positive")

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """Lazily load the cross-encoder model."""

        if self._model is None:
            logger.info(
                "Loading reranker model: %s",
                self.model_name,
            )

            self._model = CrossEncoder(
                self.model_name,
            )

            logger.info(
                "Reranker model loaded: %s",
                self.model_name,
            )

        return self._model

    @staticmethod
    def _build_pairs(
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> List[Tuple[str, str]]:
        """Build query-document pairs for the cross-encoder."""

        doc_store = get_doc_store()
        pairs = []

        for candidate in candidates:
            document = doc_store.get_chunk(candidate.index_id)

            if document is None:
                continue

            title = getattr(document, "title", "") or ""
            content = getattr(document, "content", "") or ""

            document_text = f"{title}\n{content}".strip()

            pairs.append(
                (
                    query,
                    document_text,
                )
            )

        return pairs

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """
        Rerank retrieved candidates using the cross-encoder.

        Args:
            query: Original user query.
            candidates: Candidates produced by BM25/dense/hybrid retrieval.
            top_k: Number of final candidates to return.

        Returns:
            Candidates sorted by cross-encoder relevance score.
        """

        if not isinstance(query, str):
            raise TypeError("Reranker query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("Reranker query must not be empty")

        if top_k <= 0:
            return []

        if not candidates:
            return []

        pairs = self._build_pairs(
            query=query,
            candidates=candidates,
        )

        if not pairs:
            return []

        valid_candidates = [
            candidate
            for candidate in candidates
            if get_doc_store().get_chunk(candidate.index_id) is not None
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = [
            RetrievalCandidate(
                index_id=candidate.index_id,
                score=float(score),
                retriever="reranker",
            )
            for candidate, score in zip(valid_candidates, scores)
        ]

        reranked.sort(
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        return reranked[:top_k]


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Return the shared cross-encoder reranker singleton."""

    global _reranker

    if _reranker is None:
        _reranker = CrossEncoderReranker()

    return _reranker