"""
hybrid_retriever.py — Hybrid BM25 + Dense retrieval.

Combines:

    BM25 lexical retrieval
            +
    Dense FAISS retrieval
            ↓
       Hybrid Fusion
            ↓
     Ranked candidates
"""

from typing import List

from app.core.logger import logger
from ai.rag.bm25_store import get_bm25_store
from ai.rag.hybrid_fusion import HybridFusion
from ai.rag.retrieval_models import RetrievalCandidate
from ai.rag.retriever import retrieve_context
from ai.config.ai_config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)


class HybridRetriever:
    """Hybrid BM25 + dense retriever."""

    def __init__(
        self,
        bm25_weight: float = BM25_WEIGHT,
        dense_weight: float = DENSE_WEIGHT,
    ):
        self.bm25_store = get_bm25_store()

        self.fusion = HybridFusion(
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def _retrieve_bm25(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalCandidate]:
        """Retrieve lexical candidates using BM25."""

        results = self.bm25_store.search(
            query=query,
            top_k=top_k,
        )

        return [
            RetrievalCandidate(
                index_id=result["index_id"],
                score=float(result["score"]),
                retriever="bm25",
            )
            for result in results
        ]

    def _retrieve_dense(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalCandidate]:
        """Retrieve semantic candidates using the existing FAISS retriever."""

        results = retrieve_context(
            query=query,
            top_k=top_k,
            score_threshold=-1.0,
        )

        candidates = []

        for result in results:
            index_id = result.get("index_id")

            if index_id is None:
                continue

            candidates.append(
                RetrievalCandidate(
                    index_id=int(index_id),
                    score=float(result.get("score", 0.0)),
                    retriever="dense",
                )
            )

        return candidates

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        candidate_k: int = RETRIEVAL_CANDIDATE_K,
    ) -> List[RetrievalCandidate]:
        """
        Perform hybrid BM25 + dense retrieval.

        Args:
            query: User search query.
            top_k: Number of final hybrid results.
            candidate_k: Number of candidates retrieved from
                each individual retriever before fusion.

        Returns:
            Ranked hybrid retrieval candidates.
        """

        if not isinstance(query, str):
            raise TypeError("Hybrid query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("Hybrid query must not be empty")

        if top_k <= 0:
            return []

        if candidate_k <= 0:
            return []

        bm25_candidates = self._retrieve_bm25(
            query=query,
            top_k=candidate_k,
        )

        dense_candidates = self._retrieve_dense(
            query=query,
            top_k=candidate_k,
        )

        logger.info(
            "Hybrid retrieval candidates: query=%r, bm25=%d, dense=%d",
            query,
            len(bm25_candidates),
            len(dense_candidates),
        )

        hybrid_candidates = self.fusion.fuse(
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
            top_k=top_k,
        )

        logger.info(
            "Hybrid retrieval completed: query=%r, results=%d",
            query,
            len(hybrid_candidates),
        )

        return hybrid_candidates
