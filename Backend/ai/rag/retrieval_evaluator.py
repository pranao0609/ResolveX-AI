"""
retrieval_evaluator.py — Retrieval evaluation for ResolveX.

Evaluates:

    BM25
    Dense FAISS
    Hybrid BM25 + Dense
    Hybrid BM25 + Dense + Cross-Encoder Reranker

Metrics:

    Recall@K
    Precision@K
    MRR
    nDCG@K
"""

from typing import Dict, List, Set

from ai.rag.bm25_store import get_bm25_store
from ai.rag.hybrid_retriever import HybridRetriever
from ai.rag.retriever import retrieve_context
from ai.rag.reranker import get_reranker

from evaluation.retrieval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class RetrievalEvaluator:
    """Evaluate ResolveX retrieval strategies."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
    ):
        self.bm25_store = get_bm25_store()

        # Production retriever remains configurable through the
        # normal application configuration.
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()

        # Controlled experimental retriever used only for
        # Phase 8 evaluation.
        #
        # This does NOT modify production configuration.
        self.experimental_hybrid_retriever = HybridRetriever(
            bm25_weight=0.5,
            dense_weight=0.5,
        )

        self.reranker = get_reranker()

    # ------------------------------------------------------------------
    # Metric wrappers
    # ------------------------------------------------------------------

    @staticmethod
    def _recall_at_k(
        retrieved: List[int],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Recall@K.

        Converts internal FAISS index IDs into the KB document ID
        format used by the evaluation dataset.
        """

        retrieved_ids = [f"kb:{index_id + 1}" for index_id in retrieved]

        return recall_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
            k=k,
        )

    @staticmethod
    def _precision_at_k(
        retrieved: List[int],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        Precision@K.

        Converts internal FAISS index IDs into the KB document ID
        format used by the evaluation dataset.
        """

        retrieved_ids = [f"kb:{index_id + 1}" for index_id in retrieved]

        return precision_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
            k=k,
        )

    @staticmethod
    def _mrr(
        retrieved: List[int],
        relevant: Set[str],
    ) -> float:
        """Mean Reciprocal Rank for one query."""

        retrieved_ids = [f"kb:{index_id + 1}" for index_id in retrieved]

        return reciprocal_rank(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
        )

    @staticmethod
    def _ndcg_at_k(
        retrieved: List[int],
        relevant: Set[str],
        k: int,
    ) -> float:
        """
        nDCG@K using binary relevance.

        Relevant document:
            relevance = 1

        Non-relevant document:
            relevance = 0
        """

        retrieved_ids = [f"kb:{index_id + 1}" for index_id in retrieved]

        return ndcg_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant,
            k=k,
        )

    # ------------------------------------------------------------------
    # Result ID extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_bm25_ids(
        results: List[Dict],
    ) -> List[int]:
        """Extract internal document IDs from BM25 results."""

        return [int(result["index_id"]) for result in results]

    @staticmethod
    def _extract_dense_ids(
        results: List[Dict],
    ) -> List[int]:
        """Extract internal document IDs from dense results."""

        return [
            int(result["index_id"])
            for result in results
            if result.get("index_id") is not None
        ]

    @staticmethod
    def _extract_hybrid_ids(
        results,
    ) -> List[int]:
        """Extract internal document IDs from hybrid results."""

        return [int(result.index_id) for result in results]

    @staticmethod
    def _extract_reranker_ids(
        results,
    ) -> List[int]:
        """Extract internal document IDs from reranker results."""

        return [int(result.index_id) for result in results]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_retriever(
        self,
        retriever_name: str,
        evaluation_queries: List[Dict],
        k_values: tuple[int, ...] = (5, 10),
    ) -> Dict:
        """Evaluate one retrieval strategy."""

        all_recall = {k: [] for k in k_values}

        all_precision = {k: [] for k in k_values}

        all_ndcg = {k: [] for k in k_values}

        reciprocal_ranks = []

        per_query = []

        max_k = max(k_values)

        for item in evaluation_queries:
            query = item["query"]

            relevant = set(item["relevant_document_ids"])

            # ----------------------------------------------------------
            # BM25
            # ----------------------------------------------------------

            if retriever_name == "bm25":
                results = self.bm25_store.search(
                    query=query,
                    top_k=max_k,
                )

                retrieved_ids = self._extract_bm25_ids(results)

            # ----------------------------------------------------------
            # Dense FAISS
            # ----------------------------------------------------------

            elif retriever_name == "dense":
                results = retrieve_context(
                    query=query,
                    top_k=max_k,
                    score_threshold=-1.0,
                )

                retrieved_ids = self._extract_dense_ids(results)

            # ----------------------------------------------------------
            # Experimental 50/50 Hybrid
            # ----------------------------------------------------------

            elif retriever_name == "hybrid":
                results = self.experimental_hybrid_retriever.retrieve(
                    query=query,
                    top_k=max_k,
                    candidate_k=20,
                )

                retrieved_ids = self._extract_hybrid_ids(results)

            # ----------------------------------------------------------
            # Experimental 50/50 Hybrid + Cross Encoder
            # ----------------------------------------------------------

            elif retriever_name == "hybrid_reranker":
                candidates = self.experimental_hybrid_retriever.retrieve(
                    query=query,
                    top_k=20,
                    candidate_k=20,
                )

                results = self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_k=max_k,
                )

                retrieved_ids = self._extract_reranker_ids(results)

            else:
                raise ValueError(f"Unknown retriever: {retriever_name}")

            # ----------------------------------------------------------
            # Per-query result container
            # ----------------------------------------------------------

            query_metrics = {
                "query_id": item["query_id"],
                "query": query,
                "retrieved_ids": retrieved_ids,
            }

            # ----------------------------------------------------------
            # Recall / Precision / nDCG
            # ----------------------------------------------------------

            for k in k_values:
                recall = self._recall_at_k(
                    retrieved=retrieved_ids,
                    relevant=relevant,
                    k=k,
                )

                precision = self._precision_at_k(
                    retrieved=retrieved_ids,
                    relevant=relevant,
                    k=k,
                )

                ndcg = self._ndcg_at_k(
                    retrieved=retrieved_ids,
                    relevant=relevant,
                    k=k,
                )

                all_recall[k].append(recall)
                all_precision[k].append(precision)
                all_ndcg[k].append(ndcg)

                query_metrics[f"recall@{k}"] = recall

                query_metrics[f"precision@{k}"] = precision

                query_metrics[f"ndcg@{k}"] = ndcg

            # ----------------------------------------------------------
            # MRR
            # ----------------------------------------------------------

            reciprocal_rank_score = self._mrr(
                retrieved=retrieved_ids,
                relevant=relevant,
            )

            reciprocal_ranks.append(reciprocal_rank_score)

            query_metrics["reciprocal_rank"] = reciprocal_rank_score

            per_query.append(query_metrics)

        # --------------------------------------------------------------
        # Aggregate metrics
        # --------------------------------------------------------------

        count = len(evaluation_queries)

        metrics = {
            "retriever": retriever_name,
            "queries": count,
            "mrr": (sum(reciprocal_ranks) / count if count else 0.0),
            "per_query": per_query,
        }

        for k in k_values:
            metrics[f"recall@{k}"] = sum(all_recall[k]) / count if count else 0.0

            metrics[f"precision@{k}"] = sum(all_precision[k]) / count if count else 0.0

            metrics[f"ndcg@{k}"] = sum(all_ndcg[k]) / count if count else 0.0

        return metrics

    # ------------------------------------------------------------------
    # Evaluate all retrieval strategies
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        evaluation_queries: List[Dict],
    ) -> Dict[str, Dict]:
        """
        Evaluate:

            BM25
            Dense
            50/50 Hybrid
            50/50 Hybrid + Reranker
        """

        return {
            "bm25": self.evaluate_retriever(
                "bm25",
                evaluation_queries,
            ),
            "dense": self.evaluate_retriever(
                "dense",
                evaluation_queries,
            ),
            "hybrid": self.evaluate_retriever(
                "hybrid",
                evaluation_queries,
            ),
            "hybrid_reranker": self.evaluate_retriever(
                "hybrid_reranker",
                evaluation_queries,
            ),
        }
