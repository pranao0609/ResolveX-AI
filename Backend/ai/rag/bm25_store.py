"""
bm25_store.py — BM25 lexical retrieval store.

The BM25 index is built directly from the persistent RAG
DocumentStore so its document ordering remains aligned with:

    BM25 index N
        ↕
    DocumentStore metadata N
        ↕
    FAISS vector N
"""

from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from app.core.logger import logger
from ai.rag.doc_store import get_doc_store


class BM25Store:
    """BM25 index over the current RAG chunk corpus."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self._document_count: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text for BM25 retrieval.

        A deliberately simple whitespace tokenizer is used initially.
        This preserves technical tokens such as:

            VPN
            API
            401
            password-reset
            connection_failed

        without introducing aggressive normalization.
        """

        if not isinstance(text, str):
            return []

        return text.lower().split()

    @staticmethod
    def _build_search_text(document: Dict) -> str:
        """
        Build the lexical representation indexed by BM25.

        Title and category are included because they often contain
        high-value lexical signals for IT support queries.
        """

        parts = []

        title = document.get("title")
        category = document.get("category")
        content = document.get("content")

        if title:
            parts.append(str(title))

        if category:
            parts.append(str(category))

        if content:
            parts.append(str(content))

        return " ".join(parts)

    def build(self) -> None:
        """
        Build the BM25 index from the current DocumentStore.
        """

        doc_store = get_doc_store()
        documents = doc_store.documents

        self.tokenized_corpus = []
        self.bm25 = None
        self._document_count = 0

        if not documents:
            logger.warning(
                "Cannot build BM25 index — DocumentStore is empty"
            )
            return

        for document in documents:
            search_text = self._build_search_text(document)
            tokens = self._tokenize(search_text)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._document_count = len(documents)

        logger.info(
            "BM25 index built successfully: %d documents",
            self._document_count,
        )

    def validate_alignment(self) -> bool:
        """
        Verify BM25 document count matches DocumentStore count.
        """

        doc_store = get_doc_store()
        expected_count = doc_store.total_documents

        if self._document_count != expected_count:
            logger.warning(
                "BM25/DocumentStore mismatch: BM25=%d, docs=%d",
                self._document_count,
                expected_count,
            )
            return False

        logger.info(
            "BM25/DocumentStore aligned: documents=%d",
            expected_count,
        )

        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search the BM25 index and return ranked document IDs/scores.
    
        If the BM25 index has not been built yet, it is automatically
        built from the current persistent DocumentStore.
        """
    
        if self.bm25 is None:
            logger.info(
                "BM25 index is not built — building from DocumentStore"
            )
    
            self.build()
    
        if self.bm25 is None:
            logger.warning(
                "BM25 index could not be built — returning no results"
            )
            return []
    
        if not self.validate_alignment():
            logger.warning(
                "Skipping BM25 search due to index misalignment"
            )
            return []
    
        if not isinstance(query, str):
            raise TypeError("BM25 query must be a string")
    
        query = query.strip()
    
        if not query:
            raise ValueError("BM25 query must not be empty")
    
        if top_k <= 0:
            return []
    
        query_tokens = self._tokenize(query)
    
        if not query_tokens:
            return []
    
        scores = self.bm25.get_scores(query_tokens)
    
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]
    
        results = []
    
        for index in ranked_indices:
            results.append(
                {
                    "index_id": index,
                    "score": float(scores[index]),
                }
            )
    
        return results

    @property
    def total_documents(self) -> int:
        """Return the number of documents indexed by BM25."""

        return self._document_count

    @property
    def is_built(self) -> bool:
        """Return whether the BM25 index is currently built."""

        return self.bm25 is not None


_bm25_store: Optional[BM25Store] = None


def get_bm25_store() -> BM25Store:
    """Return the shared BM25 store singleton."""

    global _bm25_store

    if _bm25_store is None:
        _bm25_store = BM25Store()

    return _bm25_store