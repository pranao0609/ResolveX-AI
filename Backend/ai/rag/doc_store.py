"""
doc_store.py — Persistent chunk metadata store for RAG.

Maps:

    FAISS vector index
        ↓
    ChunkMetadata

The store is JSON-backed so FAISS vectors and their metadata
remain aligned across application restarts.
"""

import json
import os
from typing import Dict, List, Optional

from app.core.logger import logger
from ai.config.ai_config import FAISS_DOCSTORE_PATH
from ai.rag.models import ChunkMetadata


class DocumentStore:
    """Persistent JSON-backed chunk metadata store."""

    def __init__(self):
        self.documents: List[Dict] = []
        self._load()

    def _load(self) -> None:
        """Load chunk metadata from disk."""

        if not os.path.exists(FAISS_DOCSTORE_PATH):
            logger.info(
                "No existing doc store found — starting fresh"
            )
            self.documents = []
            return

        try:
            with open(
                FAISS_DOCSTORE_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if not isinstance(data, list):
                logger.warning(
                    "Doc store file invalid — resetting"
                )
                self.documents = []
                return

            self.documents = data

            logger.info(
                "Loaded doc store from %s (%d entries)",
                FAISS_DOCSTORE_PATH,
                len(self.documents),
            )

        except Exception as exc:
            logger.error(
                "Failed to load doc store: %s",
                exc,
            )
            self.documents = []

    def save(self) -> None:
        """Persist chunk metadata to disk."""

        try:
            directory = os.path.dirname(
                FAISS_DOCSTORE_PATH
            )

            if directory:
                os.makedirs(
                    directory,
                    exist_ok=True,
                )

            with open(
                FAISS_DOCSTORE_PATH,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    self.documents,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info(
                "Doc store saved to %s",
                FAISS_DOCSTORE_PATH,
            )

        except Exception as exc:
            logger.error(
                "Failed to save doc store: %s",
                exc,
            )

            raise

    def add_chunk(
        self,
        metadata: ChunkMetadata,
    ) -> int:
        """
        Add chunk metadata and return its FAISS vector index.

        The returned index must correspond exactly to the vector
        position used when the chunk embedding is added to FAISS.
        """

        index_id = len(self.documents)

        document = metadata.model_dump(mode="json")

        document["index_id"] = index_id

        self.documents.append(document)

        return index_id

    def add_chunks(
        self,
        metadata_list: List[ChunkMetadata],
    ) -> List[int]:
        """Add multiple chunk metadata records."""

        index_ids = []

        for metadata in metadata_list:
            index_id = self.add_chunk(metadata)
            index_ids.append(index_id)

        return index_ids

    def get_document(
        self,
        index_id: int,
    ) -> Optional[Dict]:
        """Get metadata by FAISS vector index."""

        if 0 <= index_id < len(self.documents):
            return self.documents[index_id]

        return None

    def get_chunk(
        self,
        index_id: int,
    ) -> Optional[ChunkMetadata]:
        """
        Get structured chunk metadata by FAISS vector index.
        """

        document = self.get_document(index_id)

        if document is None:
            return None

        try:
            document = dict(document)

            # index_id belongs to the FAISS mapping and is not part
            # of the ChunkMetadata model.
            document.pop("index_id", None)

            return ChunkMetadata.model_validate(
                document
            )

        except Exception as exc:
            logger.error(
                "Failed to parse chunk metadata at index %s: %s",
                index_id,
                exc,
            )

            return None

    def clear(self) -> None:
        """Clear all stored chunk metadata."""

        self.documents = []

    @property
    def total_documents(self) -> int:
        """Return the number of stored chunks."""

        return len(self.documents)

    @property
    def total_chunks(self) -> int:
        """Return the number of stored chunks."""

        return len(self.documents)


_doc_store: Optional[DocumentStore] = None


def get_doc_store() -> DocumentStore:
    """Return the shared document-store singleton."""

    global _doc_store

    if _doc_store is None:
        _doc_store = DocumentStore()

    return _doc_store