from typing import List

import numpy as np

from app.core.logger import logger
from ai.rag.doc_store import get_doc_store
from ai.rag.embedder import RAGEmbedder
from ai.rag.models import DocumentChunk
from ai.rag.vector_store import get_vector_store


class RAGIndexWriter:
    """
    Coordinates embedding generation, FAISS indexing,
    and chunk metadata registration.

    FAISS vector positions and DocumentStore index IDs
    must remain strictly aligned.
    """

    @classmethod
    def index_chunks(cls, chunks: List[DocumentChunk]) -> List[int]:
        """
        Embed and index a batch of document chunks.

        If metadata registration fails after vectors have been
        added to FAISS, both stores are rolled back to their
        original state.

        Returns:
            List of FAISS/DocumentStore index IDs.
        """

        if not chunks:
            logger.info("No chunks supplied for indexing")
            return []

        embeddings = RAGEmbedder.embed(chunks)

        cls._validate_alignment(
            chunks=chunks,
            embeddings=embeddings,
        )

        vector_store = get_vector_store()
        doc_store = get_doc_store()

        initial_vector_count = vector_store.total_vectors
        initial_document_count = doc_store.total_chunks

        if initial_vector_count != initial_document_count:
            raise RuntimeError(
                "Cannot index chunks because FAISS and DocumentStore "
                "are already misaligned: "
                f"FAISS={initial_vector_count}, "
                f"DocumentStore={initial_document_count}"
            )

        logger.info(
            "Indexing %d chunks starting at FAISS index %d",
            len(chunks),
            initial_vector_count,
        )

        try:
            # ---------------------------------------------------------
            # 1. Add vectors to FAISS
            # ---------------------------------------------------------
            vector_store.add(embeddings)

            # ---------------------------------------------------------
            # 2. Add corresponding metadata
            # ---------------------------------------------------------
            index_ids = []

            for chunk in chunks:
                metadata = cls._build_metadata(chunk)

                index_id = doc_store.add_chunk(metadata)

                expected_index = initial_document_count + len(index_ids)

                if index_id != expected_index:
                    raise RuntimeError(
                        "FAISS/DocumentStore index alignment mismatch: "
                        f"expected {expected_index}, "
                        f"got {index_id}"
                    )

                index_ids.append(index_id)

            # ---------------------------------------------------------
            # 3. Final consistency check
            # ---------------------------------------------------------
            final_vector_count = vector_store.total_vectors
            final_document_count = doc_store.total_chunks

            expected_count = initial_vector_count + len(chunks)

            if final_vector_count != expected_count:
                raise RuntimeError(
                    "Unexpected FAISS vector count after indexing: "
                    f"expected {expected_count}, "
                    f"got {final_vector_count}"
                )

            if final_document_count != expected_count:
                raise RuntimeError(
                    "Unexpected DocumentStore count after indexing: "
                    f"expected {expected_count}, "
                    f"got {final_document_count}"
                )

            logger.info(
                "Successfully indexed %d chunks",
                len(index_ids),
            )

            return index_ids

        except Exception as exc:
            logger.error(
                "RAG indexing failed; rolling back batch: %s",
                exc,
            )

            # ---------------------------------------------------------
            # Roll back DocumentStore
            # ---------------------------------------------------------
            if doc_store.total_chunks > initial_document_count:
                del doc_store.documents[initial_document_count:]

            # ---------------------------------------------------------
            # Roll back FAISS
            # ---------------------------------------------------------
            if vector_store.total_vectors > initial_vector_count:
                try:
                    vector_store.rollback_to(initial_vector_count)
                except Exception as rollback_exc:
                    logger.critical(
                        "FAISS rollback failed: %s",
                        rollback_exc,
                    )
                    raise RuntimeError(
                        "RAG indexing failed and FAISS rollback " "also failed"
                    ) from rollback_exc

            raise

    @staticmethod
    def _build_metadata(chunk: DocumentChunk):
        """
        Convert a DocumentChunk into ChunkMetadata.
        """

        from ai.rag.metadata import MetadataBuilder

        return MetadataBuilder.build(chunk)

    @staticmethod
    def _validate_alignment(
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Validate that the embedding batch corresponds exactly
        to the supplied chunks.
        """

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "Chunk/embedding count mismatch: "
                f"{len(chunks)} chunks vs "
                f"{embeddings.shape[0]} embeddings"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or infinite values")
