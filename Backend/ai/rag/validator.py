from dataclasses import dataclass, field
from typing import List

from ai.rag.models import CanonicalDocument


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)


class DocumentValidator:
    """
    Domain-level validation for canonical RAG documents.

    Structural validation is handled by Pydantic models.
    This validator checks whether a document is usable by
    the RAG ingestion pipeline.
    """

    @staticmethod
    def validate(document: CanonicalDocument) -> ValidationResult:
        errors: List[str] = []

        # Identity
        if not document.document_id.strip():
            errors.append("document_id must not be empty")

        if not document.source_id.strip():
            errors.append("source_id must not be empty")

        # Content
        if not document.title.strip():
            errors.append("title must not be empty")

        if not document.content.strip():
            errors.append("content must not be empty")

        # Source information
        if not document.source_type.strip():
            errors.append("source_type must not be empty")

        # Version
        if not document.version.strip():
            errors.append("version must not be empty")

        # Timestamp
        if document.created_at is None:
            errors.append("created_at must be provided")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )

    @classmethod
    def validate_all(
        cls,
        documents: List[CanonicalDocument],
    ) -> tuple[List[CanonicalDocument], List[CanonicalDocument]]:
        """
        Validate a collection of documents.

        Returns:
            valid_documents:
                Documents that passed validation.

            invalid_documents:
                Documents that failed validation.
        """

        valid_documents: List[CanonicalDocument] = []
        invalid_documents: List[CanonicalDocument] = []

        for document in documents:
            result = cls.validate(document)

            if result.valid:
                valid_documents.append(document)
            else:
                invalid_documents.append(document)

        return valid_documents, invalid_documents
