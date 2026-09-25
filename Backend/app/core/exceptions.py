"""
exceptions.py — Custom exception classes for ResolveX-AI.
"""

from fastapi import HTTPException


class ResolveXException(HTTPException):
    """Base exception for all domain-specific errors."""
    def __init__(self, status_code: int = 500, detail: str = "An error occurred"):
        super().__init__(status_code=status_code, detail=detail)


class TicketNotFoundException(ResolveXException):
    def __init__(self, ticket_id: int):
        super().__init__(status_code=404, detail=f"Ticket with id={ticket_id} not found")


class PipelineException(ResolveXException):
    def __init__(self, detail: str = "AI pipeline failed"):
        super().__init__(status_code=500, detail=detail)


class StorageException(ResolveXException):
    def __init__(self, detail: str = "File storage error"):
        super().__init__(status_code=500, detail=detail)


class ValidationException(ResolveXException):
    def __init__(self, detail: str = "Validation error"):
        super().__init__(status_code=422, detail=detail)


class VectorIndexException(ResolveXException):
    def __init__(self, detail: str = "FAISS vector index error"):
        super().__init__(status_code=500, detail=detail)


class RetrievalException(ResolveXException):
    def __init__(self, detail: str = "RAG context retrieval failure"):
        super().__init__(status_code=500, detail=detail)


class LLMServiceException(ResolveXException):
    def __init__(self, detail: str = "LLM resolution service failure"):
        super().__init__(status_code=500, detail=detail)

