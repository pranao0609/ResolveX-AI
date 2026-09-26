"""
common_schema.py — Shared Pydantic schemas used across multiple modules.
"""

from pydantic import BaseModel
from typing import Optional


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str
    success: bool = True


class PaginationParams(BaseModel):
    """Common pagination parameters."""

    page: int = 1
    page_size: int = 20


class IDResponse(BaseModel):
    """Response containing just an ID."""

    id: int
    message: Optional[str] = None
