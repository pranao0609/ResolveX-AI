"""
ticket_schema.py — Pydantic schemas for ticket request/response contracts.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List
from datetime import datetime
from app.core.constants import CATEGORIES


class TicketCreate(BaseModel):
    """Payload for creating a new ticket (POST /tickets)."""
    title: str = Field(..., min_length=3, max_length=255, example="Application crashes on login")
    description: str = Field(..., min_length=10, max_length=10000, example="The app throws a 500 error whenever I try to log in.")
    submitted_by: Optional[str] = Field(None, max_length=255, example="user@company.com")
    category: str = Field(..., example="software")
    image: Optional[str] = Field(None, max_length=255, example="screenshot.png")   # filename only; no processing done

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        clean_cat = value.strip().lower()
        if clean_cat not in CATEGORIES:
            raise ValueError(f"Invalid category '{value}'. Allowed categories: {', '.join(CATEGORIES)}")
        return clean_cat


class TicketUpdate(BaseModel):
    """Payload for partial ticket updates."""
    title: Optional[str] = Field(None, min_length=3, max_length=255)
    description: Optional[str] = Field(None, min_length=10, max_length=10000)
    status: Optional[str] = Field(None, max_length=50)
    assigned_to: Optional[str] = Field(None, max_length=255)
    category: Optional[str] = None

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        clean_cat = value.strip().lower()
        if clean_cat not in CATEGORIES:
            raise ValueError(f"Invalid category '{value}'. Allowed categories: {', '.join(CATEGORIES)}")
        return clean_cat


class TicketResponse(BaseModel):
    """Full ticket response returned to the client."""
    id: int
    title: str
    description: str
    category: Optional[str] = None
    status: str
    solution: Optional[str] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    submitted_by: Optional[str] = None
    assigned_to: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)
    assigned_resolver_id: Optional[str] = None
    assigned_resolver_name: Optional[str] = None
    assigned_resolver_category: Optional[str] = None
    assigned_at: Optional[datetime] = None
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class PipelineStep(BaseModel):
    key: str
    label: str
    state: str
    timestamp: Optional[str] = None
    details: Optional[dict] = None
    model_config = ConfigDict(from_attributes=True)

class TicketPipelineTracker(BaseModel):
    ticket_id: int
    current_step: str
    steps: List[PipelineStep]

    class Config:
        from_attributes = True   # replaces orm_mode in Pydantic v2


class TicketListResponse(BaseModel):
    """Paginated list of tickets."""
    total: int
    page: int
    page_size: int
    tickets: list[TicketResponse]
    model_config = ConfigDict(from_attributes=True)