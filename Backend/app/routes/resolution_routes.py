"""
resolution_routes.py — AI resolution and HITL review endpoints.

Routes:
    POST   /resolve/{ticket_id}   → run AI pipeline on a ticket
    POST   /hitl/review           → human agent submits a manual resolution
    GET    /resolve/{ticket_id}   → get resolution result for a ticket
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.resolution_schema import ResolutionResult, HITLReviewRequest
from app.schemas.common_schema import MessageResponse
from app.services.resolution_service import ResolutionService
from app.services.hitl_service import HITLService

router = APIRouter()


@router.post("/resolve/{ticket_id}", response_model=ResolutionResult)
async def resolve_ticket(
    ticket_id: int, force: bool = False, db: Session = Depends(get_db)
):
    """
    Trigger the AI pipeline for the given ticket.
    Steps: preprocess → classify → embed → RAG → LLM → confidence → decision.
    If confidence >= threshold → auto-resolve; else → escalate to HITL.
    """
    service = ResolutionService(db)
    return await service.resolve(ticket_id=ticket_id, force=force)


@router.get("/resolve/{ticket_id}", response_model=ResolutionResult)
def get_resolution(ticket_id: int, db: Session = Depends(get_db)):
    """Fetch the stored AI resolution result for a ticket."""
    service = ResolutionService(db)
    return service.get_resolution(ticket_id)


@router.post("/hitl/review", response_model=MessageResponse)
def human_review(payload: HITLReviewRequest, db: Session = Depends(get_db)):
    """
    Human agent submits a manual solution for an escalated ticket.
    Marks the ticket as closed and records the agent's solution.
    """
    service = HITLService(db)
    service.submit_review(payload)
    return MessageResponse(message=f"Ticket {payload.ticket_id} reviewed and closed.")
