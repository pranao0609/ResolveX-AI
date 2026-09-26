"""
ticket_service.py — Business logic for ticket creation, listing, and management.
Calls TicketRepository for DB access and storage for file handling.
"""

import os
from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.repositories.ticket_repo import TicketRepository
from app.models.ticket_model import Ticket
from app.schemas.ticket_schema import TicketUpdate, TicketListResponse, TicketResponse
from app.core.exceptions import TicketNotFoundException
from app.core.logger import logger
from storage.file_manager import FileManager


class TicketService:
    def __init__(self, db: Session):
        self.repo = TicketRepository(db)
        self.file_manager = FileManager()

    async def create_ticket(
        self,
        title: str,
        description: str,
        submitted_by: Optional[str],
        category: str,
        image: Optional[UploadFile] = None,
    ) -> Ticket:
        """
        Validate inputs & attachments, then persist a new ticket.
        """
        from app.core.constants import ALLOWED_EXTENSIONS, CATEGORIES
        from app.core.exceptions import ResolveXException
        from app.config import settings

        # Input validation
        clean_cat = category.strip().lower() if category else ""
        if clean_cat not in CATEGORIES:
            raise ResolveXException(
                status_code=400,
                detail=f"Invalid category '{category}'. Allowed: {', '.join(CATEGORIES)}",
            )

        if not (3 <= len(title.strip()) <= 255):
            raise ResolveXException(
                status_code=400,
                detail="Title length must be between 3 and 255 characters.",
            )

        if not (10 <= len(description.strip()) <= 10000):
            raise ResolveXException(
                status_code=400,
                detail="Description length must be between 10 and 10,000 characters.",
            )

        saved_path = None
        if image and image.filename:
            ext = os.path.splitext(image.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise ResolveXException(
                    status_code=400,
                    detail=f"Unsupported file type '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                )

            contents = await image.read()
            if len(contents) == 0:
                raise ResolveXException(
                    status_code=400, detail="Uploaded attachment file is empty."
                )
            if len(contents) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ResolveXException(
                    status_code=400,
                    detail=f"Attachment file size ({len(contents) / (1024*1024):.2f}MB) exceeds maximum limit of {settings.MAX_FILE_SIZE_MB}MB.",
                )

            # Reset file pointer or write using file_manager
            image.file.seek(0)
            saved_path = await self.file_manager.save(image)

        logger.info(
            f"Creating ticket: '{title}' category='{clean_cat}' attachment='{saved_path}'"
        )

        ticket = Ticket(
            title=title.strip(),
            description=description.strip(),
            submitted_by=submitted_by,
            category=clean_cat,
            attachment_paths=saved_path,
            status="open",
        )
        created = self.repo.create(ticket)
        logger.info(f"Ticket created with id={created.id}")
        return created

    def get_ticket(self, ticket_id: int) -> Ticket:
        """Fetch a single ticket or raise 404."""
        ticket = self.repo.get_by_id(ticket_id)
        if not ticket:
            raise TicketNotFoundException(ticket_id)
        return ticket

    def list_tickets(
        self,
        page: int,
        page_size: int,
        status: Optional[str],
        category: Optional[str],
    ) -> TicketListResponse:
        """Return a paginated list of tickets."""
        skip = (page - 1) * page_size
        tickets = self.repo.list_all(
            skip=skip, limit=page_size, status=status, category=category
        )
        total = self.repo.count(status=status, category=category)
        return TicketListResponse(
            total=total,
            page=page,
            page_size=page_size,
            tickets=[TicketResponse.model_validate(t) for t in tickets],
        )

    def update_ticket(self, ticket_id: int, payload: TicketUpdate) -> Ticket:
        """Apply partial updates to a ticket."""
        ticket = self.get_ticket(ticket_id)
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(ticket, field, value)
        return self.repo.update(ticket)

    def delete_ticket(self, ticket_id: int) -> None:
        """Delete a ticket (hard delete)."""
        ticket = self.get_ticket(ticket_id)
        self.repo.delete(ticket)
