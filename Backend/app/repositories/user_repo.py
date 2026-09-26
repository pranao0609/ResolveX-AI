"""
user_repo.py — Repository layer for User database operations.
"""

from sqlalchemy.orm import Session
from typing import Optional, List

from app.models.user_model import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, user: User) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_by_id(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def list_agents(self) -> List[User]:
        return (
            self.db.query(User)
            .filter(User.role == "agent", User.is_active == True)
            .all()
        )
