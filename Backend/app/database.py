"""
database.py — SQLAlchemy engine, session factory, and Base declarative model.
All ORM models inherit from `Base`. Use `get_db` as FastAPI dependency.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.config import settings

# Build SSL connect_args for AWS RDS (psycopg2)
_connect_args = {"sslmode": "require"} if settings.RDS_SSL else {}

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,  # verify connections before use
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,  # recycle connections every 30 min (RDS best-practice)
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency: yields a DB session and ensures it is closed after use.
    Usage:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables in the database (used on startup)."""
    # Import models here so Base is aware of them before create_all
    from app.models import ticket_model, kb_model, user_model  # noqa: F401

    Base.metadata.create_all(bind=engine)
