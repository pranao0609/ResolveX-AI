"""
config.py — Application configuration using Pydantic Settings.
Reads settings from environment variables or .env file.
For AWS RDS, populate the RDS* variables; DATABASE_URL is built automatically.
"""

from pydantic_settings import BaseSettings
from pydantic import computed_field
from functools import lru_cache


class Settings(BaseSettings):
    # -- App --
    APP_NAME: str = "ResolveX-AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # -- Security & CORS --
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @computed_field
    @property
    def parsed_cors_origins(self) -> list[str]:
        """Parse comma-separated CORS_ORIGINS into a list of cleaned origin URLs."""
        if not self.CORS_ORIGINS:
            return ["http://localhost:5173"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # -- AWS RDS PostgreSQL --
    # Set these in your .env file (copy from .env.example)
    RDS_HOST: str = "localhost"
    RDS_PORT: int = 5432
    RDS_DB: str = "postgres"
    RDS_USER: str = "postgres"
    RDS_PASSWORD: str = "postgres"
    RDS_SSL: bool = False
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """Build the asyncpg-compatible PostgreSQL DSN from RDS fields."""
        import urllib.parse
        quoted_password = urllib.parse.quote_plus(self.RDS_PASSWORD)
        return (
            f"postgresql+psycopg2://{self.RDS_USER}:{quoted_password}"
            f"@{self.RDS_HOST}:{self.RDS_PORT}/{self.RDS_DB}"
        )

    # -- Groq LLM --
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TEMPERATURE: float = 0.3
    LLM_TIMEOUT_SECONDS: float = 15.0
    LLM_MAX_RETRIES: int = 2

    # -- Embedding --
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # -- Confidence Thresholds & Weights --
    AUTO_RESOLVE_THRESHOLD: float = 0.75  # confidence >= this → auto-resolve
    HITL_THRESHOLD: float = 0.50          # confidence < this → escalate to human
    CONFIDENCE_WEIGHT_SIMILARITY: float = 0.4
    CONFIDENCE_WEIGHT_LLM_SCORE: float = 0.3
    CONFIDENCE_WEIGHT_CLASSIFICATION: float = 0.3

    # -- Classification --
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.6

    # -- Storage --
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 10
    # -- FAISS & DocStore --
    FAISS_INDEX_PATH: str = "data/faiss/index.faiss"
    FAISS_DOCSTORE_PATH: str = "data/faiss/docstore.json"
    FAISS_TOP_K: int = 5
    FAISS_SCORE_THRESHOLD: float = 0.35

    # -- Hybrid Retrieval --
    RETRIEVAL_STRATEGY: str = "hybrid"
    BM25_WEIGHT: float = 0.0
    DENSE_WEIGHT: float = 1.0
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_CANDIDATE_K: int = 20

    # -- RAG Chunking --
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 120
    RAG_MIN_CHUNK_SIZE: int = 100

    LLM_RATE_LIMIT_REQUESTS: int = 60
    LLM_RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    from pydantic import model_validator

    @model_validator(mode="after")
    def validate_confidence_weights(self):
        weights_sum = round(
            self.CONFIDENCE_WEIGHT_SIMILARITY + self.CONFIDENCE_WEIGHT_LLM_SCORE + self.CONFIDENCE_WEIGHT_CLASSIFICATION, 4
        )
        if weights_sum != 1.0:
            raise ValueError(f"Confidence weights must sum to 1.0 (got {weights_sum})")
        return self

    def validate_critical_settings(self) -> None:
        """Validate critical configuration bounds without printing secret values."""
        if not (0.0 <= self.AUTO_RESOLVE_THRESHOLD <= 1.0):
            raise ValueError(f"AUTO_RESOLVE_THRESHOLD must be between 0.0 and 1.0 (got {self.AUTO_RESOLVE_THRESHOLD})")
        if not (0.0 <= self.HITL_THRESHOLD <= 1.0):
            raise ValueError(f"HITL_THRESHOLD must be between 0.0 and 1.0 (got {self.HITL_THRESHOLD})")
        if self.HITL_THRESHOLD > self.AUTO_RESOLVE_THRESHOLD:
            raise ValueError(f"HITL_THRESHOLD ({self.HITL_THRESHOLD}) cannot exceed AUTO_RESOLVE_THRESHOLD ({self.AUTO_RESOLVE_THRESHOLD})")
        if self.MAX_FILE_SIZE_MB <= 0:
            raise ValueError(f"MAX_FILE_SIZE_MB must be positive (got {self.MAX_FILE_SIZE_MB})")
        if self.RAG_CHUNK_SIZE <= 0:
            raise ValueError(
                f"RAG_CHUNK_SIZE must be positive "
                f"(got {self.RAG_CHUNK_SIZE})"
            )
        if self.RAG_CHUNK_OVERLAP < 0:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP cannot be negative "
                f"(got {self.RAG_CHUNK_OVERLAP})"
            )

        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be smaller than "
                f"RAG_CHUNK_SIZE "
                f"(got overlap={self.RAG_CHUNK_OVERLAP}, "
                f"size={self.RAG_CHUNK_SIZE})"
            )

        if self.RAG_MIN_CHUNK_SIZE <= 0:
            raise ValueError(
                f"RAG_MIN_CHUNK_SIZE must be positive "
                f"(got {self.RAG_MIN_CHUNK_SIZE})"
            )

        if self.RAG_MIN_CHUNK_SIZE > self.RAG_CHUNK_SIZE:
            raise ValueError(
                "RAG_MIN_CHUNK_SIZE cannot exceed "
                f"RAG_CHUNK_SIZE "
                f"(got min={self.RAG_MIN_CHUNK_SIZE}, "
                f"size={self.RAG_CHUNK_SIZE})"
            )
        if self.RETRIEVAL_STRATEGY not in {
            "dense",
            "bm25",
            "hybrid",
        }:
            raise ValueError(
                "RETRIEVAL_STRATEGY must be one of "
                "'dense', 'bm25', or 'hybrid' "
                f"(got {self.RETRIEVAL_STRATEGY!r})"
            )

        if self.BM25_WEIGHT < 0.0:
            raise ValueError(
                f"BM25_WEIGHT cannot be negative "
                f"(got {self.BM25_WEIGHT})"
            )

        if self.DENSE_WEIGHT < 0.0:
            raise ValueError(
                f"DENSE_WEIGHT cannot be negative "
                f"(got {self.DENSE_WEIGHT})"
            )

        if (
            self.BM25_WEIGHT == 0.0
            and self.DENSE_WEIGHT == 0.0
        ):
            raise ValueError(
                "BM25_WEIGHT and DENSE_WEIGHT "
                "cannot both be zero"
            )

        if self.RETRIEVAL_TOP_K <= 0:
            raise ValueError(
                f"RETRIEVAL_TOP_K must be positive "
                f"(got {self.RETRIEVAL_TOP_K})"
            )

        if self.RETRIEVAL_CANDIDATE_K <= 0:
            raise ValueError(
                f"RETRIEVAL_CANDIDATE_K must be positive "
                f"(got {self.RETRIEVAL_CANDIDATE_K})"
            )

        if self.RETRIEVAL_CANDIDATE_K < self.RETRIEVAL_TOP_K:
            raise ValueError(
                "RETRIEVAL_CANDIDATE_K cannot be smaller "
                "than RETRIEVAL_TOP_K "
                f"(got candidate_k={self.RETRIEVAL_CANDIDATE_K}, "
                f"top_k={self.RETRIEVAL_TOP_K})"
            )
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


settings = get_settings()
