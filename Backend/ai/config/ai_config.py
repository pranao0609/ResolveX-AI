"""
ai_config.py — Bridge module re-exporting centralized settings for AI components.
All values derive from app.config.settings (single source of truth).
"""

from app.config import settings

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
EMBEDDING_DIMENSION = settings.EMBEDDING_DIMENSION

# ── FAISS ─────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH
FAISS_DOCSTORE_PATH = settings.FAISS_DOCSTORE_PATH
FAISS_TOP_K = settings.FAISS_TOP_K
FAISS_SCORE_THRESHOLD = settings.FAISS_SCORE_THRESHOLD

# -- Hybrid Retrieval --
RETRIEVAL_STRATEGY = settings.RETRIEVAL_STRATEGY
BM25_WEIGHT = settings.BM25_WEIGHT
DENSE_WEIGHT = settings.DENSE_WEIGHT
RETRIEVAL_TOP_K = settings.RETRIEVAL_TOP_K
RETRIEVAL_CANDIDATE_K = settings.RETRIEVAL_CANDIDATE_K
# ── RAG Chunking ──────────────────────────────────────────────────────────────

RAG_CHUNK_SIZE = settings.RAG_CHUNK_SIZE

RAG_CHUNK_OVERLAP = settings.RAG_CHUNK_OVERLAP

RAG_MIN_CHUNK_SIZE = settings.RAG_MIN_CHUNK_SIZE
# ── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_MODEL = settings.GROQ_MODEL
GROQ_MAX_TOKENS = settings.GROQ_MAX_TOKENS
GROQ_TEMPERATURE = settings.GROQ_TEMPERATURE

# ── Classification ────────────────────────────────────────────────────────────
SUPPORTED_CATEGORIES = ["billing", "technical", "account", "feature_request", "bug_report", "other"]
CLASSIFICATION_CONFIDENCE_THRESHOLD = settings.CLASSIFICATION_CONFIDENCE_THRESHOLD

# ── Confidence weights ───────────────────────────────────────────────────────
CONFIDENCE_WEIGHT_SIMILARITY = settings.CONFIDENCE_WEIGHT_SIMILARITY
CONFIDENCE_WEIGHT_LLM_SCORE = settings.CONFIDENCE_WEIGHT_LLM_SCORE
CONFIDENCE_WEIGHT_CLASSIFICATION = settings.CONFIDENCE_WEIGHT_CLASSIFICATION

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_LANGUAGE = "eng"
OCR_ENABLED = True

