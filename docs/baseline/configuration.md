# ResolveX-AI — Baseline Configuration & Secrets Mapping

## 1. Application Configuration Settings

Configuration is loaded in [Backend/app/config.py](file:///d:/ResolveX-AI/Backend/app/config.py) using Pydantic `BaseSettings`. Variables are populated from environment variables or `Backend/.env`.

> [!CAUTION]
> All passwords, API keys, and sensitive tokens are **REDACTED** in this document. Never commit unredacted credentials to documentation files.

### Backend Environment Configuration ([Backend/.env](file:///d:/ResolveX-AI/Backend/.env))

| Variable Name | Type | Default Value | Description | Secret? |
| ------------- | ---- | ------------- | ----------- | ------- |
| `APP_NAME` | str | `"ResolveX-AI"` | Application display name | No |
| `APP_VERSION` | str | `"1.0.0"` | Application semantic version | No |
| `DEBUG` | bool | `True` | Debug logging and exception mode | No |
| `RDS_HOST` | str | `resolvex-ai...rds.amazonaws.com` | AWS RDS PostgreSQL hostname | No |
| `RDS_PORT` | int | `5432` | PostgreSQL port | No |
| `RDS_DB` | str | `"postgres"` | PostgreSQL database name | No |
| `RDS_USER` | str | `"postgres"` | PostgreSQL username | No |
| `RDS_PASSWORD` | str | **REDACTED** | PostgreSQL password | **YES** |
| `RDS_SSL` | bool | `False` | Require SSL connection to database | No |
| `GROQ_API_KEY` | str | **REDACTED** | API Key for Groq Cloud LLM | **YES** |
| `GROQ_MODEL` | str | `"llama-3.3-70b-versatile"` | Target Groq LLM model ID | No |
| `AUTO_RESOLVE_THRESHOLD` | float | `0.75` | Confidence threshold for auto-resolution | No |
| `HITL_THRESHOLD` | float | `0.50` | Confidence threshold for human review | No |
| `UPLOAD_DIR` | str | `"uploads"` | Local directory for file attachments | No |
| `MAX_FILE_SIZE_MB` | int | `10` | Maximum allowed attachment file size (MB) | No |
| `FAISS_INDEX_PATH` | str | `"storage/faiss_index.bin"` | Path to backup FAISS index file | No |

---

### AI Hyperparameter Configuration ([Backend/ai/config/ai_config.py](file:///d:/ResolveX-AI/Backend/ai/config/ai_config.py))

| Constant Name | Value | Description |
| ------------- | ----- | ----------- |
| `EMBEDDING_MODEL_NAME` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace embedding model ID |
| `EMBEDDING_DIMENSION` | `384` | Embedding vector output size |
| `FAISS_INDEX_PATH` | `"data/faiss/index.faiss"` | Primary FAISS index file location |
| `FAISS_DOCSTORE_PATH` | `"data/faiss/docstore.json"` | RAG document metadata JSON file |
| `FAISS_TOP_K` | `5` | Number of documents retrieved during RAG search |
| `FAISS_SCORE_THRESHOLD` | `0.35` | Minimum inner-product similarity score cutoff |
| `GROQ_MODEL` | `"llama-3.3-70b-versatile"` | Groq LLM model name |
| `GROQ_MAX_TOKENS` | `1024` | Maximum tokens for generated solution |
| `GROQ_TEMPERATURE` | `0.3` | LLM sampling temperature |
| `CONFIDENCE_WEIGHT_SIMILARITY` | `0.4` | Weight of RAG similarity in confidence formula |
| `CONFIDENCE_WEIGHT_LLM_SCORE` | `0.3` | Weight of LLM score proxy in confidence formula |
| `CONFIDENCE_WEIGHT_CLASSIFICATION` | `0.3` | Weight of classifier score in confidence formula |

---

### Frontend Environment Configuration ([Frontend/.env](file:///d:/ResolveX-AI/Frontend/.env))

| Variable Name | Type | Value | Description |
| ------------- | ---- | ----- | ----------- |
| `VITE_API_URL` | str | `"http://localhost:8000/api"` | Base API URL prefix for backend REST requests |
