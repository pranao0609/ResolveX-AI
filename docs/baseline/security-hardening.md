# ResolveX-AI — Phase 2 Security, Configuration & AI Runtime Hardening

## 1. Executive Summary

Phase 2 of **ResolveX 2.0** focuses on hardening secret management, environment configuration, input validation, file upload safety, LLM API call reliability, error protection, and request observability.

In accordance with the project scope decisions, **enterprise IAM/authentication (JWT, OAuth, RBAC) was intentionally deferred** to keep Phase 2 focused on core AI system reliability and runtime security.

All Phase 2 hardening changes have been validated using the automated verification suite [Backend/scripts/verify_phase2_security.py](file:///d:/ResolveX-AI/Backend/scripts/verify_phase2_security.py).

---

## 2. Implemented Hardening Improvements

### 2.1 Secret Management & Configuration Template
* Created [Backend/.env.example](file:///d:/ResolveX-AI/Backend/.env.example) containing safe variable name placeholders for all backend options (`ENVIRONMENT`, `LOG_LEVEL`, `CORS_ORIGINS`, `RDS_*`, `GROQ_*`, `AUTO_RESOLVE_THRESHOLD`, `HITL_THRESHOLD`, `UPLOAD_DIR`, `MAX_FILE_SIZE_MB`, `FAISS_INDEX_PATH`).
* Verified `.gitignore` tracking rules ensuring `.env` files are excluded from source control.
* Audit Finding on Leaked Secrets:
  * Secret Detected: YES (in local `Backend/.env`)
  * Groq API Key: REDACTED
  * Database Password: REDACTED
  * Manual Rotation Required: YES (Groq API Key and PostgreSQL password require manual provider rotation).

### 2.2 Centralized Configuration & Startup Validation
* **File**: [Backend/app/config.py](file:///d:/ResolveX-AI/Backend/app/config.py)
* Added `ENVIRONMENT`, `LOG_LEVEL`, `CORS_ORIGINS`, `LLM_TIMEOUT_SECONDS`, and `LLM_MAX_RETRIES` to `Settings(BaseSettings)`.
* Added `validate_critical_settings()` method enforcing:
  * $0.0 \le \text{AUTO\_RESOLVE\_THRESHOLD} \le 1.0$
  * $0.0 \le \text{HITL\_THRESHOLD} \le \text{AUTO\_RESOLVE\_THRESHOLD}$
  * $\text{MAX\_FILE\_SIZE\_MB} > 0$
* **Startup Validation**: In [Backend/app/main.py](file:///d:/ResolveX-AI/Backend/app/main.py#L22), the application validates all configuration bounds at startup and logs safe status flags without revealing secret values (e.g. `Groq API Key configured: True`).

### 2.3 CORS Hardening
* **File**: [Backend/app/main.py](file:///d:/ResolveX-AI/Backend/app/main.py#L48)
* Replaced wildcard CORS `allow_origins=["*"]` with computed configuration property `settings.parsed_cors_origins`.
* Configurable via environment variable: `CORS_ORIGINS=http://localhost:5173,http://localhost:3000`.

### 2.4 Input Validation & Schema Hardening
* **File**: [Backend/app/schemas/ticket_schema.py](file:///d:/ResolveX-AI/Backend/app/schemas/ticket_schema.py)
* Enforced input bounds:
  * `title`: Minimum length 3, maximum length 255 characters.
  * `description`: Minimum length 10, maximum length 10,000 characters.
  * `category`: Validated against standard taxonomy (`software`, `hardware`, `network`, `access_permission`, `security`, `other`) using Pydantic `@field_validator`.

### 2.5 File Upload Hardening
* **File**: [Backend/app/services/ticket_service.py](file:///d:/ResolveX-AI/Backend/app/services/ticket_service.py#L24)
* Validates attachment file extension against `ALLOWED_EXTENSIONS` (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`, `.log`). Raises HTTP 400 on invalid types.
* Validates attachment size against `settings.MAX_FILE_SIZE_MB * 1024 * 1024`. Raises HTTP 400 if oversized.
* Rejects 0-byte empty files.
* Preserves UUID-based filename generation in [Backend/storage/local_storage.py](file:///d:/ResolveX-AI/Backend/storage/local_storage.py) to prevent path traversal attacks.

### 2.6 LLM API Runtime Hardening
* **File**: [Backend/ai/llm/solution_generator.py](file:///d:/ResolveX-AI/Backend/ai/llm/solution_generator.py) & [Backend/ai/classification/classifier.py](file:///d:/ResolveX-AI/Backend/ai/classification/classifier.py)
* **Timeout & Retries**: Configured Groq API client with explicit timeout (`settings.LLM_TIMEOUT_SECONDS = 15.0`) and exponential backoff retry loop up to `settings.LLM_MAX_RETRIES = 2`.
* **Prompt Input Isolation**: User ticket descriptions are truncated to 4,000 characters and explicitly wrapped inside markdown code block boundaries (```ticket ... ``` and ```context ... ```) to prevent prompt formatting injection.
* **Failure Tracking**: `generate_solution` returns `fallback_used: bool` and logs model latency. [Backend/ai/pipeline/ticket_pipeline.py](file:///d:/ResolveX-AI/Backend/ai/pipeline/ticket_pipeline.py#L90) captures and returns `fallback_used` in pipeline metadata.

### 2.7 Request Correlation ID & Production Error Sanitization
* **File**: [Backend/app/main.py](file:///d:/ResolveX-AI/Backend/app/main.py#L43)
* Added `X-Request-ID` middleware assigning a unique UUID to every incoming HTTP request and injecting it into response headers and exception logs.
* In production (`ENVIRONMENT=production`), generic exception responses are sanitized to `{"detail": "Internal server error", "request_id": req_id}` to prevent stack trace or database credential exposure. Server-side log files retain full diagnostic traces.

---

## 3. Security Audit & Dependency Scanning

### Frontend Package Audit (`npm audit`)
* **Commands Run**: `npm audit` inside `Frontend/`.
* **Vulnerabilities Found**: 12 vulnerabilities (1 low, 2 moderate, 9 high in dev dependencies: `vite`, `postcss`, `nanoid`, `react-router-dom`).
* **Fix Action**: Dev dependencies can be updated via `npm audit fix` in Phase 3 testing/build modernization. Production bundles are unaffected.

---

## 4. Verification & Regression Suite

* **Script**: [Backend/scripts/verify_phase2_security.py](file:///d:/ResolveX-AI/Backend/scripts/verify_phase2_security.py)
* **Tests Included**:
  1. Configuration bounds & CORS list parsing.
  2. Schema validation (invalid category and short description rejection).
  3. AI prompt input boundary isolation.
  4. LLM API fallback & timeout resilience.
  5. FastAPI Router lifecycle & middleware loading.
* **Result**: `=== [PASSED] All Phase 2 Security Regression Checks PASSED ===`.

---

## 5. Explicitly Deferred Work

Per Phase 2 scope design, the following items were intentionally deferred to later phases:

* **JWT Authentication**: Deferred to Production Engineering phase.
* **OAuth & User Roles**: Deferred to Production Engineering phase.
* **Enterprise IAM & Session Management**: Deferred.
* **LangGraph / Agent Architecture**: Deferred to Phase 8 (Agentic System).
* **BM25 / Reranking / Hybrid Search**: Deferred to Phase 5 (Advanced RAG).
* **MLflow / DVC / OpenTelemetry**: Deferred to MLOps/LLMOps phases.
