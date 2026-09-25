# ResolveX-AI — Known Technical Limitations & Gaps Baseline

This document lists all **empirically verified limitations, bugs, and missing capabilities** in the baseline implementation. These limitations are documented here as part of Phase 1 and will be addressed in future ResolveX 2.0 phases.

---

## 1. Security & Authentication Limitations

* **Plaintext Credential Exposure**:
  * **Problem**: [Backend/.env](file:///d:/ResolveX-AI/Backend/.env) is committed to source control with active AWS RDS database passwords and Groq API keys.
  * **Evidence**: File present in working tree with unredacted values.
  * **Affected Component**: Backend environment configuration.
  * **Planned Phase**: Phase 2 (Security & Configuration Hardening).
* **Missing Authentication & Authorization**:
  * **Problem**: All backend API endpoints are unauthenticated and publicly accessible.
  * **Evidence**: No JWT middleware, API key check, or session validator in `app/main.py` or route files.
  * **Planned Phase**: Phase 2 / Production Engineering.
* **Permissive CORS Configuration**:
  * **Problem**: `allow_origins=["*"]` allows cross-origin requests from any website.
  * **Evidence**: Line 49 of [Backend/app/main.py](file:///d:/ResolveX-AI/Backend/app/main.py#L49).
  * **Planned Phase**: Phase 2.

---

## 2. AI, ML & Pipeline Limitations

* **Synchronous PyTorch CPU Blocking**:
  * **Problem**: Machine learning models (`SentenceTransformer` and zero-shot BART `facebook/bart-large-mnli`) run synchronously on CPU inside FastAPI async request handlers.
  * **Evidence**: `classify_ticket` and `generate_embedding` called directly in `run_pipeline` without executor threads or background workers.
  * **Impact**: Server event loop blocks for 1.5 - 3.0 seconds during model execution.
* **Absence of Reinforcement Learning (RL)**:
  * **Problem**: System documentation claims RL routing, but implementation is rule-based keyword matching and round-robin iteration.
  * **Evidence**: [Backend/app/core/expert_resolvers.py](file:///d:/ResolveX-AI/Backend/app/core/expert_resolvers.py).
  * **Planned Phase**: Agentic Architecture / RL.
* **Heuristic LLM Confidence Proxy**:
  * **Problem**: LLM score is calculated based on solution character length: $\min(\text{len(solution)}/1000, 0.95)$.
  * **Evidence**: Line 70 of [Backend/ai/llm/solution_generator.py](file:///d:/ResolveX-AI/Backend/ai/llm/solution_generator.py#L70).
  * **Impact**: Longer responses arbitrarily get higher confidence scores regardless of quality.

---

## 3. RAG Subsystem Limitations

* **No Document Chunking**:
  * **Problem**: Knowledge base articles are embedded as single large strings without sliding windows or semantic splitting.
  * **Evidence**: `ingest_documents` in [Backend/ai/rag/ingestion.py](file:///d:/ResolveX-AI/Backend/ai/rag/ingestion.py).
* **No Hybrid Search or Reranking**:
  * **Problem**: RAG relies solely on dense vector search via FAISS (`IndexFlatIP`). No sparse BM25 keyword search or Cross-Encoder reranking is present.
* **No RAG Evaluation Framework**:
  * **Problem**: No evaluation metrics (Recall@K, MRR, Faithfulness, Context Relevance) are integrated.

---

## 4. Software Bugs & Technical Debt

* **Broken Script `scripts/build_vector_index.py`**:
  * **Problem**: Line 42 invokes `register_document(text)` passing 1 argument, but `register_document` in `ai/rag/retriever.py` expects 4 required arguments (`source`, `title`, `category`, `content`). Running the script crashes with a `TypeError`.
  * **Evidence**: [Backend/scripts/build_vector_index.py:42](file:///d:/ResolveX-AI/Backend/scripts/build_vector_index.py#L42).
* **Missing Database Migrations**:
  * **Problem**: `alembic` package is listed in `requirements.txt`, but no migration folder or migration scripts exist in the repository.
* **Zero Automated Unit/Integration Tests**:
  * **Problem**: No test scripts (`test_*.py` or pytest suites) exist in the entire repository.
