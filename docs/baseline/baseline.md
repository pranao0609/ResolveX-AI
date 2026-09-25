# ResolveX-AI — Baseline Snapshot Specification (Phase 1 Freeze)

## 1. Baseline Snapshot Metadata

| Parameter | Baseline Value | Source / Evidence |
| --------- | -------------- | ----------------- |
| **Baseline Version** | `1.0.0-Phase1-Baseline` | Phase 1 Freeze Target |
| **Snapshot Date** | September 23, 2026 | Audit Execution Date |
| **Git Branch** | `main` | Repository Working Branch |
| **Backend Framework** | FastAPI v0.115+ | `Backend/requirements.txt` |
| **Python Version** | Python 3.10+ | Virtual environment specification |
| **Frontend Framework**| React 19.2 + TypeScript 5.9 + Vite 8.0 | `Frontend/package.json` |
| **Database Engine** | PostgreSQL (AWS RDS) | `Backend/app/database.py` |
| **Database Models** | `Ticket`, `KnowledgeBaseEntry`, `User` | `Backend/app/models/` |
| **Classification Model**| `facebook/bart-large-mnli` (Zero-shot) | `Backend/ai/classification/classifier.py` |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | `Backend/ai/embedding/embedding_model.py` |
| **Embedding Dimensions**| 384 dimensions | `Backend/ai/config/ai_config.py` |
| **Vector Store** | FAISS `IndexFlatIP` (CPU) | `Backend/ai/rag/vector_store.py` |
| **DocStore Format** | JSON Metadata Store | `Backend/data/faiss/docstore.json` |
| **RAG Top-K Search** | 5 documents | `Backend/ai/config/ai_config.py` |
| **RAG Score Cutoff** | 0.35 similarity threshold | `Backend/ai/config/ai_config.py` |
| **LLM Provider** | Groq Cloud API | `Backend/ai/llm/solution_generator.py` |
| **LLM Target Model** | `llama-3.3-70b-versatile` | `Backend/ai/config/ai_config.py` |
| **Auto-Resolve Threshold**| 0.75 (75%) confidence | `Backend/app/config.py` |
| **HITL Threshold** | 0.50 (50%) confidence | `Backend/app/config.py` |
| **Deployment Target** | AWS EC2 (Single instance) via SSH | `.github/workflows/` |

---

## 2. Architecture Summary

ResolveX-AI operates as a synchronous REST API server backed by PostgreSQL, FAISS vector search, Groq LLM, and a React SPA frontend.

```text
User → React SPA → FastAPI REST Routes → Service Layer → 7-Step AI Pipeline → PostgreSQL / FAISS
```

All 7 processing steps (Preprocessing $\rightarrow$ Classification $\rightarrow$ Embedding $\rightarrow$ RAG Search $\rightarrow$ Groq LLM $\rightarrow$ Confidence Scoring $\rightarrow$ Explanation & Expert Assignment) run sequentially on the backend server.

---

## 3. AI Pipeline Summary

1. **Preprocessing**: Cleans raw text using regex and extracts text from attached PDF/image files.
2. **Classification**: 15 regex patterns $\rightarrow$ HuggingFace zero-shot BART $\rightarrow$ Groq LLM fallback.
3. **Embedding**: `all-MiniLM-L6-v2` produces $L_2$-normalized 384-dimensional vectors.
4. **Retrieval**: FAISS `IndexFlatIP` retrieves top 5 KB articles above 0.35 threshold.
5. **LLM Generation**: Groq API (`llama-3.3-70b-versatile`) generates solution markdown text.
6. **Confidence Engine**: Weighted score $C = 0.4 \cdot S_{\text{sim}} + 0.3 \cdot S_{\text{llm}} + 0.3 \cdot S_{\text{cls}}$.
7. **Decision & Routing**: $C \ge 0.75 \rightarrow$ `auto_resolved`; $C < 0.75 \rightarrow$ `escalated` with keyword/round-robin expert assignment.

---

## 4. API Summary

14 REST endpoints exposed under `/api/v1`:
* Tickets: `POST /tickets`, `GET /tickets`, `GET /tickets/{id}`, `GET /tickets/{id}/pipeline`, `PATCH /tickets/{id}`, `DELETE /tickets/{id}`
* Resolution: `POST /resolve/{id}`, `GET /resolve/{id}`, `POST /hitl/review`
* Analytics: `GET /analytics`, `GET /analytics/tickets`, `GET /analytics/confidence`
* Health: `GET /health`, `GET /health/ready`

---

## 5. Database Summary

3 main tables in PostgreSQL:
* `tickets`: 19 columns tracking ticket data, AI solution, confidence, status, and assigned expert resolver.
* `knowledge_base`: 8 columns storing 195 seeded solution articles.
* `users`: 6 columns storing agent and admin user accounts.

---

## 6. Known Limitations Baseline Summary

* **Security**: Hardcoded credentials in `.env`, no API authentication, permissive CORS.
* **AI/ML**: Synchronous CPU execution of PyTorch models, length-heuristic LLM confidence proxy, rule-based expert routing (no RL).
* **RAG**: No document chunking, no BM25 keyword search, no reranking, no automated RAG evaluation.
* **Quality**: 0 unit/integration test files in entire codebase; broken `scripts/build_vector_index.py`.

---

## 7. Baseline Freeze Confirmation

* **Application behavior**: **100% UNCHANGED**
* **Application source code**: **100% UNCHANGED**
* **Documentation**: **ADDED (Phase 1 Baseline Complete)**
* **Readiness for Phase 2**: **READY**
