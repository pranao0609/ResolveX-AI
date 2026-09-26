<div align="center">

# 🚀 ResolveX-AI
**Enterprise Agentic AI IT Support Ticket Resolution Platform**

[![ResolveX CI](https://github.com/pranao0609/ResolveX-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/pranao0609/ResolveX-AI/actions/workflows/ci.yml)
**Demo Video Link:** [https://youtu.be/QMl_igWXLVM](https://youtu.be/QMl_igWXLVM)

</div>

---

## 📖 Overview

**ResolveX-AI** is an enterprise-grade agentic AI IT support ticket resolution platform. It combines **LangGraph** multi-agent orchestration, **Hybrid Retrieval** (Okapi BM25 + FAISS Dense Vector Search), **Cross-Encoder Reranking**, evidence-gated **Verification Gates**, and a **Contextual Bandit Policy Layer** to diagnose, resolve, and safely route incoming IT helpdesk tickets.

If the AI's verification confidence score falls below strict safety thresholds ($\ge 0.85$), or if external rate limits occur, ResolveX triggers a **Human-In-The-Loop (HITL)** protocol, dynamically routing the ticket to `human_review` or `ask_clarification` while preserving complete diagnostic context.

---

## 💥 Problem Statement

Enterprise IT helpdesks suffer from high ticket backlogs, slow MTTR (Mean Time to Resolution), and manual overhead handling repetitive Level-1 requests (password resets, VPN access, account lockouts). Uncontrolled LLM automation introduces severe operational risks:
1. **Hallucinations**: Generating incorrect or dangerous resolution steps.
2. **Security Leakage**: Exposing secrets, passwords, or API keys in model responses.
3. **Improper Auto-Resolution**: Auto-resolving tickets without verified evidence, violating IT compliance.

---

## 💡 Solution

ResolveX solves these challenges by placing deterministic safety guards downstream of multi-agent LLM reasoning:
* **Hybrid RAG & Reranking**: Combines precise keyword lookup (BM25) with semantic vector search (FAISS) and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`).
* **Evidence Verification Gate**: Every proposed resolution is evaluated against retrieved Knowledge Base (KB) facts before auto-resolution eligibility.
* **Fail-Closed Architecture**: Any missing context, low verification confidence, or API rate limit automatically forces safe human review.

---

## ✨ Key Highlights

* **1.0000 Reranked Recall@5**: Cross-Encoder reranking elevates candidate document retrieval to 100% top-5 recall across benchmark cases.
* **0 Unsafe Auto-Resolutions**: Zero safety violations or unverified auto-resolutions across full benchmark evaluations.
* **753 Unit & Integration Tests**: 100% pass rate across backend graph logic, retrieval pipelines, and policy modules.
* **23 Dedicated Security Tests**: Automated verification of log secret redaction, prompt injection isolation, CORS boundaries, and security response headers.
* **34.92% Token Overhead Reduction**: Prompt V2 refactoring cuts token payloads across Diagnosis, Resolution, and Verification agents.

---

## 🏗️ Architecture

ResolveX employs a two-tier microservice model with stateful agentic graph execution:

```mermaid
flowchart TD
    Client[React 19 Frontend SPA] -->|REST / WebSockets| API[FastAPI Application Layer]
    API --> Graph[LangGraph Agentic Orchestrator]
    
    subgraph Agentic Workflow
        Analyzer[Ticket Analyzer] --> Retrieval[Hybrid Retrieval Agent]
        Retrieval --> Reranker[Cross-Encoder Reranker]
        Reranker --> Diagnosis[Diagnosis Agent]
        Diagnosis --> Resolution[Resolution Agent]
        Resolution --> Verification[Verification Gate]
        Verification --> Policy[Policy Decision Agent]
        Policy --> Safety[Deterministic Safety Guard]
    end
    
    Graph --> Agentic Workflow
    
    Safety -->|Confidence >= 0.85 & Clean Safety| Auto[Auto Resolve]
    Safety -->|Missing Info| Clarify[Ask Clarification]
    Safety -->|Low Confidence / Error| Review[Human Review]
    Safety -->|Security Risk| Escalate[Escalate to Admin]
```

For full architectural blueprints, see [`Backend/docs/FINAL_ARCHITECTURE.md`](Backend/docs/FINAL_ARCHITECTURE.md).

---

## 🤖 AI/ML Pipeline

### 1. Hybrid Retrieval & Reranking Pipeline
* **Sparse Index**: Okapi BM25 index over IT Knowledge Base SOPs.
* **Dense Index**: FAISS vector store using 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` embeddings.
* **Reranker**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scoring query-passage relevance.

### 2. Multi-Agent Reasoning Core
* **Ticket Analyzer**: Extracts category, urgency, intent, and entities.
* **Diagnosis Agent**: Identifies technical root cause grounded strictly in retrieved KB evidence.
* **Resolution Agent**: Formulates step-by-step user instructions.
* **Verification Agent**: Validates proposed steps against retrieved KB facts.

---

## 🛡️ Safety & Human-in-the-Loop (HITL)

Safety is enforced via deterministic controls:
1. **Verification Threshold**: Resolution must score $\ge 0.85$ confidence.
2. **Secret Redaction**: Regex and entropy filtering redact passwords, tokens, and API keys.
3. **Fail-Closed Fallback**: LLM timeouts, network failures, or rate limits automatically route tickets to `human_review`.
4. **Unsafe Auto-Resolutions**: **0** (Verified across all test suites).

---

## 📊 Evaluation & Verified Metrics

*Evaluated on the authoritative project benchmark suite:*

| Metric | Result | Source Artifact |
| :--- | :--- | :--- |
| **BM25 Recall@5** | 0.8500 | [`resolveX_final_benchmark.json`](Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Dense FAISS Recall@5** | 0.8500 | [`resolveX_final_benchmark.json`](Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Hybrid Recall@5** | 0.8500 | [`resolveX_final_benchmark.json`](Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Reranked Recall@5** | **1.0000** | [`resolveX_final_benchmark.json`](Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Reranked MRR** | **0.8333** | [`resolveX_final_benchmark.json`](Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Unsafe Auto-Resolutions** | **0** | [`resolveX_final_benchmark.json`](Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Fail-Closed Rate** | **1.0 (100%)** | [`resolveX_final_benchmark.json`](Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Workflow P50 Latency** | **8076.85 ms** | [`resolveX_performance_profile.json`](Backend/data/evaluation/resolveX_performance_profile.json) |
| **Prompt Token Payload Reduction** | **-34.92%** | [`resolveX_llm_optimization.json`](Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Total Test Suite Count** | **753 passed** | `pytest Backend/tests` |
| **Dedicated Security Tests** | **23 passed** | `pytest Backend/tests/security` |

For complete detailed metric breakdowns, see [`Backend/docs/FINAL_PROJECT_METRICS.md`](Backend/docs/FINAL_PROJECT_METRICS.md).

---

## ⚡ Performance Profiling

* **P50 Workflow Latency**: `8076.85 ms`
* **P95 Workflow Latency**: `17951.79 ms`
* **BM25 Retrieval**: `2.83 ms`
* **FAISS Vector Search**: `515.26 ms`
* **Cross-Encoder Reranker**: `705.22 ms`
* **LLM Aggregate Latency**: `12541.65 ms`
* **Bottleneck Analysis**: External cloud LLM API inference accounts for **90.86%** of total workflow duration. Local retrieval and reranking contribute less than 10%.

---

## 🔒 Security Hardening

* **23 Security Tests**: Validated in [`Backend/tests/security`](Backend/tests/security).
* **Remediated Vulnerabilities**: 8 security findings resolved (CORS wildcard credentials fix, HTTP security response headers, log secret redaction, prompt injection boundaries, and stack trace sanitization).

For security verification details, see [`Backend/docs/SECURITY.md`](Backend/docs/SECURITY.md).

---

## 📈 Observability & Experiment Tracking

* **LangSmith**: Integrated graph execution tracing (`Backend/ai/observability/langsmith_config.py`).
* **MLflow**: Experiment metrics logging for retrieval parameters, prompt versions, and safety evaluations (`Backend/ai/observability/mlflow_config.py`).

---

## 🧪 Testing & Quality Gates

```bash
# Code Formatting Check
black --check Backend

# Linting Rules
ruff check Backend

# Security Test Suite (23 Tests)
pytest Backend/tests/security -v

# Full Integration Test Regression (753 Tests)
pytest Backend/tests

# Deterministic Benchmark Smoke Test
python Backend/scripts/benchmark_smoke_test.py
```

---

## 🚀 CI/CD & Deployment

### CI Pipeline
* Automated GitHub Actions CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) executes on pushes and PRs, validating formatting (`Black`), linting (`Ruff`), 23 security tests, 753 full tests, and the benchmark smoke test.

### Deployment Architecture
* **Implemented Configuration**: Production-style single-node deployment setup using Systemd (`resolvex-backend.service`), Uvicorn, and Nginx reverse proxy serving compiled static React assets (`Frontend/dist`).
* **Verified Execution**: `/health` (liveness) and `/api/v1/health/ready` (readiness) probes verified locally.
* **External Deployment Status**: Production deployment workflows ([`backend.yml`](.github/workflows/backend.yml), [`frontend.yml`](.github/workflows/frontend.yml)) are fully implemented; active cloud VPS deployment execution requires configured repository deployment secrets (`EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`).

For deployment guides, see [`Backend/docs/DEPLOYMENT.md`](Backend/docs/DEPLOYMENT.md).

---

## 🛠️ Tech Stack

* **Backend & API**: Python 3.11+, FastAPI, Uvicorn, Pydantic V2, SQLAlchemy ORM
* **Agentic Framework**: LangGraph, LangChain, Groq Cloud API (`openai/gpt-oss-120b`)
* **Vector Store & ML**: FAISS, BM25, `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`
* **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons, Recharts
* **Observability & MLOps**: LangSmith, MLflow, GitHub Actions CI, Systemd, Nginx
* **Testing & Formatting**: pytest, Black, Ruff

---

## 📁 Project Structure

```text
ResolveX-AI/
├── .github/workflows/         # GitHub Actions CI & Deployment Workflows
│   ├── ci.yml                 # Automated Quality & Test Pipeline
│   ├── backend.yml            # Backend Deployment Workflow
│   └── frontend.yml           # Frontend Deployment Workflow
├── Backend/                   # Python FastAPI Backend
│   ├── ai/                    # LangGraph Agents, RAG Pipeline, Policy, Safety
│   │   ├── agents/            # Ticket Analyzer, Retrieval, Diagnosis, Verification
│   │   ├── evaluation/        # Benchmark Engine & JSON/MD Metric Reports
│   │   └── policy/            # Contextual Bandit & Deterministic Safety Guards
│   ├── app/                   # FastAPI Web Server, Routes, Schemas, Core Config
│   ├── data/                  # FAISS Indices, Docstores, Evaluation Datasets
│   ├── docs/                  # Final Architecture, Metrics, & Reports
│   ├── scripts/               # Benchmarking, Ingestion, & Deployment Scripts
│   └── tests/                 # 753 Integration Tests + 23 Security Tests
└── Frontend/                  # React 19 + TypeScript SPA
    ├── src/                   # Components, Pages, WebSockets, API Services
    └── package.json           # Frontend Dependencies
```

---

## ⚙️ Local Setup & Execution

### 1. Environment Setup
```bash
git clone https://github.com/pranao0609/ResolveX-AI.git
cd ResolveX-AI/Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` in `Backend/` and set your API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
AUTO_RESOLVE_THRESHOLD=0.75
HITL_THRESHOLD=0.50
```

### 3. Run Application Server
```bash
uvicorn app.main:app --reload --port 8000
```
*Access swagger documentation at `http://localhost:8000/docs`.*

### 4. Run Frontend Server
```bash
cd ../Frontend
npm install
npm run dev
```

---

## ⚠️ Limitations

1. **Cloud LLM Latency**: Total workflow duration is dominated by external Groq cloud API network response times (P95 = 17.95s).
2. **Single-Node Scale**: Default deployment target is single-node systemd/Nginx rather than multi-region Kubernetes clusters.
3. **Offline Policy Training**: Policy weight updates run via batch replay evaluation rather than real-time online gradient updates.

---

## 🔮 Future Improvements

1. **Self-Hosted LLM Serving**: Transition to local vLLM / Ollama instances on GPU nodes to eliminate external API latency.
2. **Kubernetes Helm Packaging**: Package application into Helm manifests for multi-replica horizontal autoscaling.

---

## 📝 Resume Highlights

* **Architected an Enterprise Agentic RAG Platform** using LangGraph, FastAPI, and React, executing a 12-stage stateful workflow that diagnoses, resolves, and routes IT helpdesk tickets with 0 unsafe auto-resolutions.
* **Engineered a Dual Hybrid Retrieval & Reranking Engine** combining Okapi BM25, FAISS vector search (`all-MiniLM-L6-v2`), and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`), boosting Recall@5 from 0.8500 to 1.0000 and MRR to 0.8333.
* **Implemented Evidence Verification Gates & Security Hardening**, building 23 security tests, automated log secret redaction, prompt injection boundaries, and a 100% fail-closed safety routing architecture.

---

## 📄 License & Author

Developed by **Pranav** ([@pranao0609](https://github.com/pranao0609)).  
ResolveX Phase 30 is **ENGINEERING COMPLETE**. All tests, benchmarks, security gates, and documentation are verified and finalized.
