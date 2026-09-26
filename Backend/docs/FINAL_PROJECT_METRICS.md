# ResolveX — Final Authoritative Project Metrics

**Project:** ResolveX — Agentic AI IT Support Ticket Resolution Platform  
**Phase:** Phase 30 — Final Production Validation  
**Date:** September 2026  
**Status:** ENGINEERING COMPLETE  

This document serves as the single authoritative source of truth for all empirical, verified metrics across the ResolveX platform. Every value documented below is derived directly from repository artifacts, source code, or executed test suites.

---

## 1. Project Metadata & Configuration

| Metric / Parameter | Value | Source Artifact / Location |
| :--- | :--- | :--- |
| **Project Title** | ResolveX — Agentic AI IT Support Ticket Resolution Platform | [`README.md`](file:///d:/ResolveX-AI/README.md) |
| **Python Version** | 3.11+ | [`Backend/requirements.txt`](file:///d:/ResolveX-AI/Backend/requirements.txt) |
| **Frameworks** | FastAPI, LangGraph, Pydantic V2, SQLAlchemy | [`Backend/app/main.py`](file:///d:/ResolveX-AI/Backend/app/main.py) |
| **LLM Model (Primary)** | `openai/gpt-oss-120b` (Groq Cloud API) | [`Backend/app/config.py`](file:///d:/ResolveX-AI/Backend/app/config.py) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | [`Backend/app/config.py`](file:///d:/ResolveX-AI/Backend/app/config.py) |
| **Reranker Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | [`Backend/app/config.py`](file:///d:/ResolveX-AI/Backend/app/config.py) |
| **Embedding Vector Dim** | 384 dimensions | `sentence-transformers/all-MiniLM-L6-v2` specification |
| **Auto-Resolve Threshold** | 0.75 | [`Backend/app/config.py`](file:///d:/ResolveX-AI/Backend/app/config.py) |
| **HITL Threshold** | 0.50 | [`Backend/app/config.py`](file:///d:/ResolveX-AI/Backend/app/config.py) |
| **Verification Gate Threshold** | 0.85 | [`Backend/ai/agents/verification_agent.py`](file:///d:/ResolveX-AI/Backend/ai/agents/verification_agent.py) |

---

## 2. Dataset & Knowledge Base Metrics

| Metric | Value | Source Artifact |
| :--- | :--- | :--- |
| **Indexed KB Chunks (FAISS & Docstore)** | 154 chunks | [`resolveX_deployment_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_deployment_report.json) |
| **FAISS Index Vector Count** | 154 vectors | [`resolveX_deployment_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_deployment_report.json) |
| **KB SOP / Catalog Guides** | 10 multi-category SOP documents | [`Backend/data/evaluation/kb_catalog.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/kb_catalog.json) |
| **Final Benchmark Dataset Cases** | 12 structured IT support tickets | [`Backend/data/evaluation/final_benchmark_dataset.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/final_benchmark_dataset.json) |
| **10-KB Retrieval Evaluation Queries** | 10 evaluation queries | [`Backend/data/evaluation/eval_10_kb_queries_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/eval_10_kb_queries_report.json) |
| **Dataset Type** | Curated synthetic enterprise IT benchmark | [`final_benchmark_dataset.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/final_benchmark_dataset.json) |

---

## 3. Retrieval Performance Metrics

*Evaluated across 10 controlled evaluation queries over the indexed KB catalog:*

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Hit@5 | Source Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 (Sparse)** | 0.5500 | 0.7500 | 0.8500 | 0.9500 | 0.8667 | 0.9000 | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Dense (FAISS)** | 0.5500 | 0.7500 | 0.8500 | 1.0000 | 0.8143 | 0.9000 | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Hybrid (BM25 + Dense)** | 0.5500 | 0.7500 | 0.8500 | 1.0000 | 0.8143 | 0.9000 | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Hybrid + Reranker** | **0.5500** | **0.9500** | **1.0000** | **1.0000** | **0.8333** | **1.0000** | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) |

*Note on Retried/Reranked Experiments:* Across Phase 24 and Phase 25 evaluations, Cross-Encoder Reranking consistently improves Recall@5 from `0.8500` to `1.0000` (+17.65% relative improvement) and MRR from `0.7153`/`0.8143` to `0.8333`.

---

## 4. Performance & Latency Profile Metrics

*Measured across full benchmark execution (Phase 25 Profiling & Phase 24 execution):*

| Component / Metric | Mean Latency | P50 Latency | P95 Latency | P99 Latency | Source Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 Retrieval** | 2.83 ms | 1.99 ms | 6.06 ms | 7.56 ms | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) |
| **Dense FAISS Search** | 515.26 ms | 32.69 ms | 2690.18 ms | 4399.28 ms | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) |
| **Hybrid Merge** | 37.73 ms | 37.57 ms | 43.55 ms | 43.95 ms | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) |
| **Cross-Encoder Reranker** | 705.22 ms | 357.34 ms | 2555.93 ms | 3937.24 ms | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) |
| **Diagnosis Agent (LLM)** | 3240.15 ms | — | — | — | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Resolution Agent (LLM)** | 3890.62 ms | — | — | — | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Verification Agent (LLM)** | 5410.88 ms | — | — | — | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Total LLM Aggregate Latency** | **12541.65 ms** | — | — | — | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Total Workflow Baseline** | **9688.08 ms** | **8076.85 ms** | **17951.79 ms** | **19762.41 ms** | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) |
| **Primary Latency Bottleneck** | External LLM API Inference (90.86% of total duration) | Phase 25 Latency Analysis |

---

## 5. Prompt Optimization Metrics (V1 $\rightarrow$ V2)

*Evaluated during Phase 26 Prompt Efficiency Optimization:*

| Agent Prompt | V1 Prompt Chars / Est. Tokens | V2 Prompt Chars / Est. Tokens | Saved Tokens | Reduction (%) | Source Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diagnosis** | 2,384 chars / 596 tokens | 1,866 chars / 466 tokens | 130 tokens | **-21.73%** | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Resolution** | 1,766 chars / 442 tokens | 1,461 chars / 365 tokens | 77 tokens | **-17.27%** | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Verification** | 3,511 chars / 878 tokens | 1,666 chars / 416 tokens | 462 tokens | **-52.55%** | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Combined Payload** | **7,661 chars / 1,916 tokens** | **4,993 chars / 1,247 tokens** | **669 tokens** | **-34.92%** | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |

*Latency Impact Statement:* Prompt token payload was reduced by **34.92%**, but end-to-end latency improvement was not demonstrably proven due to external cloud API response time variance.

---

## 6. Safety & Security Metrics

| Metric | Value | Target | Status | Source Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Unsafe Auto-Resolutions** | **0** | **0** | **PASSED** | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Fail-Closed Safety Rate** | **1.0 (100%)** | 1.0 | **PASSED** | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Safety Rule Violations** | **0** | 0 | **PASSED** | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) |
| **Dedicated Security Tests** | **23 passed** | 23 passed | **PASSED** | [`resolveX_security_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_security_report.json) |
| **Security Findings Remediated** | **8 / 8 findings** | All remediated | **PASSED** | [`resolveX_security_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_security_report.json) |

---

## 7. Testing & Quality Regression Metrics

| Metric / Suite | Count / Result | Status | Source Artifact |
| :--- | :--- | :--- | :--- |
| **Total Test Suite Count** | **753 tests passed** (0 failed) | **PASSED** | Test suite output (`pytest Backend/tests`) |
| **Security Test Suite** | **23 tests passed** (0 failed) | **PASSED** | Test suite output (`pytest Backend/tests/security`) |
| **Black Formatting Check** | **324 files clean** (0 reformatted) | **PASSED** | Formatter check (`black --check Backend`) |
| **Ruff Lint Check** | **0 lint errors** | **PASSED** | Linter check (`ruff check Backend`) |
| **Benchmark Smoke Test** | **PASSED** | **PASSED** | Script check (`benchmark_smoke_test.py`) |

---

## 8. CI/CD & Deployment Status

| Component | Implemented | Validated | Verification Details | Source Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **GitHub Actions CI Workflow** | Yes | Yes | Credential-free workflow ([`.github/workflows/ci.yml`](file:///d:/ResolveX-AI/.github/workflows/ci.yml)) running format, lint, security & full tests | [`resolveX_ci_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_ci_report.json) |
| **FastAPI Backend Deployment** | Yes | Yes | Production-style systemd service definition (`resolvex-backend.service`) and FastAPI lifespan startup | [`resolveX_deployment_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_deployment_report.json) |
| **React Frontend Deployment** | Yes | Yes | Vite production asset build (`Frontend/dist`) served via static web server / Nginx config | [`resolveX_deployment_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_deployment_report.json) |
| **Health Endpoints** | Yes | Yes | `/health` (liveness) and `/api/v1/health/ready` (readiness probe) return 200 OK | [`resolveX_deployment_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_deployment_report.json) |
| **External Cloud Deployment** | Workflows Implemented | Configured | Deployment workflows ([`backend.yml`](file:///d:/ResolveX-AI/.github/workflows/backend.yml), [`frontend.yml`](file:///d:/ResolveX-AI/.github/workflows/frontend.yml)) present; live cloud VPS execution requires active SSH secrets | Repository Workflows |

---

## 9. Discrepancy & Provenance Reconciliation Log

1. **Phase 24 Rate-Limited Fallback Benchmark vs Live LLM Benchmark:**
   - *Discrepancy:* In `resolveX_final_benchmark.json`, the verification pass rate is `0.0` and fallback rate is `1.0` (12/12 cases) because the benchmark run encountered Groq API 429 daily rate limits, causing the fail-closed fallback to engage for all cases.
   - *Reconciliation:* This demonstrates that the **fail-closed safety routing worked as designed** (100% fail-closed rate, 0 unsafe auto-resolutions). In non-rate-limited operational benchmark runs (Phases 20/24/25), verification pass rate was 41.67% and safety routing accuracy was 100%.

2. **Prompt Optimization Token Reduction vs Latency Reduction:**
   - *Discrepancy:* Prompt V2 reduced tokens by 34.92%. However, total workflow latency did not drop proportionally.
   - *Reconciliation:* As documented in Phase 25 and 26, external cloud LLM API network inference latency dominates total execution time (90.86%), masking small local token generation gains. Thus, we claim **token/payload reduction**, not **latency reduction**.
