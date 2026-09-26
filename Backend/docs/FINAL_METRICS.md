# ResolveX 2.0 — Final Consolidated Metrics Reference

This document provides the single authoritative repository record of all empirical performance, retrieval, quality, safety, security, and test metrics verified across ResolveX 2.0.

---

## 1. Evaluation Dataset Metrics

| Metric | Value | Unit | Experiment | Source Artifact | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Test Cases** | `12` | cases | Phase 24 Final Benchmark | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) | Full evaluation suite covering procedural, incident, safety, and clarification ticket categories. |
| **Ground-Truth Cases**| `12` | cases | Phase 24 Final Benchmark | [`resolveX_final_benchmark.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_final_benchmark.json) | 100% human-curated ground-truth benchmark dataset. |
| **Indexed KB Documents**| `154` | documents | KB Expansion Phase | [`docstore.json`](file:///d:/ResolveX-AI/Backend/data/faiss/docstore.json) | Complete knowledge base covering IT support procedures and troubleshooting guides. |

---

## 2. Retrieval Metrics (10-Query & Final Benchmark Sets)

> [!NOTE]
> Retrieval metrics are documented separately for the **Final 12-Case Benchmark** (Phase 24) and the **10-KB Query Evaluation** script (`evaluate_10_kb_queries.py`).

### 2.1 Final Benchmark Retrieval Performance (12 Cases)

| Method | R@1 | R@3 | R@5 | R@10 | MRR | Hit@5 | Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 Keyword Search** | 0.5500 | 0.7500 | 0.8500 | 0.9500 | 0.8667 | 0.9000 | `resolveX_final_benchmark.json` |
| **Dense Vector Search** | 0.5500 | 0.7500 | 0.8500 | 1.0000 | 0.8143 | 0.9000 | `resolveX_final_benchmark.json` |
| **Hybrid Search** | 0.5500 | 0.7500 | 0.8500 | 1.0000 | 0.8143 | 0.9000 | `resolveX_final_benchmark.json` |
| **Hybrid + CrossEncoder Reranker** | **0.5500** | **0.9500** | **1.0000** | **1.0000** | **0.8333** | **1.0000** | `resolveX_final_benchmark.json` |

---

## 3. Reranking Impact Analysis

| Metric | Hybrid Baseline | Hybrid + Reranker | Absolute Delta | Relative Gain | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@5** | `0.8500` | `1.0000` | `+0.1500` | **+17.65%** | `resolveX_final_benchmark.json` |
| **Recall@10** | `1.0000` | `1.0000` | `+0.0000` | `0.00%` | `resolveX_final_benchmark.json` |
| **MRR** | `0.8143` | `0.8333` | `+0.0190` | **+2.33%** | `resolveX_final_benchmark.json` |

*Interpretation: Cross-Encoder reranking successfully elevates relevant evidence into the top-5 context window for 100% of benchmark cases.*

---

## 4. Performance & Latency Profiling (Phase 25)

| Latency Metric | Value | Unit | Experiment | Source Artifact | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Workflow P50** | `8,076.85` | ms | Phase 25 Performance Profiling | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) | Median end-to-end execution time per ticket. |
| **Total Workflow P95** | `17,951.79` | ms | Phase 25 Performance Profiling | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) | 95th percentile execution time. |
| **Total Workflow P99** | `19,762.41` | ms | Phase 25 Performance Profiling | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) | Tail latency under rate-limit retries. |
| **Mean Total Latency** | `9,688.08` | ms | Phase 25 Performance Profiling | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) | Average workflow execution time. |
| **LLM Aggregate Latency** | `12,541.65` | ms | Phase 25 Performance Profiling | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) | **Primary Bottleneck**: Combined API calls across Diagnosis, Resolution, and Verification. |
| **BM25 Search Mean** | `2.83` | ms | Phase 25 Performance Profiling | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) | Ultra-fast local keyword indexing. |
| **FAISS Dense Mean** | `515.26` | ms | Phase 25 Performance Profiling | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) | Local vector embedding & similarity search. |
| **Reranker Mean** | `705.22` | ms | Phase 25 Performance Profiling | [`resolveX_performance_profile.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_performance_profile.json) | Local Cross-Encoder inference. |

---

## 5. Prompt Efficiency & Token Optimization (Phase 26)

| Agent Stage | Prompt V1 Tokens | Prompt V2 Tokens | Saved Tokens | Reduction % | Source Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diagnosis Agent** | `596` | `466` | `130` | **21.73%** | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Resolution Agent** | `442` | `365` | `77` | **17.27%** | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Verification Agent** | `878` | `416` | `462` | **52.55%** | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |
| **Combined Payload** | `1,916` | `1,247` | `669` | **34.92%** | [`resolveX_llm_optimization.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_llm_optimization.json) |

*Note: Prompt payload was significantly reduced (34.92%), but end-to-end latency improvement was not demonstrated due to LLM provider network inference variability.*

---

## 6. Safety & Verification Metrics

| Metric | Value | Unit | Experiment | Source Artifact | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unsafe Auto-Resolutions** | `0` | count | Final Benchmark | `resolveX_final_benchmark.json` | **100% Target Met**: Zero high-risk or destructive actions auto-resolved. |
| **Safety Rule Violations** | `0` | count | Final Benchmark | `resolveX_final_benchmark.json` | Zero safety policy boundary breaches. |
| **Safety Routing Accuracy** | `91.67%` | % | Final Benchmark | `resolveX_final_benchmark.json` | Accurate safety classification and fail-closed routing. |
| **Fail-Closed Rate** | `100.0%` | % | Final Benchmark | `resolveX_final_benchmark.json` | All ambiguous/rate-limited tickets defaulted safely to human review or clarification. |

---

## 7. Security Metrics (Phase 27)

| Metric | Value | Unit | Experiment | Source Artifact | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Audit Findings** | `8` | findings | Phase 27 Audit | [`resolveX_security_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_security_report.json) | Audited across CORS, error leaks, headers, prompts, input validation, and log redaction. |
| **Remediated Findings** | `8` | findings | Phase 27 Audit | [`resolveX_security_report.json`](file:///d:/ResolveX-AI/Backend/data/evaluation/resolveX_security_report.json) | 100% remediation of audit findings. |
| **Security Test Count** | `23` | tests | Security Test Suite | `test_security_*.py` | 23 dedicated unit/integration security tests passing cleanly. |

---

## 8. Testing & CI/CD Metrics (Phase 28 & Phase 29)

| Metric | Value | Unit | Command / Tool | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Total Passed Pytest** | `753` | tests | `pytest Backend/tests` | **PASSED** (0 failures, 78.55s) |
| **Security Pytest** | `23` | tests | `pytest Backend/tests/security` | **PASSED** (0 failures, 0.38s) |
| **Black Formatter Check** | `324` | files | `black --check Backend` | **PASSED** (0 formatting errors) |
| **Ruff Linter Check** | `0` | errors | `ruff check Backend` | **PASSED** (0 lint errors) |
| **Benchmark Smoke Test** | `12` | cases | `python Backend/scripts/benchmark_smoke_test.py` | **PASSED** (100% offline & deterministic) |
| **GitHub Actions Pipeline**| `.github/workflows/ci.yml` | workflow | GitHub Actions | **VERIFIED** (Credential-free CI) |
| **Deployment Model** | Systemd + Nginx | architecture | Production-style single-node | **VERIFIED** (`GET /health` 200 OK) |
