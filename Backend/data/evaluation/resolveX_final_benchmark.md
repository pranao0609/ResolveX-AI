# ResolveX 2.0 — Final Benchmark & Evaluation Report (Phase 24)

**Generated At:** 2026-09-26T07:51:36.412761+00:00  
**Git Commit SHA:** `111c1e41e67a880821223950211f993da2cc0d2b`  
**Environment:** `development`  
**LLM Model:** `openai/gpt-oss-120b`  
**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  

---

## 1. Executive Summary

This report presents the final multi-level evaluation of **ResolveX 2.0**, measuring Retrieval, Cross-Encoder Reranking, Diagnosis, Resolution, Verification, Policy Routing, Safety Guard Enforcement, End-to-End Graph Workflow Execution, and Component Latencies.

---

## 2. Evaluation Dataset Summary

- **Total Test Cases:** 12
- **Ground-Truth Curated Cases:** 12
- **Synthetic Cases:** 0
- **Dataset Source:** `D:\ResolveX-AI\Backend\data\evaluation\final_benchmark_dataset.json`

---

## 3. Retrieval Performance

| Retriever Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Hit Rate @ 5 |
|---|---|---|---|---|---|---|
| **BM25** | 0.55 | 0.75 | 0.85 | 0.95 | 0.8667 | 0.9 |
| **Dense (FAISS)** | 0.55 | 0.75 | 0.85 | 1.0 | 0.8143 | 0.9 |
| **Hybrid (BM25+Dense)** | 0.55 | 0.75 | 0.85 | 1.0 | 0.8143 | 0.9 |
| **Hybrid + Reranker** | 0.55 | 0.95 | 1.0 | 1.0 | 0.8333 | 1.0 |

---

## 4. Cross-Encoder Reranking Comparison

| Metric | Hybrid Retrieval | Hybrid + Reranker | Delta |
|---|---|---|---|
| **Recall@5** | 0.85 | 1.0 | +0.15 |
| **Recall@10** | 1.0 | 1.0 | +0.0 |
| **MRR** | 0.8143 | 0.8333 | +0.019 |

---

## 5. Verification Metrics

- **Verification Pass Rate:** `0.00%`
- **Verification Failure Rate:** `100.00%`
- **Evidence-Supported Rate:** `0.00%`
- **Policy-Compliant Rate:** `100.00%`
- **Hallucination Detection Rate:** `100.00%`
- **Incorrect-Resolution Detection Rate:** `100.00%`

---

## 6. Policy & Safety Results

| Metric | Measured Value |
|---|---|
| **Auto-Resolution Rate** | `0.00%` |
| **Clarification Rate** | `100.00%` |
| **Human Review Rate** | `0.00%` |
| **Escalation Rate** | `0.00%` |
| **Fallback Rate** | `100.00%` |
| **Safety Routing Accuracy** | `91.67%` |
| **Unsafe Auto-Resolutions** | `0` (Target: 0) |
| **Fail-Closed Rate** | `100.00%` |

---

## 7. Latency Benchmarks (ms)

| Workflow Component | P50 (Median) | P95 | P99 | Mean |
|---|---|---|---|---|
| **Total Workflow** | `8095.57 ms` | `19080.58 ms` | `19930.01 ms` | `9870.64 ms` |

---

## 8. Error & Fallback Analysis

- **Retrieval Failures:** `0`
- **LLM Failures:** `0`
- **Timeout Failures:** `0`
- **Verification Failures:** `12`
- **Policy Failures:** `0`
- **Fallback Count:** `12`

---

## 9. Reproducibility Instructions

To reproduce this benchmark from any terminal:

```bash
$env:PYTHONPATH='Backend'
python Backend/scripts/run_final_benchmark.py
```
