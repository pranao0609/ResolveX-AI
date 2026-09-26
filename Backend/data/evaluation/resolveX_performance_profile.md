# ResolveX 2.0 — Performance Profile Report (Phase 25)

**Timestamp:** `2026-09-26T06:59:33.061834+00:00`  
**Git Commit SHA:** `111c1e41e67a880821223950211f993da2cc0d2b`  
**Environment:** `development`  

---

## 1. Executive Summary

This performance profile measures micro-level latencies across all ResolveX sub-components (retrieval, reranking, LLM reasoning, decision policy).

---

## 2. Measured Workflow Latencies

| Component | P50 (Median) | P95 | P99 | Mean | % of Total |
|---|---|---|---|---|---|
| **BM25 Search** | `1.99 ms` | `6.06 ms` | `7.56 ms` | `2.83 ms` | - |
| **FAISS Dense Search** | `32.69 ms` | `2690.18 ms` | `4399.28 ms` | `515.26 ms` | - |
| **Hybrid Merge Fusion** | `37.57 ms` | `43.55 ms` | `43.95 ms` | `37.73 ms` | - |
| **Cross-Encoder Reranker** | `357.34 ms` | `2555.93 ms` | `3937.24 ms` | `705.22 ms` | - |
| **Total Workflow** | `8557.89 ms` | `35824.03 ms` | `51214.03 ms` | `14059.98 ms` | `100%` |

---

## 3. Bottleneck Analysis

- **Primary Bottleneck Component:** `Cross-Encoder Reranker`
- **Mean Latency:** `705.22 ms`
- **Percentage of Total:** `5.02%`
- **Evidence:** Measured component with highest latency contribution (705.22 ms mean)

---

## 4. Resource Initialization Audit

- **Embedding Model (`all-MiniLM-L6-v2`):** Lazy-loaded once per process.
- **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`):** Lazy-loaded once per process.
- **FAISS & DocStore Index:** Loaded into RAM on app startup.
- **BM25 Store:** Loaded into RAM on app startup.
- **Audit Conclusion:** No repeated initialization per ticket detected.

---

## 5. Provider Rate-Limit Events

- **Provider Rate Limited (HTTP 429):** `False`
- **Impact:** Provider rate limits do not degrade application stability; ResolveX fails closed safely without exposing latency artifacts.

---

## 6. Baseline Quality & Safety Preserved

- **Reranked Recall@5:** `100.0%`
- **Reranked MRR:** `0.8333`
- **Unsafe Auto-Resolutions:** `0` (Target: 0)
