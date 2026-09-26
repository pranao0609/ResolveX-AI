# ResolveX — Resume Bullets, LinkedIn Material & Portfolio Guide

**Project:** ResolveX — Agentic AI IT Support Ticket Resolution Platform  
**Date:** September 2026  
**Status:** ENGINEERING COMPLETE  

This document provides defensible, metric-backed resume bullet points, LinkedIn posts, ATS-friendly variations, and interviewing guardrails based strictly on verified repository artifacts.

---

## 1. Top 10 Verified Metrics for Resume Usage

1. **Reranked Recall@5 = 1.0000** (Cross-Encoder Reranking improved Recall@5 from 0.8500 to 1.0000 across benchmark evaluations).
2. **0 Unsafe Auto-Resolutions** (Achieved 0 safety rule violations across all evaluated test cases).
3. **100% Fail-Closed Safety Rate** (System automatically degrades to human review whenever LLM output fails verification or rate limits occur).
4. **753 Unit & Integration Tests Passing** (0 test failures across complete regression suite).
5. **23 Dedicated Security Tests Passing** (Covering secret redaction, prompt injection boundaries, CORS, headers, and exception sanitization).
6. **34.92% Prompt Token Payload Reduction** (V1 $\rightarrow$ V2 prompt refactoring reduced token overhead across Diagnosis, Resolution, and Verification agents).
7. **8076.85 ms P50 Workflow Latency** (Full end-to-end multi-agent execution profile).
8. **12-Stage Stateful LangGraph Workflow** (Structured node state machine orchestrating analysis, retrieval, diagnosis, resolution, verification, and safety routing).
9. **154 Indexed Knowledge Base Vectors** (FAISS Dense Vector Store + BM25 Sparse Index over enterprise IT SOPs).
10. **8 Security Vulnerabilities Remediated** (Validated via automated security regression suite).

---

## 2. Recommended Resume Bullet Options

### Option A: 3 Strongest Resume Bullets (General Software / AI Role)

* **Architected an Enterprise Agentic RAG Platform** using LangGraph, FastAPI, and React, executing a 12-stage stateful workflow that diagnoses, resolves, and routes IT tickets with 0 unsafe auto-resolutions across benchmark evaluation suites.
* **Engineered a Dual Hybrid Retrieval & Reranking Pipeline** combining Okapi BM25, FAISS dense vector search (`all-MiniLM-L6-v2`), and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`), boosting Recall@5 from 0.8500 to 1.0000 and MRR to 0.8333.
* **Implemented Evidence-Gated Safety Verification & Security Controls**, enforcing confidence gates ($\ge 0.85$), log secret redaction, prompt injection boundaries, 23 security tests, and a 100% fail-closed safety routing architecture.

---

### Option B: 4 Strongest Resume Bullets

* **Architected an Enterprise Agentic RAG Platform** using LangGraph, FastAPI, and React, executing a 12-stage stateful workflow that diagnoses, resolves, and routes IT tickets with 0 unsafe auto-resolutions across benchmark evaluation suites.
* **Engineered a Dual Hybrid Retrieval & Reranking Pipeline** combining Okapi BM25, FAISS dense vector search (`all-MiniLM-L6-v2`), and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`), boosting Recall@5 from 0.8500 to 1.0000 and MRR to 0.8333.
* **Optimized Multi-Agent Prompt Token Payloads**, refactoring system/user prompt architecture to achieve a 34.92% reduction in token overhead across Diagnosis, Resolution, and Verification agents while preserving 100% safety routing accuracy.
* **Designed Evidence-Gated Safety Verification & Production Security Controls**, authoring a 753-test suite, 23 security tests, Black/Ruff quality gates, credential-free GitHub Actions CI, and Systemd/Nginx single-node deployment.

---

### Option C: 5 Strongest Resume Bullets

* **Architected an Enterprise Agentic RAG Platform** using LangGraph, FastAPI, and React, executing a 12-stage stateful workflow that diagnoses, resolves, and routes IT tickets with 0 unsafe auto-resolutions across benchmark evaluation suites.
* **Engineered a Dual Hybrid Retrieval & Reranking Pipeline** combining Okapi BM25, FAISS dense vector search (`all-MiniLM-L6-v2`), and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`), boosting Recall@5 from 0.8500 to 1.0000 and MRR to 0.8333.
* **Built an Independent Evidence Verification Gate**, evaluating LLM-generated resolution steps against retrieved KB facts with a strict $\ge 0.85$ confidence threshold and 100% fail-closed routing under rate-limit or context failures.
* **Optimized Multi-Agent Prompt Token Payloads**, refactoring system/user prompt architecture to achieve a 34.92% reduction in token overhead across Diagnosis, Resolution, and Verification agents while preserving 100% safety routing accuracy.
* **Established Automated CI/CD & Testing Infrastructure**, authoring credential-free GitHub Actions workflows, 23 security tests, a 753-test regression suite, Black/Ruff quality gates, and a Systemd/Nginx deployment setup.

---

## 3. Specialized Resume Variations

### Best 3 Bullets for AI / ML Engineer Resume

* **Engineered a Stateful Multi-Agent Orchestration Engine** in LangGraph and Python, managing 12 discrete reasoning stages, context passing, and fallback routing for automated IT support ticket resolution.
* **Developed a Hybrid RAG Retrieval & Cross-Encoder Reranking Architecture** (BM25 + FAISS + `ms-marco-MiniLM-L-6-v2`), improving Recall@5 from 0.8500 to 1.0000 (+17.65%) and MRR to 0.8333 on benchmark IT evaluation datasets.
* **Designed an Evidence-Gated LLM Verification & Prompt Optimization Framework**, achieving a 34.92% reduction in prompt token payload while guaranteeing 0 unsafe auto-resolutions via deterministic verification gates.

---

### Best 3 Bullets for Software / Full-Stack AI Engineer Resume

* **Built an AI-Powered IT Ticket Resolution Platform** featuring a FastAPI backend, React 19 SPA frontend, PostgreSQL database, and a 12-stage LangGraph stateful multi-agent workflow.
* **Implemented Enterprise Security Hardening & Fail-Closed Safety Controls**, developing 23 security tests, automated log secret redaction, prompt injection isolation, CORS protection, and HTTP security headers.
* **Established Full MLOps & CI/CD Pipelines**, integrating GitHub Actions, Black/Ruff quality checks, 753 regression tests, MLflow experiment tracking, and Systemd/Nginx deployment configurations.

---

## 4. LinkedIn Post Material

### Detailed LinkedIn Post

🚀 Excited to announce **ResolveX** — an Enterprise Agentic AI IT Support Ticket Resolution Platform built to automate Level-1 service desk requests cleanly and safely!

In real-world enterprise operations, raw LLM text generation isn't enough — safety, factual verification, and deterministic fallback routing are non-negotiable.

Here is a breakdown of what I built:
🔹 **Multi-Agent Orchestration**: Designed a 12-stage state machine in **LangGraph** & **FastAPI** that handles ticket analysis, retrieval, diagnosis, resolution, verification, and safety routing.
🔹 **Hybrid RAG & Cross-Encoder Reranking**: Combined BM25 sparse search, FAISS dense vector search (`all-MiniLM-L6-v2`), and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`), boosting Recall@5 from 0.85 to 1.0000.
🔹 **Deterministic Safety & HITL**: Created an independent verification gate requiring $\ge 0.85$ evidence confidence — achieving **0 unsafe auto-resolutions** and 100% fail-closed routing across benchmark evaluations.
🔹 **Prompt Efficiency**: Refactored multi-agent prompt payloads to cut token overhead by **34.92%** without degrading safety controls.
🔹 **Production Security & CI/CD**: Authored 23 dedicated security tests, log secret redaction, automated GitHub Actions CI, and Systemd/Nginx single-node deployment.

Check out the full open-source repo & architecture docs: https://github.com/pranao0609/ResolveX-AI

#AI #MachineLearning #AgenticAI #LangGraph #FastAPI #RAG #Python #SoftwareEngineering #MLOps

---

### Short LinkedIn Post

🚀 Just wrapped up **ResolveX** — an Agentic AI IT Support Platform built with LangGraph, FastAPI, and React!

Highlights:
• 12-stage stateful agent workflow for automated ticket diagnosis & resolution
• Hybrid BM25 + FAISS Dense Vector search with Cross-Encoder reranking (Recall@5 = 1.0000)
• Evidence-gated verification achieving 0 unsafe auto-resolutions & 100% fail-closed safety routing
• 34.92% prompt token payload reduction
• 753 tests passing + 23 dedicated security tests + GitHub Actions CI

Repo & docs: https://github.com/pranao0609/ResolveX-AI

#AgenticAI #RAG #LangGraph #FastAPI #Python #MLOps

---

## 5. Claims NOT to Make on Resume / Interview Guardrails

| DO NOT CLAIM | WHY IT IS INACCURATE | WHAT TO SAY INSTEAD |
| :--- | :--- | :--- |
| "Deploys at multi-region cloud scale on AWS Kubernetes" | Deployment setup targets a production-style single-node architecture (FastAPI/Systemd/Nginx). | "Designed production-style single-node deployment workflows using Systemd, Nginx, and GitHub Actions." |
| "Prompt optimization reduced latency by 35%" | Prompt V2 reduced **token payload** by 34.92%, but end-to-end latency was governed by cloud LLM API network response variance. | "Reduced multi-agent prompt token payloads by 34.92% while preserving 100% safety routing accuracy." |
| "Continuous online reinforcement learning" | Policy framework uses contextual bandit principles with offline evaluation/replay updates, not live gradient updates. | "Implemented a contextual bandit policy layer with offline replay evaluation and deterministic safety overrides." |
| "100% LLM accuracy" | Evaluation is based on a 12-case benchmark and 10-query KB retrieval dataset, not unlimited live traffic. | "Achieved 1.0000 Recall@5 on benchmark retrieval and 0 unsafe auto-resolutions under fail-closed safety rules." |
| "Self-hosted LLM cluster" | Primary inference relies on external Groq Cloud API (`gpt-oss-120b`). | "Integrated Groq Cloud LLM API with custom resilience wrappers, retry handling, and fail-closed fallback routing." |
