"""
evaluate_10_kb_queries.py — 10-Query How-To Evaluation & Regression Runner
"""

import sys
import os
import json
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.database import init_db, SessionLocal
from app.models.kb_model import KnowledgeBaseEntry
from ai.rag.doc_store import get_doc_store
from ai.rag.bm25_store import get_bm25_store
from ai.rag.retriever import retrieve_context, validate_store_alignment
from ai.rag.hybrid_retriever import HybridRetriever
from ai.rag.reranker import get_reranker
from ai.graph.executor import execute_resolvex_graph


class EvalTicket:
    def __init__(self, ticket_id: int, title: str, description: str):
        self.id = ticket_id
        self.title = title
        self.description = description
        self.attachment_paths = []


QUERIES_HOWTO = [
    {
        "id": "Q1",
        "title": "How do I reset my password using the Forgot Password option?",
        "description": "I am not experiencing a login error and my account is not locked. I simply need the standard documented procedure for resetting my password. I can access my registered email address. Please provide the password reset steps from the knowledge base.",
        "expected_kb_title": "Standard Password Reset Procedure",
        "expected_intent": "Standard password reset procedure.",
    },
    {
        "id": "Q2",
        "title": "How do I connect to the company VPN?",
        "description": "I need the standard procedure for connecting to the company VPN. I have the approved VPN client and authorized credentials. Please provide the documented connection steps.",
        "expected_kb_title": "Standard VPN Connection Procedure",
        "expected_intent": "VPN connection procedure.",
    },
    {
        "id": "Q3",
        "title": "How do I clear my browser cache?",
        "description": "I only need the standard documented steps for clearing browser cached data. There is no current application failure.",
        "expected_kb_title": "Clear Browser Cache Procedure",
        "expected_intent": "Clear browser cache procedure.",
    },
    {
        "id": "Q4",
        "title": "How do I enable MFA?",
        "description": "I need the standard procedure for enrolling my account in multi-factor authentication. There is no authentication failure.",
        "expected_kb_title": "Standard MFA Enrollment Procedure",
        "expected_intent": "MFA enrollment procedure.",
    },
    {
        "id": "Q5",
        "title": "How do I unlock my account?",
        "description": "My account is confirmed to be locked and I need the standard approved account unlock procedure.",
        "expected_kb_title": "Standard Account Unlock Procedure",
        "expected_intent": "Account unlock procedure.",
    },
    {
        "id": "Q6",
        "title": "How do I configure my company email?",
        "description": "I need the standard documented procedure for configuring my approved company email account in the supported email application.",
        "expected_kb_title": "Standard Email Account Configuration Procedure",
        "expected_intent": "Email configuration procedure.",
    },
    {
        "id": "Q7",
        "title": "What is the standard way to restart the application?",
        "description": "I only need the documented procedure for safely closing and restarting the application. There is no current crash or error.",
        "expected_kb_title": "Standard Application Restart Procedure",
        "expected_intent": "Application restart procedure.",
    },
    {
        "id": "Q8",
        "title": "My password reset email did not arrive",
        "description": "I requested a password reset but the email has not arrived. I want to follow the standard troubleshooting procedure for the missing reset email.",
        "expected_kb_title": "Password Reset Email Not Received",
        "expected_intent": "Password reset email troubleshooting.",
    },
    {
        "id": "Q9",
        "title": "How should an application connect to PostgreSQL?",
        "description": "I need the standard documented procedure for configuring and verifying an application's PostgreSQL connection. This is a configuration question, not an active database outage.",
        "expected_kb_title": "Standard PostgreSQL Application Connection Procedure",
        "expected_intent": "PostgreSQL connection procedure.",
    },
    {
        "id": "Q10",
        "title": "How do I perform a basic network connectivity check?",
        "description": "I need the standard procedure for checking network connectivity to a relevant host. There is no confirmed outage; I only need the documented diagnostic procedure.",
        "expected_kb_title": "Standard Network Connectivity Check",
        "expected_intent": "Basic network connectivity procedure.",
    },
]

REGRESSION_QUERIES = [
    {
        "id": "R1",
        "title": "Application Crash",
        "description": "The application crashes immediately upon opening with error 0x8004005. I cannot open any workspace.",
        "expected_intent": "Incident troubleshooting / application crash",
    },
    {
        "id": "R2",
        "title": "PostgreSQL Connection Failure",
        "description": "Application cannot connect to PostgreSQL database server after network maintenance. Error: Connection refused on port 5432.",
        "expected_intent": "Incident troubleshooting / DB outage",
    },
    {
        "id": "R3",
        "title": "VPN Connection Failure",
        "description": "VPN client fails to connect with error code 800: Remote connection was not made because attempted VPN tunnels failed.",
        "expected_intent": "Incident troubleshooting / VPN failure",
    },
    {
        "id": "R4",
        "title": "Ambiguous Ticket",
        "description": "Something is slow and not working properly today.",
        "expected_intent": "Ambiguous request requiring clarification",
    },
]


def run_evaluation():
    init_db()
    doc_store = get_doc_store()
    bm25_store = get_bm25_store()
    hybrid_retriever = HybridRetriever()
    reranker = get_reranker()

    print("=" * 100)
    print("RESOLVEX KNOWLEDGE BASE EVALUATION: 10 HOW-TO PROCEDURES")
    print("=" * 100)

    results_howto = []

    for index, q in enumerate(QUERIES_HOWTO, start=1001):
        query_text = f"{q['title']}\n\n{q['description']}"
        expected_title = q["expected_kb_title"]

        # 1. Direct Retrieval Checks
        bm25_res = bm25_store.search(query=query_text, top_k=20)
        dense_res = retrieve_context(query=query_text, top_k=20, score_threshold=-1.0)
        hybrid_candidates = hybrid_retriever.retrieve(
            query=query_text, top_k=20, candidate_k=20
        )
        reranked_candidates = reranker.rerank(
            query=query_text, candidates=hybrid_candidates, top_k=5
        )

        # Check BM25 rank
        bm25_titles = [
            doc_store.get_document(r["index_id"]).get("title", "") for r in bm25_res
        ]
        bm25_in_top5 = expected_title in bm25_titles[:5]
        bm25_rank = (
            (bm25_titles.index(expected_title) + 1)
            if expected_title in bm25_titles
            else None
        )

        # Check Dense rank
        dense_titles = [r.get("title", "") for r in dense_res]
        dense_in_top5 = expected_title in dense_titles[:5]
        dense_rank = (
            (dense_titles.index(expected_title) + 1)
            if expected_title in dense_titles
            else None
        )

        # Check Hybrid rank
        hybrid_titles = [
            doc_store.get_document(c.index_id).get("title", "")
            for c in hybrid_candidates
        ]
        hybrid_in_top5 = expected_title in hybrid_titles[:5]
        hybrid_rank = (
            (hybrid_titles.index(expected_title) + 1)
            if expected_title in hybrid_titles
            else None
        )

        # Check Reranker rank & top result
        reranked_docs = [
            doc_store.get_document(c.index_id) for c in reranked_candidates
        ]
        reranked_titles = [d.get("title", "") if d else "" for d in reranked_docs]
        reranker_in_top5 = expected_title in reranked_titles
        reranker_rank = (
            (reranked_titles.index(expected_title) + 1)
            if expected_title in reranked_titles
            else None
        )
        top_reranked_title = reranked_titles[0] if reranked_titles else "N/A"
        retrieval_top_score = (
            reranked_candidates[0].score if reranked_candidates else 0.0
        )
        score_gap = (
            (reranked_candidates[0].score - reranked_candidates[1].score)
            if len(reranked_candidates) > 1
            else 0.0
        )

        # 2. Execute End-to-End LangGraph
        ticket = EvalTicket(
            ticket_id=index, title=q["title"], description=q["description"]
        )
        graph_output = execute_resolvex_graph(ticket, include_evaluation_details=True)

        eval_details = graph_output.get("evaluation", {})
        decision = graph_output.get("decision", "N/A")
        auto_resolved = graph_output.get("auto_resolved", False)
        requires_human = graph_output.get("requires_human", False)

        res_item = {
            "query_id": q["id"],
            "title": q["title"],
            "expected_kb": expected_title,
            "top_retrieved": top_reranked_title,
            "expected_rank_hybrid": hybrid_rank,
            "in_hybrid_top5": hybrid_in_top5,
            "expected_rank_reranker": reranker_rank,
            "in_reranker_top5": reranker_in_top5,
            "retrieval_top_score": float(retrieval_top_score),
            "retrieval_score_gap": float(score_gap),
            "retrieval_decision": eval_details.get("retrieval_metadata", {}).get(
                "decision", "N/A"
            ),
            "diagnosis_confidence": graph_output.get("diagnosis_confidence", 0.0),
            "resolution_confidence": graph_output.get("resolution_confidence", 0.0),
            "verification_passed": graph_output.get("verification_passed", False),
            "verification_confidence": graph_output.get("verification_confidence", 0.0),
            "overall_confidence": graph_output.get("confidence", 0.0),
            "final_decision": decision,
            "auto_resolved": auto_resolved,
            "requires_human": requires_human,
            "fallback_used": graph_output.get("fallback_used", False),
            "errors": len(graph_output.get("errors", [])),
            "warnings": len(graph_output.get("warnings", [])),
        }
        results_howto.append(res_item)

        print(f"\n[{q['id']}] {q['title']}")
        print(f"   Expected KB:     {expected_title}")
        print(f"   Top Reranked:    {top_reranked_title} (Rank: {reranker_rank})")
        print(
            f"   In Top 5?        Reranker={reranker_in_top5}, Hybrid={hybrid_in_top5}"
        )
        print(f"   Verification:    Passed={graph_output.get('verification_passed')}")
        print(f"   Final Decision:  {decision} (AutoResolved={auto_resolved})")

    # 3. Compute How-To Metrics
    total = len(results_howto)
    reranker_top5_hits = sum(1 for r in results_howto if r["in_reranker_top5"])
    hybrid_top5_hits = sum(1 for r in results_howto if r["in_hybrid_top5"])
    verif_pass_count = sum(1 for r in results_howto if r["verification_passed"])
    auto_resolve_count = sum(
        1 for r in results_howto if r["final_decision"] == "auto_resolve"
    )
    clarification_count = sum(
        1 for r in results_howto if r["final_decision"] == "ask_clarification"
    )
    human_review_count = sum(
        1 for r in results_howto if r["final_decision"] == "human_review"
    )
    escalation_count = sum(
        1 for r in results_howto if r["final_decision"] == "escalate"
    )
    fallback_count = sum(1 for r in results_howto if r["fallback_used"])

    metrics_howto = {
        "queries_total": total,
        "hybrid_top5_recall": hybrid_top5_hits / total if total else 0.0,
        "reranker_top5_recall": reranker_top5_hits / total if total else 0.0,
        "verification_pass_rate": verif_pass_count / total if total else 0.0,
        "auto_resolution_rate": auto_resolve_count / total if total else 0.0,
        "clarification_rate": clarification_count / total if total else 0.0,
        "human_review_rate": human_review_count / total if total else 0.0,
        "escalation_rate": escalation_count / total if total else 0.0,
        "fallback_rate": fallback_count / total if total else 0.0,
    }

    print("\n" + "=" * 100)
    print("REGRESSION EVALUATION: INCIDENT & AMBIGUOUS TICKETS")
    print("=" * 100)

    results_regression = []
    for index, q in enumerate(REGRESSION_QUERIES, start=2001):
        ticket = EvalTicket(
            ticket_id=index, title=q["title"], description=q["description"]
        )
        graph_output = execute_resolvex_graph(ticket, include_evaluation_details=True)
        decision = graph_output.get("decision", "N/A")
        auto_resolved = graph_output.get("auto_resolved", False)

        reg_item = {
            "id": q["id"],
            "title": q["title"],
            "expected_intent": q["expected_intent"],
            "final_decision": decision,
            "auto_resolved": auto_resolved,
            "verification_passed": graph_output.get("verification_passed", False),
            "confidence": graph_output.get("confidence", 0.0),
        }
        results_regression.append(reg_item)

        print(f"\n[{q['id']}] {q['title']}")
        print(f"   Intent:          {q['expected_intent']}")
        print(f"   Verification:    Passed={graph_output.get('verification_passed')}")
        print(f"   Final Decision:  {decision} (AutoResolved={auto_resolved})")

    # Output JSON summary
    summary_report = {
        "how_to_results": results_howto,
        "how_to_metrics": metrics_howto,
        "regression_results": results_regression,
    }

    out_file = BACKEND_ROOT / "data" / "evaluation" / "eval_10_kb_queries_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("10-QUERY EVALUATION SUMMARY METRICS")
    print("=" * 100)
    print(f"Total How-To Queries:     {metrics_howto['queries_total']}")
    print(f"Retrieval Top-5 Recall:   {metrics_howto['hybrid_top5_recall']:.2%}")
    print(f"Reranker Top-5 Recall:    {metrics_howto['reranker_top5_recall']:.2%}")
    print(f"Verification Pass Rate:   {metrics_howto['verification_pass_rate']:.2%}")
    print(f"Auto-Resolution Rate:     {metrics_howto['auto_resolution_rate']:.2%}")
    print(f"Clarification Rate:       {metrics_howto['clarification_rate']:.2%}")
    print(f"Human-Review Rate:        {metrics_howto['human_review_rate']:.2%}")
    print(f"Escalation Rate:          {metrics_howto['escalation_rate']:.2%}")
    print(f"Fallback Rate:            {metrics_howto['fallback_rate']:.2%}")
    print("=" * 100)


if __name__ == "__main__":
    run_evaluation()
