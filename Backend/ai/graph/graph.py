from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai.agents.classification_agent import run_classification_agent
from ai.agents.retrieval_tools import (
    rerank_documents,
    search_knowledge_base,
    search_previous_tickets,
)
from ai.config.ai_config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_STRATEGY,
    RETRIEVAL_TOP_K,
)
from ai.graph.nodes import (
    ask_clarification,
    auto_resolve,
    decision_agent,
    diagnosis_agent,
    escalate,
    human_review,
    initialize_graph_state,
    resolution_agent,
    retrieval_agent,
    retrieval_decision_agent,
    ticket_analyzer,
)
from ai.graph.routing import route_decision, route_retrieval_decision
from ai.graph.state import ResolveXState
from ai.llm.diagnosis_generator import generate_diagnosis
from ai.llm.solution_generator import generate_solution
from ai.llm.verification_generator import verify_resolution
from ai.graph.nodes.verification import verification_agent

# =====================================================================
# Graph Construction
# =====================================================================


def build_resolvex_graph():
    """
    Build the ResolveX LangGraph orchestration graph.

    Nodes:
        - initialize_state
        - ticket_analyzer
        - retrieval_agent
        - retrieval_decision_agent
        - diagnosis_agent
        - resolution_agent
        - verification_agent
        - decision_agent
        - auto_resolve
        - ask_clarification
        - human_review
        - escalate
    """

    builder = StateGraph(ResolveXState)

    # ---------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------

    builder.add_node(
        "initialize_state",
        initialize_graph_state,
    )

    builder.add_node(
        "ticket_analyzer",
        ticket_analyzer,
    )

    builder.add_node(
        "retrieval_agent",
        retrieval_agent,
    )

    builder.add_node(
        "retrieval_decision_agent",
        retrieval_decision_agent,
    )

    builder.add_node(
        "diagnosis_agent",
        diagnosis_agent,
    )

    builder.add_node(
        "resolution_agent",
        resolution_agent,
    )

    builder.add_node(
        "verification_agent",
        verification_agent,
    )

    builder.add_node(
        "decision_agent",
        decision_agent,
    )

    builder.add_node(
        "auto_resolve",
        auto_resolve,
    )

    builder.add_node(
        "ask_clarification",
        ask_clarification,
    )

    builder.add_node(
        "human_review",
        human_review,
    )

    builder.add_node(
        "escalate",
        escalate,
    )

    # ---------------------------------------------------------------
    # Main Workflow
    # ---------------------------------------------------------------

    builder.add_edge(
        START,
        "initialize_state",
    )

    builder.add_edge(
        "initialize_state",
        "ticket_analyzer",
    )

    builder.add_edge(
        "ticket_analyzer",
        "retrieval_agent",
    )

    builder.add_edge(
        "retrieval_agent",
        "retrieval_decision_agent",
    )

    builder.add_conditional_edges(
        "retrieval_decision_agent",
        route_retrieval_decision,
        {
            "continue": "diagnosis_agent",
            "retry": "retrieval_agent",
        },
    )

    builder.add_edge(
        "diagnosis_agent",
        "resolution_agent",
    )

    builder.add_edge(
        "resolution_agent",
        "verification_agent",
    )

    builder.add_edge(
        "verification_agent",
        "decision_agent",
    )

    # ---------------------------------------------------------------
    # Decision Routing
    # ---------------------------------------------------------------

    builder.add_conditional_edges(
        "decision_agent",
        route_decision,
        {
            "auto_resolve": "auto_resolve",
            "ask_clarification": "ask_clarification",
            "human_review": "human_review",
            "escalate": "escalate",
        },
    )

    # ---------------------------------------------------------------
    # Terminal Nodes
    # ---------------------------------------------------------------

    builder.add_edge(
        "auto_resolve",
        END,
    )

    builder.add_edge(
        "ask_clarification",
        END,
    )

    builder.add_edge(
        "human_review",
        END,
    )

    builder.add_edge(
        "escalate",
        END,
    )

    return builder.compile()


resolvex_graph = build_resolvex_graph()

__all__ = [
    "ResolveXState",
    "build_resolvex_graph",
    "resolvex_graph",
    "initialize_graph_state",
    "ticket_analyzer",
    "retrieval_agent",
    "retrieval_decision_agent",
    "diagnosis_agent",
    "resolution_agent",
    "verification_agent",
    "decision_agent",
    "auto_resolve",
    "ask_clarification",
    "human_review",
    "escalate",
    "route_retrieval_decision",
    "route_decision",
    "generate_diagnosis",
    "generate_solution",
    "verify_resolution",
    "run_classification_agent",
    "search_knowledge_base",
    "search_previous_tickets",
    "rerank_documents",
    "RETRIEVAL_STRATEGY",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_CANDIDATE_K",
    "BM25_WEIGHT",
    "DENSE_WEIGHT",
]
