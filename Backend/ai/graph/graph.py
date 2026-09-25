from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from ai.graph.state import ResolveXState
from ai.preprocessing.text_cleaner import clean_text
from ai.preprocessing.file_parser import parse_attachments
from ai.agents.classification_agent import run_classification_agent
from ai.config.ai_config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_STRATEGY,
    RETRIEVAL_TOP_K,
)
from ai.llm.diagnosis_generator import generate_diagnosis
from ai.llm.solution_generator import generate_solution
from ai.llm.verification_generator import verify_resolution

from app.core.logger import logger
from ai.agents.retrieval_metadata import build_retrieval_metadata
from ai.agents.query_analyzer import analyze_query
from ai.rag.doc_store import get_doc_store
from ai.rag.hybrid_retriever import HybridRetriever
from ai.rag.retriever import retrieve_context
from ai.agents.retrieval_tools import (
    search_knowledge_base,
    search_previous_tickets,
    rerank_documents,
)
# =====================================================================
# Agentic Retrieval Configuration
# =====================================================================

MAX_RETRIEVAL_ATTEMPTS = 2

MIN_RETRIEVAL_DOCUMENTS = 2
MIN_RETRIEVAL_TOP_SCORE = 0.15

RETRIEVAL_RETRY_STRATEGIES = (
    "broaden",
    "narrow",
    "rewrite",
)

hybrid_retriever = HybridRetriever(
    bm25_weight=BM25_WEIGHT,
    dense_weight=DENSE_WEIGHT,
)

doc_store = get_doc_store()

# =====================================================================
# Runtime State / Metadata Helpers
# =====================================================================


def _ensure_runtime_metadata(
    state: ResolveXState,
) -> dict:
    """
    Initialize graph execution metadata.

    Creates stable identifiers for a single graph execution
    without introducing an external persistence dependency.
    """

    request_id = state.get("request_id")

    if not request_id:
        request_id = str(uuid.uuid4())

    graph_run_id = state.get("graph_run_id")

    if not graph_run_id:
        graph_run_id = str(uuid.uuid4())

    metadata = dict(
        state.get("metadata", {})
    )

    if "started_at" not in metadata:
        metadata["started_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

    metadata["ticket_id"] = state.get(
        "ticket_id"
    )

    metadata["last_stage"] = (
        "initialize_state"
    )

    return {
        "request_id": request_id,
        "graph_run_id": graph_run_id,
        "metadata": metadata,
    }


def initialize_graph_state(
    state: ResolveXState,
) -> dict:
    """
    Initialize runtime state before the first agent executes.
    """

    runtime = _ensure_runtime_metadata(
        state
    )

    existing_errors = list(
        state.get("errors", [])
    )

    existing_warnings = list(
        state.get("warnings", [])
    )

    return {
        "request_id": runtime["request_id"],
        "graph_run_id": runtime["graph_run_id"],
        "metadata": runtime["metadata"],
        "errors": existing_errors,
        "warnings": existing_warnings,
        "fallback_used": bool(
            state.get("fallback_used", False)
        ),
    }


def _stage_metadata(
    state: ResolveXState,
    stage: str,
    completed: bool = False,
) -> dict:
    """
    Create updated metadata for a graph stage.

    Metadata is intentionally lightweight and serializable so that
    it can later be connected to LangGraph checkpointing,
    Redis, PostgreSQL, or another persistence backend.
    """

    metadata = dict(
        state.get("metadata", {})
    )

    metadata["last_stage"] = stage

    metadata["updated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    if completed:
        metadata["completed_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

    return metadata


def _append_error(
    state: ResolveXState,
    stage: str,
    exc: Exception,
) -> list[str]:
    """
    Append an error while preserving all previous errors.
    """

    errors = list(
        state.get("errors", [])
    )

    errors.append(
        f"{stage}: "
        f"{type(exc).__name__}: {exc}"
    )

    return errors


# =====================================================================
# Graph Construction
# =====================================================================


def build_resolvex_graph():
    """
    Build the ResolveX LangGraph orchestration graph.

    Phase 14.7 adds:
        - Runtime request ID
        - Graph run ID
        - State metadata
        - Error accumulation
        - Persistence-ready metadata
        - Safe state propagation
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


# =====================================================================
# Ticket Analyzer Agent
# =====================================================================


def ticket_analyzer(
    state: ResolveXState,
) -> dict:
    """
    Analyze the incoming support ticket.

    Responsibilities:
        1. Clean ticket text.
        2. Extract attachment text.
        3. Execute the Classification Agent.
        4. Write validated results into graph state.
    """

    ticket_text = (
        state.get("ticket_text", "")
        or ""
    )

    attachment_paths = state.get(
        "attachment_paths"
    )

    try:
        # -----------------------------------------------------------
        # Step 1: Text preprocessing
        # -----------------------------------------------------------

        cleaned_text = clean_text(
            ticket_text
        )

        # -----------------------------------------------------------
        # Step 2: Attachment extraction
        # -----------------------------------------------------------

        if attachment_paths:

            if isinstance(
                attachment_paths,
                str,
            ):
                paths = [
                    path.strip()
                    for path in attachment_paths.split(
                        ","
                    )
                    if path.strip()
                ]

            else:
                paths = [
                    str(path).strip()
                    for path in attachment_paths
                    if str(path).strip()
                ]

            if paths:
                extracted = parse_attachments(
                    paths
                )

                if extracted:
                    cleaned_text = (
                        f"{cleaned_text}\n\n"
                        f"[Attachments]\n"
                        f"{extracted}"
                    )

        # -----------------------------------------------------------
        # Step 3: Classification Agent
        # -----------------------------------------------------------

        classification = run_classification_agent(
            cleaned_text,
            ticket_id=state.get("ticket_id"),
        )

        category = classification["category"]
        classification_score = classification["confidence"]
        classification_confidence_level = (classification["confidence_level"])
        classification_requires_review = (classification["requires_review"])
        classification_requires_reclassification = (classification["requires_reclassification"])
        classification_fallback = classification["fallback_used"]
        classification_error = classification.get("error")
        reclassified = classification.get("reclassified",False,)
        reclassification_used = classification.get("reclassification_used",False,)
        reclassification_error = classification.get("reclassification_error",)

        logger.info(
    f"ticket_id={state.get('ticket_id')} "
    f"stage=ticket_analyzer "
    f"category={category} "
    f"score={classification_score:.3f} "
    f"confidence_level={classification_confidence_level} "
    f"requires_review={classification_requires_review} "
    f"requires_reclassification="
    f"{classification_requires_reclassification} "
    f"fallback={classification_fallback} "
    f"text_length={len(cleaned_text)} "
    f"request_id={state.get('request_id')} "
    f"graph_run_id={state.get('graph_run_id')}"
)

        # -----------------------------------------------------------
        # Step 4: Stage metadata
        # -----------------------------------------------------------

        metadata = _stage_metadata(
            state,
            "ticket_analyzer",
        )

        result = {
            "cleaned_ticket": cleaned_text,
            "category": category,
            "category_confidence": float(
                classification_score
            ),
            "classification_confidence_level": (
                classification_confidence_level
            ),
            "classification_requires_review": (
                classification_requires_review
             ),
            "classification_requires_reclassification": (
        classification_requires_reclassification
                ),
                "reclassified": reclassified,
    "reclassification_used": reclassification_used,
    "reclassification_error": reclassification_error,   
            "metadata": metadata,
        }

        # -----------------------------------------------------------
        # Step 5: Propagate classification fallback/error
        # -----------------------------------------------------------

        if classification_fallback:
            result["fallback_used"] = True

        if classification_error:
            result["errors"] = _append_error(
                state,
                "classification_agent",
                classification_error,
            )

        return result

    except Exception as exc:

        logger.exception(
            f"Ticket analyzer failed for "
            f"ticket_id={state.get('ticket_id')}: {exc}"
        )

        metadata = _stage_metadata(
            state,
            "ticket_analyzer",
        )

        errors = _append_error(
            state,
            "ticket_analyzer",
            exc,
        )

        return {
            "cleaned_ticket": (
                cleaned_text
                if "cleaned_text" in locals()
                else ""
            ),
            "category": "software",
            "category_confidence": 0.0,
            "errors": errors,
            "fallback_used": True,
            "metadata": metadata,
        }


# =====================================================================
# Retrieval Agent
# =====================================================================

def retrieval_agent(
    state: ResolveXState,
) -> dict:
    """
    Agentic Retrieval Agent.

    Phase 16.3 responsibilities:

        1. Analyze the retrieval query.
        2. Use an agent-generated retry query when available.
        3. Search the knowledge base through the retrieval tool.
        4. Evaluate whether additional historical-ticket evidence
           should be requested.
        5. Search previous tickets when the KB evidence is weak.
        6. Rerank the combined evidence using the cross-encoder.
        7. Build grounded LLM context.
        8. Produce structured retrieval metadata.
        9. Preserve retrieval-loop state.
        10. Emit tool-call and retrieval observability metadata.
        11. Preserve safe fallback semantics.

    Retrieval tools:

        search_knowledge_base()
        search_previous_tickets()
        rerank_documents()

    The Retrieval Decision Agent remains responsible for deciding
    whether another retrieval attempt should occur.
    """

    # =================================================================
    # 1. Read Existing State
    # =================================================================

    original_query = (
        state.get("cleaned_ticket", "")
        or ""
    )

    existing_retrieval_metadata = dict(
        state.get("retrieval_metadata", {})
    )

    graph_metadata = dict(
        state.get("metadata", {})
    )

    previous_attempt = int(
        graph_metadata.get(
            "retrieval_attempt",
            0,
        )
    )

    current_attempt = previous_attempt + 1

    retry_query = graph_metadata.get(
        "retrieval_next_query"
    )

    previous_decision = (
        existing_retrieval_metadata.get(
            "retrieval_decision"
        )
    )

    # =================================================================
    # 2. Query Analysis / Retry Query Selection
    # =================================================================

    if (
        retry_query
        and previous_decision == "retry"
    ):
        query = str(
            retry_query
        ).strip()

        query_analysis = analyze_query(
            query
        )

        query_source = "agent_retry_query"

    else:
        query_analysis = analyze_query(
            original_query
        )

        query = (
            query_analysis.retrieval_query
        )

        query_source = "query_analyzer"

    query = " ".join(
        query.split()
    ).strip()

    query_length = len(query)

    # =================================================================
    # 3. Empty Query Handling
    # =================================================================

    if not query:

        logger.warning(
            "Retrieval agent received empty query "
            f"for ticket_id={state.get('ticket_id')} "
            f"attempt={current_attempt}"
        )

        metadata = _stage_metadata(
            state,
            "retrieval_agent",
        )

        warnings = list(
            state.get("warnings", [])
        )

        warnings.append(
            "retrieval_agent: empty query"
        )

        retrieval_metadata = (
            build_retrieval_metadata(
                strategy=RETRIEVAL_STRATEGY,
                top_k=RETRIEVAL_TOP_K,
                candidate_k=RETRIEVAL_CANDIDATE_K,
                bm25_weight=BM25_WEIGHT,
                dense_weight=DENSE_WEIGHT,
                candidate_count=0,
                documents=[],
                status="empty",
                query_length=0,
                fallback_used=False,
            )
        )

        retrieval_metadata.update(
            {
                "retrieval_attempt": current_attempt,
                "query_source": query_source,
                "original_query": (
                    query_analysis.original_query
                ),
                "normalized_query": (
                    query_analysis.normalized_query
                ),
                "retrieval_query": "",
                "rewrite_needed": (
                    query_analysis.rewrite_needed
                ),
                "rewrite_strategy": (
                    query_analysis.rewrite_strategy
                ),
                "rewrite_reason": (
                    query_analysis.rewrite_reason
                ),
                "tool_calls": [],
                "knowledge_base_search_used": False,
                "previous_ticket_search_used": False,
                "reranking_used": False,
                "retrieval_decision": "continue",
                "enough_evidence": False,
                "evidence_reason": "empty_query",
                "retry_allowed": False,
                "retry_strategy": "none",
                "next_query": "",
                "termination_reason": (
                    "Retrieval terminated because "
                    "the query was empty."
                ),
                "top_score": 0.0,
                "score_gap": 0.0,
                "retrieval_top_score": 0.0,
                "retrieval_score_gap": 0.0,
                "reranker_top_score": 0.0,
                "reranker_score_gap": 0.0,
            }
        )

        metadata["retrieval_attempt"] = (
            current_attempt
        )

        metadata["retrieval_query_source"] = (
            query_source
        )

        return {
            "retrieved_context": "",
            "retrieved_documents": [],
            "retrieval_metadata": retrieval_metadata,
            "warnings": warnings,
            "metadata": metadata,
        }

    # =================================================================
    # 4. Tool-Based Retrieval
    # =================================================================

    try:

        # Retrieval tools are imported at module level intentionally.
        # This keeps the agent testable and allows dependency injection
        # through graph-level mocks.
        tool_calls: list[dict] = []

        # -------------------------------------------------------------
        # Tool 1 — Knowledge Base Search
        # -------------------------------------------------------------

        knowledge_base_documents = (
            search_knowledge_base(
                query,
                strategy=RETRIEVAL_STRATEGY,
                top_k=RETRIEVAL_CANDIDATE_K,
                candidate_k=RETRIEVAL_CANDIDATE_K,
            )
        )

        tool_calls.append(
            {
                "tool": "search_knowledge_base",
                "status": "success",
                "query": query,
                "result_count": len(
                    knowledge_base_documents
                ),
            }
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=retrieval_agent "
            f"tool=search_knowledge_base "
            f"query_source={query_source} "
            f"result_count="
            f"{len(knowledge_base_documents)} "
            f"attempt={current_attempt}"
        )

        # -------------------------------------------------------------
        # 5. Initial Evidence Assessment
        # -------------------------------------------------------------

        kb_top_score = 0.0

        if knowledge_base_documents:

            try:
                kb_top_score = max(
                    float(
                        document.get(
                            "score",
                            0.0,
                        )
                    )
                    for document
                    in knowledge_base_documents
                )

            except (
                TypeError,
                ValueError,
            ):
                kb_top_score = 0.0

        kb_evidence_sufficient = (
            len(knowledge_base_documents)
            >= MIN_RETRIEVAL_DOCUMENTS
            and kb_top_score
            >= MIN_RETRIEVAL_TOP_SCORE
        )

        # -------------------------------------------------------------
        # Tool 2 — Previous Ticket Search
        # -------------------------------------------------------------
        #
        # We only invoke this tool when the KB evidence is weak.
        #
        # This prevents unnecessary historical-ticket searches when
        # the knowledge base already contains sufficient evidence.
        #

        previous_ticket_documents: list[dict] = []

        if not kb_evidence_sufficient:

            try:

                previous_ticket_documents = (
                    search_previous_tickets(
                        query,
                        top_k=RETRIEVAL_TOP_K,
                    )
                )

                tool_calls.append(
                    {
                        "tool": (
                            "search_previous_tickets"
                        ),
                        "status": "success",
                        "query": query,
                        "result_count": len(
                            previous_ticket_documents
                        ),
                        "reason": (
                            "knowledge_base_evidence_insufficient"
                        ),
                    }
                )

                logger.info(
                    f"ticket_id={state.get('ticket_id')} "
                    f"stage=retrieval_agent "
                    f"tool=search_previous_tickets "
                    f"result_count="
                    f"{len(previous_ticket_documents)} "
                    f"reason=insufficient_kb_evidence"
                )

            except Exception as exc:

                logger.warning(
                    f"Previous-ticket search failed "
                    f"for ticket_id="
                    f"{state.get('ticket_id')}: "
                    f"{type(exc).__name__}: {exc}"
                )

                tool_calls.append(
                    {
                        "tool": (
                            "search_previous_tickets"
                        ),
                        "status": "failure",
                        "query": query,
                        "result_count": 0,
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

        # -------------------------------------------------------------
        # 6. Combine Evidence
        # -------------------------------------------------------------

        combined_documents: list[dict] = []

        combined_documents.extend(
            knowledge_base_documents
        )

        combined_documents.extend(
            previous_ticket_documents
        )

        # -------------------------------------------------------------
        # Deduplicate by index_id when available
        # -------------------------------------------------------------

        deduplicated_documents: list[dict] = []

        seen_document_ids: set[int] = set()

        for document in combined_documents:

            index_id = document.get(
                "index_id"
            )

            if index_id is None:
                deduplicated_documents.append(
                    document
                )
                continue

            try:
                normalized_id = int(
                    index_id
                )

            except (
                TypeError,
                ValueError,
            ):
                deduplicated_documents.append(
                    document
                )
                continue

            if normalized_id in seen_document_ids:
                continue

            seen_document_ids.add(
                normalized_id
            )

            deduplicated_documents.append(
                document
            )

        combined_documents = (
            deduplicated_documents
        )

        # Preserve the original retrieval score before reranking.
        # The reranker uses the legacy "score" field for its own score,
        # so retrieval_score must be captured before that happens.
        for document in combined_documents:
            document["retrieval_score"] = float(
                document.get("retrieval_score", document.get("score", 0.0))
                or 0.0
            )

        candidate_count = len(
            combined_documents
        )

        # =================================================================
        # 7. Tool 3 — Reranking
        # =================================================================

        reranked_documents: list[dict] = []

        if combined_documents:

            try:

                reranked_documents = (
                    rerank_documents(
                        query,
                        combined_documents,
                        top_k=RETRIEVAL_TOP_K,
                    )
                )

                tool_calls.append(
                    {
                        "tool": "rerank_documents",
                        "status": "success",
                        "query": query,
                        "input_count": (
                            len(
                                combined_documents
                            )
                        ),
                        "result_count": (
                            len(
                                reranked_documents
                            )
                        ),
                    }
                )

                logger.info(
                    f"ticket_id={state.get('ticket_id')} "
                    f"stage=retrieval_agent "
                    f"tool=rerank_documents "
                    f"input_count="
                    f"{len(combined_documents)} "
                    f"result_count="
                    f"{len(reranked_documents)} "
                    f"attempt={current_attempt}"
                )

            except Exception as exc:

                logger.warning(
                    f"Document reranking failed "
                    f"for ticket_id="
                    f"{state.get('ticket_id')}: "
                    f"{type(exc).__name__}: {exc}. "
                    f"Using original retrieval ordering."
                )

                tool_calls.append(
                    {
                        "tool": "rerank_documents",
                        "status": "failure",
                        "query": query,
                        "input_count": (
                            len(
                                combined_documents
                            )
                        ),
                        "result_count": 0,
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

                reranked_documents = (
                    combined_documents[
                        :RETRIEVAL_TOP_K
                    ]
                )

        # Explicitly preserve empty retrieval behavior.
        if not reranked_documents:
            context_docs: list[dict] = []
        else:
            context_docs = reranked_documents[
                :RETRIEVAL_TOP_K
            ]

        # Normalize retrieval/reranker scores after reranking.
        # "score" belongs to the most recent scoring stage, while the
        # explicit fields preserve both scoring systems independently.
        for document in context_docs:
            if "retrieval_score" not in document:
                document["retrieval_score"] = float(
                    document.get("score", 0.0) or 0.0
                )

            document["reranker_score"] = float(
                document.get("score", 0.0) or 0.0
            )

        # =================================================================
        # 8. Final Evidence Statistics
        # =================================================================

        retrieval_top_score = 0.0
        retrieval_second_score = 0.0

        reranker_top_score = 0.0
        reranker_second_score = 0.0

        if context_docs:
            try:
                retrieval_top_score = float(
                    context_docs[0].get(
                        "retrieval_score",
                        0.0,
                    )
                    or 0.0
                )

                reranker_top_score = float(
                    context_docs[0].get(
                        "reranker_score",
                        0.0,
                    )
                    or 0.0
                )

                if len(context_docs) > 1:
                    retrieval_second_score = float(
                        context_docs[1].get(
                            "retrieval_score",
                            0.0,
                        )
                        or 0.0
                    )

                    reranker_second_score = float(
                        context_docs[1].get(
                            "reranker_score",
                            0.0,
                        )
                        or 0.0
                    )

            except (
                TypeError,
                ValueError,
            ):
                retrieval_top_score = 0.0
                retrieval_second_score = 0.0
                reranker_top_score = 0.0
                reranker_second_score = 0.0

        retrieval_score_gap = (
            max(
                0.0,
                retrieval_top_score - retrieval_second_score,
            )
            if len(context_docs) > 1
            else retrieval_top_score
        )

        reranker_score_gap = (
            max(
                0.0,
                reranker_top_score - reranker_second_score,
            )
            if len(context_docs) > 1
            else reranker_top_score
        )

        # Backward-compatible aliases. These continue to represent
        # retrieval scores rather than CrossEncoder scores.
        top_score = retrieval_top_score
        score_gap = retrieval_score_gap

        # =================================================================
        # 9. Build Grounded LLM Context
        # =================================================================

        context_text = "\n\n---\n\n".join(
            (
                f"[{document.get('source', '').upper()}] "
                f"{document.get('title', '')}\n"
                f"{document.get('content', '')}"
            )
            for document in context_docs
        )

        # =================================================================
        # 10. Retrieval Status
        # =================================================================

        status = (
            "success"
            if context_docs
            else "empty"
        )

        # =================================================================
        # 11. Build Structured Retrieval Metadata
        # =================================================================

        retrieval_metadata = (
            build_retrieval_metadata(
                strategy=RETRIEVAL_STRATEGY,
                top_k=RETRIEVAL_TOP_K,
                candidate_k=RETRIEVAL_CANDIDATE_K,
                bm25_weight=BM25_WEIGHT,
                dense_weight=DENSE_WEIGHT,
                candidate_count=candidate_count,
                documents=context_docs,
                status=status,
                query_length=query_length,
                fallback_used=False,
            )
        )

        retrieval_metadata.update(
            {
                "retrieval_attempt": (
                    current_attempt
                ),
                "query_source": query_source,

                "original_query": (
                    query_analysis.original_query
                ),

                "normalized_query": (
                    query_analysis.normalized_query
                ),

                "retrieval_query": query,

                "rewrite_needed": (
                    query_analysis.rewrite_needed
                ),

                "rewrite_strategy": (
                    query_analysis.rewrite_strategy
                ),

                "rewrite_reason": (
                    query_analysis.rewrite_reason
                ),

                # -----------------------------------------------------
                # Tool observability
                # -----------------------------------------------------

                "tool_calls": tool_calls,

                "knowledge_base_search_used": True,

                "knowledge_base_result_count": (
                    len(
                        knowledge_base_documents
                    )
                ),

                "previous_ticket_search_used": (
                    not kb_evidence_sufficient
                ),

                "previous_ticket_result_count": (
                    len(
                        previous_ticket_documents
                    )
                ),

                "reranking_used": bool(
                    combined_documents
                ),

                "reranked_result_count": (
                    len(context_docs)
                ),

                # -----------------------------------------------------
                # Evidence statistics
                # -----------------------------------------------------

                "document_count": len(
                    context_docs
                ),

                "candidate_count": (
                    candidate_count
                ),

                # Backward-compatible retrieval statistics.
                "top_score": top_score,
                "score_gap": score_gap,

                # Explicit retrieval-stage statistics.
                "retrieval_top_score": retrieval_top_score,
                "retrieval_score_gap": retrieval_score_gap,

                # Explicit reranking-stage statistics.
                "reranker_top_score": reranker_top_score,
                "reranker_score_gap": reranker_score_gap,

                "initial_kb_top_score": (
                    kb_top_score
                ),

                "initial_kb_evidence_sufficient": (
                    kb_evidence_sufficient
                ),
            }
        )

        # -------------------------------------------------------------
        # Preserve previous retrieval-loop decision
        # -------------------------------------------------------------

        if previous_decision:

            retrieval_metadata[
                "previous_retrieval_decision"
            ] = previous_decision

        if retry_query:

            retrieval_metadata[
                "previous_retry_query"
            ] = retry_query

        # =================================================================
        # 12. Observability
        # =================================================================

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=retrieval_agent "
            f"strategy={RETRIEVAL_STRATEGY} "
            f"status={status} "
            f"attempt={current_attempt} "
            f"query_source={query_source} "
            f"kb_results="
            f"{len(knowledge_base_documents)} "
            f"previous_ticket_results="
            f"{len(previous_ticket_documents)} "
            f"candidate_count={candidate_count} "
            f"result_count={len(context_docs)} "
            f"retrieval_top_score={retrieval_top_score:.4f} "
            f"retrieval_score_gap={retrieval_score_gap:.4f} "
            f"reranker_top_score={reranker_top_score:.4f} "
            f"reranker_score_gap={reranker_score_gap:.4f} "
            f"reranking_used="
            f"{bool(combined_documents)} "
            f"rewrite_needed="
            f"{query_analysis.rewrite_needed} "
            f"rewrite_strategy="
            f"{query_analysis.rewrite_strategy} "
            f"request_id={state.get('request_id')} "
            f"graph_run_id={state.get('graph_run_id')}"
        )

        # =================================================================
        # 13. Stage Metadata
        # =================================================================

        metadata = _stage_metadata(
            state,
            "retrieval_agent",
        )

        metadata[
            "retrieval_attempt"
        ] = current_attempt

        metadata[
            "retrieval_query_source"
        ] = query_source

        metadata[
            "retrieval_query"
        ] = query

        metadata[
            "retrieval_document_count"
        ] = len(context_docs)

        metadata[
            "retrieval_tool_count"
        ] = len(tool_calls)

        metadata[
            "retrieval_reranking_used"
        ] = bool(combined_documents)

        # =================================================================
        # 14. Return Successful Retrieval
        # =================================================================

        return {
            "retrieved_context": context_text,
            "retrieved_documents": context_docs,
            "retrieval_metadata": retrieval_metadata,
            "metadata": metadata,
        }

    except Exception as exc:

        # =================================================================
        # 15. Safe Retrieval Failure
        # =================================================================

        logger.exception(
            f"Retrieval agent failed for "
            f"ticket_id={state.get('ticket_id')} "
            f"attempt={current_attempt}: {exc}"
        )

        errors = _append_error(
            state,
            "retrieval_agent",
            exc,
        )

        metadata = _stage_metadata(
            state,
            "retrieval_agent",
        )

        metadata[
            "retrieval_attempt"
        ] = current_attempt

        metadata[
            "retrieval_query_source"
        ] = query_source

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        # Preserve the explicit configuration error contract expected by
        # retrieval tests and callers.
        if (
            isinstance(exc, ValueError)
            and "Unsupported knowledge-base retrieval strategy" in str(exc)
        ):
            error_message = (
                "ValueError: Unsupported RETRIEVAL_STRATEGY: "
                f"{RETRIEVAL_STRATEGY}"
            )

        retrieval_metadata = (
            build_retrieval_metadata(
                strategy=RETRIEVAL_STRATEGY,
                top_k=RETRIEVAL_TOP_K,
                candidate_k=RETRIEVAL_CANDIDATE_K,
                bm25_weight=BM25_WEIGHT,
                dense_weight=DENSE_WEIGHT,
                candidate_count=0,
                documents=[],
                status="failure",
                query_length=query_length,
                fallback_used=True,
                error=error_message,
            )
        )

        retrieval_metadata.update(
            {
                "retrieval_attempt": (
                    current_attempt
                ),

                "query_source": (
                    query_source
                ),

                "original_query": (
                    query_analysis.original_query
                ),

                "normalized_query": (
                    query_analysis.normalized_query
                ),

                "retrieval_query": query,

                "rewrite_needed": (
                    query_analysis.rewrite_needed
                ),

                "rewrite_strategy": (
                    query_analysis.rewrite_strategy
                ),

                "rewrite_reason": (
                    query_analysis.rewrite_reason
                ),

                "tool_calls": [],

                "knowledge_base_search_used": (
                    False
                ),

                "previous_ticket_search_used": (
                    False
                ),

                "reranking_used": False,

                "document_count": 0,

                "candidate_count": 0,

                "enough_evidence": False,

                "evidence_reason": (
                    "Retrieval tool execution failed."
                ),

                "retry_allowed": False,

                "retry_strategy": "none",

                "next_query": "",

                "retrieval_decision": "continue",

                "termination_reason": (
                    "Retrieval failed and the "
                    "current execution cannot safely "
                    "perform another retrieval."
                ),
                "top_score": 0.0,
                "score_gap": 0.0,
                "retrieval_top_score": 0.0,
                "retrieval_score_gap": 0.0,
                "reranker_top_score": 0.0,
                "reranker_score_gap": 0.0,
            }
        )

        if previous_decision:

            retrieval_metadata[
                "previous_retrieval_decision"
            ] = previous_decision

        if retry_query:

            retrieval_metadata[
                "previous_retry_query"
            ] = retry_query

        logger.error(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=retrieval_agent "
            f"status=failure "
            f"attempt={current_attempt} "
            f"query_source={query_source} "
            f"request_id={state.get('request_id')} "
            f"graph_run_id={state.get('graph_run_id')}"
        )

        return {
            "retrieved_context": "",
            "retrieved_documents": [],
            "retrieval_metadata": retrieval_metadata,
            "errors": errors,
            "fallback_used": True,
            "metadata": metadata,
        }

# =====================================================================
# Retrieval Decision Agent
# =====================================================================


def retrieval_decision_agent(
    state: ResolveXState,
) -> dict:
    """
    Decide whether the retrieved evidence is sufficient for downstream
    diagnosis or whether another retrieval attempt is required.

    Responsibilities:
        1. Evaluate retrieved evidence quality.
        2. Determine whether enough evidence exists.
        3. Decide whether retrieval should be retried.
        4. Select a deterministic retry strategy.
        5. Generate a retry query.
        6. Enforce a maximum retrieval-attempt limit.
        7. Persist retrieval-loop metadata.
        8. Emit observability signals.

    Decision values:
        - continue
        - retry

    Retry strategies:
        - broaden
        - narrow
        - rewrite
        - none
    """

    retrieval_metadata = dict(
        state.get("retrieval_metadata", {})
    )

    metadata = dict(
        state.get("metadata", {})
    )

    errors = list(
        state.get("errors", [])
    )

    warnings = list(
        state.get("warnings", [])
    )

    # -----------------------------------------------------------------
    # 1. Read retrieval state
    # -----------------------------------------------------------------

    documents = list(
        state.get("retrieved_documents", [])
        or []
    )

    current_query = (
        retrieval_metadata.get("retrieval_query")
        or retrieval_metadata.get("normalized_query")
        or state.get("cleaned_ticket", "")
        or ""
    )

    top_score = retrieval_metadata.get(
        "top_score"
    )

    score_gap = retrieval_metadata.get(
        "score_gap"
    )

    retrieval_status = retrieval_metadata.get(
        "status",
        "unknown",
    )

    previous_attempt = int(
        metadata.get(
            "retrieval_attempt",
            0,
        )
    )

    current_attempt = previous_attempt + 1

    # -----------------------------------------------------------------
    # 2. Normalize score values safely
    # -----------------------------------------------------------------

    try:
        top_score_value = (
            float(top_score)
            if top_score is not None
            else 0.0
        )
    except (TypeError, ValueError):
        top_score_value = 0.0

    try:
        score_gap_value = (
            float(score_gap)
            if score_gap is not None
            else 0.0
        )
    except (TypeError, ValueError):
        score_gap_value = 0.0

    document_count = len(documents)

    # -----------------------------------------------------------------
    # 3. Evidence quality assessment
    # -----------------------------------------------------------------

    enough_documents = (
        document_count
        >= MIN_RETRIEVAL_DOCUMENTS
    )

    sufficient_top_score = (
        top_score_value
        >= MIN_RETRIEVAL_TOP_SCORE
    )

    retrieval_successful = (
        retrieval_status == "success"
        and document_count > 0
    )

    enough_evidence = (
        retrieval_successful
        and enough_documents
        and sufficient_top_score
    )

    # -----------------------------------------------------------------
    # 4. Determine evidence quality reason
    # -----------------------------------------------------------------

    if retrieval_status == "failure":
        evidence_reason = (
            "Retrieval failed."
        )

    elif document_count == 0:
        evidence_reason = (
            "No documents were retrieved."
        )

    elif not enough_documents:
        evidence_reason = (
            f"Only {document_count} document(s) "
            f"were retrieved; at least "
            f"{MIN_RETRIEVAL_DOCUMENTS} are preferred."
        )

    elif not sufficient_top_score:
        evidence_reason = (
            f"Top retrieval score "
            f"{top_score_value:.4f} is below "
            f"the minimum evidence threshold "
            f"{MIN_RETRIEVAL_TOP_SCORE:.4f}."
        )

    else:
        evidence_reason = (
            "Retrieved evidence satisfies the "
            "minimum quality requirements."
        )

    # -----------------------------------------------------------------
    # 5. Decide whether another retrieval is allowed
    # -----------------------------------------------------------------

    retry_allowed = (
        not enough_evidence
        and current_attempt < MAX_RETRIEVAL_ATTEMPTS
    )

    # -----------------------------------------------------------------
    # 6. Select retry strategy
    # -----------------------------------------------------------------

    retry_strategy = "none"
    next_query = current_query

    if retry_allowed:

        rewrite_needed = bool(
            retrieval_metadata.get(
                "rewrite_needed",
                False,
            )
        )

        original_query = (
            retrieval_metadata.get(
                "original_query"
            )
            or state.get(
                "cleaned_ticket",
                "",
            )
            or ""
        )

        normalized_query = (
            retrieval_metadata.get(
                "normalized_query"
            )
            or current_query
        )

        # -------------------------------------------------------------
        # Strategy 1: Existing query rewrite was not sufficient
        # -------------------------------------------------------------

        if rewrite_needed and (
            current_query.strip()
            != original_query.strip()
        ):
            retry_strategy = "broaden"

            next_query = (
                f"{normalized_query} "
                f"troubleshooting solution "
                f"resolution"
            )

        # -------------------------------------------------------------
        # Strategy 2: Query is already reasonably specific
        # -------------------------------------------------------------

        elif len(
            normalized_query.split()
        ) >= 8:

            retry_strategy = "narrow"

            words = normalized_query.split()

            # Keep the most informative portion while
            # preventing an empty query.
            next_query = " ".join(
                words[: min(8, len(words))]
            )

        # -------------------------------------------------------------
        # Strategy 3: Short / ambiguous query
        # -------------------------------------------------------------

        else:
            retry_strategy = "rewrite"

            next_query = (
                f"{normalized_query} "
                f"IT support issue"
            )

        next_query = " ".join(
            next_query.split()
        ).strip()

        if not next_query:
            retry_strategy = "none"
            retry_allowed = False
            next_query = current_query

    # -----------------------------------------------------------------
    # 7. Final decision
    # -----------------------------------------------------------------

    if enough_evidence:
        decision = "continue"

        termination_reason = (
            "Sufficient retrieval evidence "
            "was found."
        )

    elif retry_allowed:
        decision = "retry"

        termination_reason = (
            "Retrieved evidence was insufficient; "
            "another retrieval attempt is allowed."
        )

    else:
        decision = "continue"

        termination_reason = (
            "Retrieval evidence remained insufficient "
            "after the maximum allowed attempts. "
            "Downstream diagnosis will proceed with "
            "the best available evidence."
        )

        if current_attempt >= MAX_RETRIEVAL_ATTEMPTS:
            warnings.append(
                "retrieval_agent: maximum retrieval "
                "attempts reached"
            )

    # -----------------------------------------------------------------
    # 8. Build retrieval loop metadata
    # -----------------------------------------------------------------

    retrieval_metadata.update(
        {
            "retrieval_attempt": current_attempt,
            "max_retrieval_attempts": (
                MAX_RETRIEVAL_ATTEMPTS
            ),
            "document_count": document_count,
            "top_score": top_score_value,
            "score_gap": score_gap_value,
            "enough_documents": enough_documents,
            "sufficient_top_score": (
                sufficient_top_score
            ),
            "enough_evidence": enough_evidence,
            "evidence_reason": evidence_reason,
            "retry_allowed": retry_allowed,
            "retry_strategy": retry_strategy,
            "next_query": next_query,
            "retrieval_decision": decision,
            "termination_reason": (
                termination_reason
            ),
        }
    )

    # -----------------------------------------------------------------
    # 9. Persist retry control state
    # -----------------------------------------------------------------

    updated_metadata = dict(metadata)

    updated_metadata[
        "last_stage"
    ] = "retrieval_decision_agent"

    updated_metadata[
        "retrieval_attempt"
    ] = current_attempt

    updated_metadata[
        "retrieval_decision"
    ] = decision

    updated_metadata[
        "retrieval_evidence_sufficient"
    ] = enough_evidence

    updated_metadata[
        "retrieval_retry_strategy"
    ] = retry_strategy

    updated_metadata[
        "retrieval_next_query"
    ] = next_query

    updated_metadata[
        "retrieval_termination_reason"
    ] = termination_reason

    updated_metadata[
        "updated_at"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    # -----------------------------------------------------------------
    # 10. Logging / Observability
    # -----------------------------------------------------------------

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=retrieval_decision_agent "
        f"attempt={current_attempt} "
        f"documents={document_count} "
        f"top_score={top_score_value:.4f} "
        f"score_gap={score_gap_value:.4f} "
        f"enough_evidence={enough_evidence} "
        f"decision={decision} "
        f"retry_allowed={retry_allowed} "
        f"retry_strategy={retry_strategy} "
        f"next_query={next_query!r} "
        f"request_id={state.get('request_id')} "
        f"graph_run_id={state.get('graph_run_id')}"
    )

    # -----------------------------------------------------------------
    # 11. Return updated state
    # -----------------------------------------------------------------

    result = {
        "retrieval_metadata": retrieval_metadata,
        "metadata": updated_metadata,
        "warnings": warnings,
    }

    # -----------------------------------------------------------------
    # 12. Preserve errors
    # -----------------------------------------------------------------

    if errors:
        result["errors"] = errors

    return result



# =====================================================================
# Retrieval Decision Routing
# =====================================================================


def route_retrieval_decision(
    state: ResolveXState,
) -> str:
    """
    Route the graph after retrieval quality evaluation.

    Returns:
        continue -> proceed to diagnosis
        retry    -> execute retrieval again
    """

    retrieval_metadata = state.get(
        "retrieval_metadata",
        {},
    ) or {}

    decision = retrieval_metadata.get(
        "retrieval_decision",
        "continue",
    )

    if decision == "retry":
        return "retry"

    return "continue"
# =====================================================================
# Diagnosis Agent
# =====================================================================


def diagnosis_agent(
    state: ResolveXState,
) -> dict:
    """
    Diagnosis Agent.

    Determines what is actually happening and identifies the
    most likely evidence-supported root cause.

    The agent consumes:

        - cleaned ticket
        - classification
        - retrieved evidence

    It does NOT generate resolution instructions.
    """

    ticket_text = (
        state.get("cleaned_ticket", "")
        or ""
    )

    context = (
        state.get("retrieved_context", "")
        or ""
    )

    classification = (
        state.get("category", "")
        or ""
    )

    try:

        diagnosis_result, fallback_used = (
            generate_diagnosis(
                ticket_text=ticket_text,
                context=context,
                classification=classification,
            )
        )

        existing_fallback = bool(
            state.get(
                "fallback_used",
                False,
            )
        )

        combined_fallback = (
            existing_fallback
            or fallback_used
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=diagnosis_agent "
            f"confidence="
            f"{diagnosis_result.confidence:.3f} "
            f"evidence_count="
            f"{len(diagnosis_result.evidence)} "
            f"missing_information_count="
            f"{len(diagnosis_result.missing_information)} "
            f"fallback={fallback_used}"
        )

        metadata = _stage_metadata(
            state,
            "diagnosis_agent",
        )

        metadata["diagnosis"] = {
            "confidence": float(
                diagnosis_result.confidence
            ),
            "evidence_count": len(
                diagnosis_result.evidence
            ),
            "missing_information_count": len(
                diagnosis_result.missing_information
            ),
            "fallback_used": fallback_used,
        }

        return {
            # New structured diagnosis contract
            "diagnosis_result": (
                diagnosis_result.model_dump()
            ),

            # New explicit state fields
            "diagnosis_problem": (
                diagnosis_result.problem
            ),
            "diagnosis_root_cause": (
                diagnosis_result.possible_root_cause
            ),
            "diagnosis_evidence": (
                diagnosis_result.evidence
            ),
            "diagnosis_missing_information": (
                diagnosis_result.missing_information
            ),
            "diagnosis_confidence": float(
                diagnosis_result.confidence
            ),

            # Backward-compatible fields
            "diagnosis": (
                diagnosis_result.problem
            ),
            "root_cause": (
                diagnosis_result.possible_root_cause
            ),

            "fallback_used": combined_fallback,
            "metadata": metadata,
        }

    except Exception as exc:

        logger.exception(
            f"Diagnosis agent failed for "
            f"ticket_id={state.get('ticket_id')}: {exc}"
        )

        errors = _append_error(
            state,
            "diagnosis_agent",
            exc,
        )

        metadata = _stage_metadata(
            state,
            "diagnosis_agent",
        )

        fallback_result = {
            "problem": (
                "The reported support issue requires "
                "further investigation."
            ),
            "possible_root_cause": (
                "No validated root cause could be "
                "established from the available evidence."
            ),
            "evidence": [],
            "missing_information": [
                "Additional diagnostic information is required."
            ],
            "confidence": 0.0,
        }

        return {
            "diagnosis_result": fallback_result,

            "diagnosis_problem": (
                fallback_result["problem"]
            ),
            "diagnosis_root_cause": (
                fallback_result["possible_root_cause"]
            ),
            "diagnosis_evidence": [],
            "diagnosis_missing_information": (
                fallback_result["missing_information"]
            ),
            "diagnosis_confidence": 0.0,

            # Backward compatibility
            "diagnosis": fallback_result["problem"],
            "root_cause": (
                fallback_result["possible_root_cause"]
            ),

            "fallback_used": True,
            "errors": errors,
            "metadata": metadata,
        }


# =====================================================================
# Resolution Agent
# =====================================================================


def resolution_agent(
    state: ResolveXState,
) -> dict:
    """
    Generate a structured resolution using ticket,
    classification, retrieved context, diagnosis,
    and conversation history.
    """

    ticket_text = (
        state.get("cleaned_ticket", "")
        or ""
    )

    context = (
        state.get("retrieved_context", "")
        or ""
    )

    diagnosis = (
        state.get("diagnosis", "")
        or state.get("diagnosis_problem", "")
        or ""
    )

    root_cause = (
        state.get("root_cause", "")
        or state.get("diagnosis_root_cause", "")
        or ""
    )

    classification = (
        state.get("category", "")
        or ""
    )

    conversation_history = (
        state.get("conversation_history", "")
        or ""
    )

    try:

        resolution, fallback_used = (
            generate_solution(
                ticket_text=ticket_text,
                context=context,
                diagnosis=diagnosis,
                root_cause=root_cause,
                classification=classification,
                conversation_history=conversation_history,
            )
        )

        existing_fallback = bool(
            state.get(
                "fallback_used",
                False,
            )
        )

        combined_fallback = (
            existing_fallback
            or fallback_used
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=resolution_agent "
            f"confidence={resolution.confidence:.3f} "
            f"requires_human={resolution.requires_human} "
            f"diagnosis_available={bool(diagnosis)} "
            f"root_cause_available={bool(root_cause)} "
            f"fallback={fallback_used}"
        )

        metadata = _stage_metadata(
            state,
            "resolution_agent",
        )

        return {
            "resolution_steps": (
                resolution.resolution_steps
            ),
            "evidence": resolution.evidence,
            "resolution_confidence": float(
                resolution.confidence
            ),
            "requires_human": bool(
                resolution.requires_human
            ),
            "fallback_used": combined_fallback,
            "metadata": metadata,
        }

    except Exception as exc:

        logger.exception(
            f"Resolution agent failed for "
            f"ticket_id={state.get('ticket_id')}: {exc}"
        )

        errors = _append_error(
            state,
            "resolution_agent",
            exc,
        )

        metadata = _stage_metadata(
            state,
            "resolution_agent",
        )

        return {
            "resolution_steps": [
                "Review the ticket details.",
                "Collect relevant system or application logs.",
                "Escalate the ticket for human investigation.",
            ],
            "evidence": [],
            "resolution_confidence": 0.0,
            "requires_human": True,
            "fallback_used": True,
            "errors": errors,
            "metadata": metadata,
        }


# =====================================================================
# Verification Agent
# =====================================================================


def verification_agent(
    state: ResolveXState,
) -> dict:
    """
    Verify whether the generated resolution is sufficiently
    supported by the ticket, diagnosis, and retrieved evidence.

    The detailed verification dimensions are preserved in state,
    while verification_passed remains the compatibility gate used
    by downstream decision logic.
    """

    ticket_text = (
        state.get("cleaned_ticket", "")
        or ""
    )

    diagnosis = (
        state.get("diagnosis", "")
        or ""
    )

    root_cause = (
        state.get("root_cause", "")
        or ""
    )

    resolution_steps = (
        state.get("resolution_steps", [])
        or []
    )

    context = (
        state.get("retrieved_context", "")
        or ""
    )

    try:

        verification_result, fallback_used = (
            verify_resolution(
                ticket_text=ticket_text,
                diagnosis=diagnosis,
                root_cause=root_cause,
                resolution_steps=resolution_steps,
                context=context,
            )
        )

        existing_fallback = bool(
            state.get(
                "fallback_used",
                False,
            )
        )

        combined_fallback = (
            existing_fallback
            or fallback_used
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=verification_agent "
            f"passed={verification_result.verification_passed} "
            f"evidence={verification_result.supported_by_evidence} "
            f"hallucination={verification_result.hallucination_detected} "
            f"complete={verification_result.complete} "
            f"policy={verification_result.policy_compliant} "
            f"correct={verification_result.resolution_correct} "
            f"confidence={verification_result.confidence:.3f} "
            f"fallback={fallback_used}"
        )

        metadata = _stage_metadata(
            state,
            "verification_agent",
        )

        return {
            # ----------------------------------------------------------
            # Existing compatibility fields
            # ----------------------------------------------------------
            "verification_passed": bool(
                verification_result.verification_passed
            ),
            "verification_reason": (
                verification_result.verification_reason
            ),
            "verification_confidence": float(
                verification_result.confidence
            ),

            # ----------------------------------------------------------
            # Detailed verification dimensions
            # ----------------------------------------------------------
            "verification_supported_by_evidence": bool(
                verification_result.supported_by_evidence
            ),
            "verification_hallucination_detected": bool(
                verification_result.hallucination_detected
            ),
            "verification_complete": bool(
                verification_result.complete
            ),
            "verification_policy_compliant": bool(
                verification_result.policy_compliant
            ),
            "verification_resolution_correct": bool(
                verification_result.resolution_correct
            ),

            # Full structured result for observability/evaluation.
            "verification_result": (
                verification_result.model_dump()
            ),

            "fallback_used": combined_fallback,
            "metadata": metadata,
        }

    except Exception as exc:

        logger.exception(
            f"Verification agent failed for "
            f"ticket_id={state.get('ticket_id')}: {exc}"
        )

        errors = _append_error(
            state,
            "verification_agent",
            exc,
        )

        metadata = _stage_metadata(
            state,
            "verification_agent",
        )

        return {
            # Existing compatibility fields
            "verification_passed": False,
            "verification_reason": (
                "Resolution verification failed. "
                "Human review is required."
            ),
            "verification_confidence": 0.0,

            # Fail closed for every detailed verification dimension.
            "verification_supported_by_evidence": False,
            "verification_hallucination_detected": True,
            "verification_complete": False,
            "verification_policy_compliant": False,
            "verification_resolution_correct": False,

            "verification_result": {
                "supported_by_evidence": False,
                "hallucination_detected": True,
                "complete": False,
                "policy_compliant": False,
                "resolution_correct": False,
                "confidence": 0.0,
                "verification_passed": False,
                "verification_reason": (
                    "Resolution verification failed. "
                    "Human review is required."
                ),
            },

            "fallback_used": True,
            "errors": errors,
            "metadata": metadata,
        }


# =====================================================================
# Decision Agent
# =====================================================================


def decision_agent(
    state: ResolveXState,
) -> dict:
    """
    Make the final deterministic resolution decision.

    Decision outcomes:

    - auto_resolve
    - ask_clarification
    - human_review
    - escalate

    The policy is deterministic and acts as the safety boundary
    before automated resolution.
    """

    AUTO_RESOLVE_THRESHOLD = 0.75

    diagnosis_confidence = float(
        state.get(
            "diagnosis_confidence",
            0.0,
        )
    )

    resolution_confidence = float(
        state.get(
            "resolution_confidence",
            0.0,
        )
    )

    verification_confidence = float(
        state.get(
            "verification_confidence",
            0.0,
        )
    )

    verification_passed = bool(
        state.get(
            "verification_passed",
            False,
        )
    )

    requires_human = bool(
        state.get(
            "requires_human",
            False,
        )
    )

    fallback_used = bool(
        state.get(
            "fallback_used",
            False,
        )
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    warnings = list(
        state.get(
            "warnings",
            [],
        )
    )

    missing_information = list(
        state.get(
            "diagnosis_missing_information",
            [],
        )
        or []
    )

    metadata = _stage_metadata(
        state,
        "decision_agent",
    )

    # ---------------------------------------------------------------
    # 1. Critical execution errors
    # ---------------------------------------------------------------

    if errors:

        reason = (
            "Automated processing encountered one or more "
            "errors: "
            + "; ".join(errors)
        )

        logger.warning(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=escalate "
            f"reason=processing_errors"
        )

        return {
            "decision": "escalate",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 2. Explicit human requirement
    # ---------------------------------------------------------------

    if requires_human:

        reason = (
            "The generated resolution explicitly requires "
            "human intervention."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=requires_human"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 3. Missing information
    # ---------------------------------------------------------------

    if missing_information:

        reason = (
            "Additional information is required before the "
            "resolution can be safely completed: "
            + "; ".join(
                str(item)
                for item in missing_information
                if str(item).strip()
            )
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=ask_clarification "
            f"reason=missing_information"
        )

        return {
            "decision": "ask_clarification",
            "requires_human": False,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 4. Verification failure
    # ---------------------------------------------------------------

    if not verification_passed:

        reason = (
            "The proposed resolution did not pass the "
            "verification gate."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=verification_failed"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 5. Verification confidence
    # ---------------------------------------------------------------

    if (
        verification_confidence
        < AUTO_RESOLVE_THRESHOLD
    ):

        reason = (
            "Verification confidence is below the "
            f"auto-resolution threshold of "
            f"{AUTO_RESOLVE_THRESHOLD:.2f}."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=low_verification_confidence"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 6. Diagnosis confidence
    # ---------------------------------------------------------------

    if (
        diagnosis_confidence
        < AUTO_RESOLVE_THRESHOLD
    ):

        reason = (
            "Diagnosis confidence is below the "
            f"auto-resolution threshold of "
            f"{AUTO_RESOLVE_THRESHOLD:.2f}."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=low_diagnosis_confidence"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 7. Resolution confidence
    # ---------------------------------------------------------------

    if (
        resolution_confidence
        < AUTO_RESOLVE_THRESHOLD
    ):

        reason = (
            "Resolution confidence is below the "
            f"auto-resolution threshold of "
            f"{AUTO_RESOLVE_THRESHOLD:.2f}."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=low_resolution_confidence"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 8. Fallback protection
    # ---------------------------------------------------------------

    if fallback_used:

        reason = (
            "One or more AI pipeline components used a fallback "
            "path. Automatic resolution is therefore disabled."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=fallback_used"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 9. Warnings
    # ---------------------------------------------------------------

    if warnings:

        reason = (
            "The automated pipeline produced warnings that "
            "prevent automatic resolution: "
            + "; ".join(
                str(item)
                for item in warnings
                if str(item).strip()
            )
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=pipeline_warnings"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 10. All safety gates passed
    # --------------------------------------------------------------- 

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=decision_agent "
        f"decision=auto_resolve "
        f"reason=all_gates_passed"
    )

    return {
        "decision": "auto_resolve",
        "requires_human": False,
        "escalation_reason": "",
        "metadata": metadata,
    }

    # ---------------------------------------------------------------
    # 5. Diagnosis confidence
    # ---------------------------------------------------------------

    if (
        diagnosis_confidence
        < AUTO_RESOLVE_THRESHOLD
    ):

        reason = (
            "Diagnosis confidence is below the "
            f"auto-resolution threshold of "
            f"{AUTO_RESOLVE_THRESHOLD:.2f}."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=low_diagnosis_confidence"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 6. Resolution confidence
    # ---------------------------------------------------------------

    if (
        resolution_confidence
        < AUTO_RESOLVE_THRESHOLD
    ):

        reason = (
            "Resolution confidence is below the "
            f"auto-resolution threshold of "
            f"{AUTO_RESOLVE_THRESHOLD:.2f}."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=low_resolution_confidence"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 7. Fallback protection
    # ---------------------------------------------------------------

    if fallback_used:

        reason = (
            "One or more AI components used a fallback path. "
            "Automatic resolution is therefore disabled."
        )

        logger.info(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"decision=human_review "
            f"reason=fallback_used"
        )

        return {
            "decision": "human_review",
            "requires_human": True,
            "escalation_reason": reason,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------
    # 8. Warning information
    # ---------------------------------------------------------------

    if warnings:

        logger.warning(
            f"ticket_id={state.get('ticket_id')} "
            f"stage=decision_agent "
            f"warnings={len(warnings)}"
        )

    # ---------------------------------------------------------------
    # 9. Automatic resolution
    # ---------------------------------------------------------------

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=decision_agent "
        f"decision=auto_resolve "
        f"diagnosis_confidence={diagnosis_confidence:.3f} "
        f"resolution_confidence={resolution_confidence:.3f} "
        f"verification_confidence={verification_confidence:.3f}"
    )

    return {
        "decision": "auto_resolve",
        "requires_human": False,
        "escalation_reason": "",
        "metadata": metadata,
    }


# =====================================================================
# Terminal Nodes
# =====================================================================


def auto_resolve(
    state: ResolveXState,
) -> dict:
    """
    Terminal node for safely verified automatic resolution.
    """

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=auto_resolve "
        f"decision=auto_resolve"
    )

    metadata = _stage_metadata(
        state,
        "auto_resolve",
        completed=True,
    )

    return {
        "decision": "auto_resolve",
        "requires_human": False,
        "metadata": metadata,
    }


def human_review(
    state: ResolveXState,
) -> dict:
    """
    Terminal node for tickets requiring human review.
    """

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=human_review "
        f"decision=human_review"
    )

    metadata = _stage_metadata(
        state,
        "human_review",
        completed=True,
    )

    return {
        "decision": "human_review",
        "requires_human": True,
        "metadata": metadata,
    }

def ask_clarification(
    state: ResolveXState,
) -> dict:
    """
    Terminal node for tickets requiring additional information
    from the requester before resolution can safely continue.
    """

    logger.info(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=ask_clarification "
        f"decision=ask_clarification"
    )

    metadata = _stage_metadata(
        state,
        "ask_clarification",
        completed=True,
    )

    return {
        "decision": "ask_clarification",
        "requires_human": False,
        "metadata": metadata,
    }

def escalate(
    state: ResolveXState,
) -> dict:
    """
    Terminal node for tickets requiring escalation.
    """

    logger.warning(
        f"ticket_id={state.get('ticket_id')} "
        f"stage=escalate "
        f"decision=escalate"
    )

    metadata = _stage_metadata(
        state,
        "escalate",
        completed=True,
    )

    return {
        "decision": "escalate",
        "requires_human": True,
        "metadata": metadata,
    }


# =====================================================================
# Conditional Routing
# =====================================================================

def route_decision(
    state: ResolveXState,
) -> str:
    """
    Route the graph after the decision node.

    Explicit routing values keep the graph deterministic
    and easy to test.
    """

    decision = state.get(
        "decision",
        "human_review",
    )

    if decision == "auto_resolve":
        return "auto_resolve"

    if decision == "ask_clarification":
        return "ask_clarification"

    if decision == "escalate":
        return "escalate"

    if decision == "human_review":
        return "human_review"

    # Fail closed for unknown decision values.
    return "human_review"

resolvex_graph = build_resolvex_graph()