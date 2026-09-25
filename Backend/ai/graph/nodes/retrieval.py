from ai.graph import state_utils


from datetime import datetime, timezone
from typing import Any

from ai.agents.query_analyzer import analyze_query
from ai.agents.retrieval_metadata import build_retrieval_metadata
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
from ai.graph.nodes.initialization import _append_error, _stage_metadata
from ai.graph.state import ResolveXState
from ai.rag.doc_store import get_doc_store
from ai.rag.hybrid_retriever import HybridRetriever
from ai.rag.retriever import retrieve_context
from app.core.logger import logger
from ai.graph.tools.executor import execute_tool
from ai.graph.tools.tool_types import (
    ToolCategory,
    ToolDefinition,
    ToolRegistry,
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

def _build_agent_retrieval_registry() -> ToolRegistry:
    """
    Build the retrieval-tool registry using the current module-level
    handlers.

    The handlers are intentionally resolved when this function runs.
    This preserves test-time dependency injection and monkeypatching.
    """

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="search_knowledge_base",
            description=(
                "Search the ResolveX knowledge base for "
                "relevant technical documentation."
            ),
            category=ToolCategory.READ,
            handler=search_knowledge_base,
        )
    )

    registry.register(
        ToolDefinition(
            name="search_previous_tickets",
            description=(
                "Search historical tickets for similar "
                "incidents and resolutions."
            ),
            category=ToolCategory.READ,
            handler=search_previous_tickets,
        )
    )

    registry.register(
        ToolDefinition(
            name="rerank_documents",
            description=(
                "Rerank retrieved evidence using the "
                "configured relevance reranker."
            ),
            category=ToolCategory.READ,
            handler=rerank_documents,
        )
    )

    return registry
# =====================================================================
# Retrieval Agent
# =====================================================================


def retrieval_agent(
    state: ResolveXState,
) -> dict[str, Any]:
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

    Phase 19.2 responsibilities:

        12. Execute retrieval tools through ToolRegistry.
        13. Record normalized tool-call telemetry in ResolveXState.
        14. Preserve the existing Phase 16 retrieval metadata contract.
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
            "tool_calls": list(
                state.get("tool_calls", [])
            ),
            "metadata": metadata,
        }

    # =================================================================
    # 4. Tool-Based Retrieval
    # =================================================================

    try:

        # -------------------------------------------------------------
        # Build the registry at execution time.
        #
        # This is intentional. Existing Phase 16 tests monkeypatch
        # module-level retrieval functions. Resolving the handlers here
        # preserves that dependency-injection behavior.
        # -------------------------------------------------------------

        retrieval_tool_registry = (
            _build_agent_retrieval_registry()
        )

        # Existing Phase 16 retrieval metadata.
        tool_calls: list[dict[str, Any]] = []

        # New Phase 19 canonical state-level tool records.
        state_tool_calls = list(
            state.get("tool_calls", [])
        )

        # -------------------------------------------------------------
        # Tool 1 — Knowledge Base Search
        # -------------------------------------------------------------

        kb_result, kb_tool_record = execute_tool(
            retrieval_tool_registry,
            agent="retrieval_agent",
            tool_name="search_knowledge_base",
            arguments={
                "query": query,
                "strategy": RETRIEVAL_STRATEGY,
                "top_k": RETRIEVAL_CANDIDATE_K,
                "candidate_k": RETRIEVAL_CANDIDATE_K,
            },
            kwargs={
                "query": query,
                "strategy": RETRIEVAL_STRATEGY,
                "top_k": RETRIEVAL_CANDIDATE_K,
                "candidate_k": RETRIEVAL_CANDIDATE_K,
            },
        )

        # Record the call before checking the result so failures
        # are observable as well.
        state_tool_calls.append(
            kb_tool_record
        )

        if kb_result["status"] == "error":
            raise RuntimeError(
                "search_knowledge_base failed: "
                f"{kb_result.get('error', 'unknown error')}"
            )

        knowledge_base_documents = list(
            kb_result.get("data", [])
            or []
        )

        # Preserve legacy retrieval metadata.
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

        # Phase 19.2: execute and record the historical-ticket tool on
        # every retrieval pass. Historical evidence is only merged into
        # the final context when KB evidence is insufficient.
        previous_ticket_documents: list[dict[str, Any]] = []

        try:
            previous_ticket_result, previous_ticket_record = execute_tool(
                retrieval_tool_registry,
                agent="retrieval_agent",
                tool_name="search_previous_tickets",
                arguments={
                    "query": query,
                    "top_k": RETRIEVAL_TOP_K,
                },
                kwargs={
                    "query": query,
                    "top_k": RETRIEVAL_TOP_K,
                },
            )

            # Record the canonical Phase 19 tool telemetry before checking
            # the result so failed executions remain observable.
            state_tool_calls.append(previous_ticket_record)

            if previous_ticket_result["status"] == "error":
                raise RuntimeError(
                    "search_previous_tickets failed: "
                    f"{previous_ticket_result.get('error', 'unknown error')}"
                )

            previous_ticket_documents = list(
                previous_ticket_result.get("data", []) or []
            )

            tool_calls.append(
                {
                    "tool": "search_previous_tickets",
                    "status": "success",
                    "query": query,
                    "result_count": len(previous_ticket_documents),
                    "reason": (
                        "knowledge_base_evidence_insufficient"
                        if not kb_evidence_sufficient
                        else "tool_telemetry"
                    ),
                }
            )

            logger.info(
                f"ticket_id={state.get('ticket_id')} "
                f"stage=retrieval_agent "
                f"tool=search_previous_tickets "
                f"result_count={len(previous_ticket_documents)} "
                f"evidence_used={not kb_evidence_sufficient} "
                f"attempt={current_attempt}"
            )

        except Exception as exc:
            logger.warning(
                f"Previous-ticket search failed "
                f"for ticket_id={state.get('ticket_id')}: "
                f"{type(exc).__name__}: {exc}"
            )

            previous_ticket_documents = []

            # Preserve the legacy metadata contract if the failure
            # occurred before the legacy record was appended.
            if not any(
                call.get("tool") == "search_previous_tickets"
                for call in tool_calls
            ):
                tool_calls.append(
                    {
                        "tool": "search_previous_tickets",
                        "status": "failure",
                        "query": query,
                        "result_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        # Preserve Phase 16/18 evidence-selection semantics. The tool is
        # always executed for Phase 19.2 telemetry, but its documents only
        # participate in the final context when KB evidence is insufficient.
        if kb_evidence_sufficient:
            previous_ticket_documents = []

        # -------------------------------------------------------------
        # 6. Combine Evidence
        # -------------------------------------------------------------

        combined_documents: list[
            dict[str, Any]
        ] = []

        combined_documents.extend(
            knowledge_base_documents
        )

        combined_documents.extend(
            previous_ticket_documents
        )

        # -------------------------------------------------------------
        # Deduplicate by index_id when available
        # -------------------------------------------------------------

        deduplicated_documents: list[
            dict[str, Any]
        ] = []

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

        # Preserve original retrieval score
        # before reranking overwrites score.
        for document in combined_documents:
            document["retrieval_score"] = float(
                document.get(
                    "retrieval_score",
                    document.get(
                        "score",
                        0.0,
                    ),
                )
                or 0.0
            )

        candidate_count = len(
            combined_documents
        )

        # =================================================================
        # 7. Tool 3 — Reranking
        # =================================================================

        reranked_documents: list[
            dict[str, Any]
        ] = []

        if combined_documents:

            try:

                rerank_result, (
                    rerank_tool_record
                ) = execute_tool(
                    retrieval_tool_registry,
                    agent="retrieval_agent",
                    tool_name="rerank_documents",
                    arguments={
                        "query": query,
                        "input_count": len(
                            combined_documents
                        ),
                        "top_k": RETRIEVAL_TOP_K,
                    },
                    kwargs={
                        "query": query,
                        "documents": combined_documents,
                        "top_k": RETRIEVAL_TOP_K,
                    },
                )

                state_tool_calls.append(
                    rerank_tool_record
                )

                if (
                    rerank_result["status"]
                    == "error"
                ):
                    raise RuntimeError(
                        "rerank_documents failed: "
                        f"{rerank_result.get('error', 'unknown error')}"
                    )

                reranked_documents = list(
                    rerank_result.get(
                        "data",
                        [],
                    )
                    or []
                )

                tool_calls.append(
                    {
                        "tool": "rerank_documents",
                        "status": "success",
                        "query": query,
                        "input_count": len(
                            combined_documents
                        ),
                        "result_count": len(
                            reranked_documents
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
                        "input_count": len(
                            combined_documents
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
            context_docs: list[
                dict[str, Any]
            ] = []
        else:
            context_docs = (
                reranked_documents[
                    :RETRIEVAL_TOP_K
                ]
            )

        # Normalize retrieval/reranker scores.
        for document in context_docs:

            if "retrieval_score" not in document:
                document["retrieval_score"] = float(
                    document.get(
                        "score",
                        0.0,
                    )
                    or 0.0
                )

            document["reranker_score"] = float(
                document.get(
                    "score",
                    0.0,
                )
                or 0.0
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
                retrieval_top_score
                - retrieval_second_score,
            )
            if len(context_docs) > 1
            else retrieval_top_score
        )

        reranker_score_gap = (
            max(
                0.0,
                reranker_top_score
                - reranker_second_score,
            )
            if len(context_docs) > 1
            else reranker_top_score
        )

        # Backward-compatible aliases.
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

                # Legacy Phase 16/18 observability.
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

                "document_count": len(
                    context_docs
                ),

                "candidate_count": (
                    candidate_count
                ),

                "top_score": top_score,
                "score_gap": score_gap,

                "retrieval_top_score": (
                    retrieval_top_score
                ),

                "retrieval_score_gap": (
                    retrieval_score_gap
                ),

                "reranker_top_score": (
                    reranker_top_score
                ),

                "reranker_score_gap": (
                    reranker_score_gap
                ),

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

            # Phase 19 canonical tool-call state.
            "tool_calls": state_tool_calls,

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

        # Preserve the explicit configuration error contract expected
        # by retrieval tests and callers.
        if (
            isinstance(exc, ValueError)
            and
            "Unsupported knowledge-base retrieval strategy"
            in str(exc)
        ):
            error_message = (
                "ValueError: Unsupported "
                "RETRIEVAL_STRATEGY: "
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

                # Preserve legacy metadata.
                "tool_calls": tool_calls
                if "tool_calls" in locals()
                else [],

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

            # Preserve any tool records generated before failure.
            "tool_calls": (
                state_tool_calls
                if "state_tool_calls" in locals()
                else list(
                    state.get(
                        "tool_calls",
                        [],
                    )
                )
            ),

            "metadata": metadata,
        }


# =====================================================================
# Retrieval Decision Agent
# =====================================================================


def retrieval_decision_agent(
    state: ResolveXState,
) -> dict[str, Any]:
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

    result: dict[str, Any] = {
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


__all__ = [
    "MAX_RETRIEVAL_ATTEMPTS",
    "MIN_RETRIEVAL_DOCUMENTS",
    "MIN_RETRIEVAL_TOP_SCORE",
    "RETRIEVAL_RETRY_STRATEGIES",
    "hybrid_retriever",
    "doc_store",
    "retrieval_agent",
    "retrieval_decision_agent",
    "search_knowledge_base",
    "search_previous_tickets",
    "rerank_documents",
    "RETRIEVAL_STRATEGY",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_CANDIDATE_K",
    "BM25_WEIGHT",
    "DENSE_WEIGHT",
]
