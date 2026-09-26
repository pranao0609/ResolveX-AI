from __future__ import annotations

from typing import Any, TypedDict


class ResolveXState(TypedDict, total=False):
    """
    Shared state carried through the ResolveX LangGraph workflow.

    The state is intentionally domain-oriented rather than tied to
    a specific implementation of any individual agent.
    """

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    ticket_id: int | str | None
    ticket_text: str
    attachment_paths: list[str] | str | None

    # ------------------------------------------------------------------
    # Ticket analysis
    # ------------------------------------------------------------------

    cleaned_ticket: str
    category: str
    category_confidence: float
    classification_confidence_level: str
    classification_requires_review: bool
    classification_requires_reclassification: bool
    reclassified: bool
    reclassification_used: bool
    reclassification_error: str | None
    classification_metadata: dict[str, Any]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    retrieved_context: str
    retrieved_documents: list[dict[str, Any]]
    retrieval_metadata: dict[str, Any]

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    diagnosis: str
    root_cause: str
    diagnosis_confidence: float
    diagnosis_result: dict[str, Any]

    # Diagnosis contract fields
    diagnosis_problem: str
    diagnosis_root_cause: str
    diagnosis_evidence: list[str]
    diagnosis_missing_information: list[str]

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    resolution_steps: list[str]
    evidence: list[str]
    resolution_confidence: float

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    verification_passed: bool
    verification_reason: str
    verification_confidence: float
    verification_supported_by_evidence: bool
    verification_hallucination_detected: bool
    verification_complete: bool
    verification_policy_compliant: bool
    verification_resolution_correct: bool
    verification_result: dict[str, Any]

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    decision: str
    requires_human: bool
    escalation_reason: str

    # ------------------------------------------------------------------
    # Runtime / reliability
    # ------------------------------------------------------------------

    fallback_used: bool
    errors: list[str]
    warnings: list[str]

    # ------------------------------------------------------------------
    # Agent Memory
    # ------------------------------------------------------------------

    # Short-term conversational context for the current ticket.
    conversation_history: list[dict[str, Any]]

    # Historical tickets/resolutions retrieved from the ticket system.
    previous_tickets: list[dict[str, Any]]

    # ------------------------------------------------------------------
    # Tooling
    # ------------------------------------------------------------------

    # Structured record of tools used during graph execution.
    tool_calls: list[dict[str, Any]]

    # ------------------------------------------------------------------
    # Observability / persistence hooks
    # ------------------------------------------------------------------

    request_id: str | None
    graph_run_id: str | None
    metadata: dict[str, Any]

    # ------------------------------------------------------------------
    # Model / Prompt / Latency Tracking
    # ------------------------------------------------------------------

    model_versions: dict[str, str]
    prompt_versions: dict[str, str]
    latency: dict[str, float]

    # ------------------------------------------------------------------
    # Decision Policy / RL
    # ------------------------------------------------------------------

    # Deterministic feature snapshot used by the future policy layer.
    policy_features: dict[str, float]

    # Proposed policy action.
    # Not used for production routing yet.
    policy_action: str

    # Additional policy metadata.
    policy_metadata: dict[str, Any]
