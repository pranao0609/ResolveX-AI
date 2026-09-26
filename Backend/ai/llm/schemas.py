from pydantic import BaseModel, Field


class DiagnosisResult(BaseModel):
    """
    Structured diagnosis produced by the Diagnosis Agent.

    The diagnosis agent determines what is happening and what the
    most likely root cause is. It does not provide user-facing
    resolution instructions.
    """

    problem: str = Field(
        ...,
        min_length=1,
        description="Concise description of the actual problem reported by the user.",
    )

    possible_root_cause: str = Field(
        ...,
        min_length=1,
        description="Most likely root cause inferred from the ticket and retrieved evidence.",
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Retrieved knowledge-base evidence supporting the diagnosis.",
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="Information that is still missing or would improve diagnostic certainty.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the diagnosis and proposed root cause.",
    )


class ResolutionResult(BaseModel):
    diagnosis: str = Field(
        ...,
        min_length=1,
        description="Concise diagnosis of the reported IT issue.",
    )

    root_cause: str = Field(
        ...,
        min_length=1,
        description="Most likely root cause of the issue.",
    )

    resolution_steps: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered steps the user should follow to resolve the issue.",
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Knowledge-base evidence supporting the diagnosis and resolution.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence in the proposed resolution.",
    )

    requires_human: bool = Field(
        ...,
        description="Whether the ticket requires human support or escalation.",
    )
