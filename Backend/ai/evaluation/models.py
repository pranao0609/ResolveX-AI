"""
models.py — Canonical Evaluation Data Model for ResolveX (Phase 20.1).

Defines structured EvaluationRecord and MetricValue representations.
Ensures unavailable ground truth values are represented explicitly as None / available=False
without fabricating real-world performance numbers.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import json


@dataclass
class MetricValue:
    """
    Standardized wrapper for evaluation metrics with explicit availability tracking.
    Prevents missing ground truth from being fabricated into misleading numbers.
    """

    value: Optional[Union[float, int, str, bool]]
    available: bool
    source: str  # e.g., "hitl", "proxy", "telemetry", "missing_ground_truth"
    sample_count: int
    numerator: Optional[int] = None
    denominator: Optional[int] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "available": self.available,
            "source": self.source,
            "sample_count": self.sample_count,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "description": self.description,
        }

    @classmethod
    def unavailable(
        cls, name_or_desc: str = "", source: str = "missing_ground_truth"
    ) -> "MetricValue":
        return cls(
            value=None,
            available=False,
            source=source,
            sample_count=0,
            description=name_or_desc,
        )


@dataclass
class EvaluationRecord:
    """
    Canonical record capturing one evaluated ticket execution.
    Supports multiple data sources: synthetic, historical, live_execution, hitl.
    """

    evaluation_id: str
    ticket_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    evaluation_source: str = (
        "live_execution"  # synthetic | historical | live_execution | hitl
    )

    # Agent execution metrics
    task_success: Optional[bool] = None
    agent_failure: Optional[bool] = None
    failure_stage: Optional[str] = None
    agent_latency_ms: Optional[float] = None

    # Tool execution metrics
    tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    accurate_tool_calls: Optional[int] = None
    unnecessary_tool_calls: Optional[int] = None

    # Resolution metrics
    resolution_success: Optional[bool] = None
    human_accepted: Optional[bool] = None
    escalation_correct: Optional[bool] = None
    auto_resolved: bool = False
    unsafe_auto_resolution: Optional[bool] = None

    # Policy identification
    policy_mode: str = "heuristic"
    policy_name: str = "heuristic_baseline"
    policy_version: str = "v1"
    heuristic_action: str = "human_review"
    rl_action: Optional[str] = None
    final_action: str = "human_review"

    # RL & Reward metrics
    reward: Optional[float] = None
    reward_source: Optional[str] = None  # observed | proxy
    policy_success: Optional[bool] = None
    auto_resolution_success: Optional[bool] = None
    human_escalation: bool = False
    decision_latency_ms: float = 0.0
    cost: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None

    # HITL Feedback fields
    human_reviewed: bool = False
    human_action: Optional[str] = None
    human_correction: Optional[bool] = None
    human_feedback: Optional[str] = None
    human_feedback_reason: Optional[str] = None

    # Safety Guard telemetry
    was_overridden: bool = False
    override_reason: str = ""
    safety_gate_passed: bool = True
    gate_failed: Optional[str] = None

    # Metadata & Versioning
    feature_version: str = "v1"
    model_versions: Dict[str, str] = field(default_factory=dict)
    prompt_versions: Dict[str, str] = field(default_factory=dict)
    context_features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRecord":
        return cls(**data)
