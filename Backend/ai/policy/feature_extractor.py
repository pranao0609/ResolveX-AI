"""
feature_extractor.py — Canonical Feature Extraction Layer for ResolveX Decision Policy.

Converts ResolveXState or state dict into a standardized DecisionPolicyFeatures object
and a deterministic numeric vector for contextual bandit models.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Union
import numpy as np

PRIORITY_MAP: Dict[str, float] = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "urgent": 1.0,
    "critical": 1.0,
}

COMPLEXITY_MAP: Dict[str, float] = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "urgent": 1.0,
    "complex": 0.75,
}


@dataclass
class DecisionPolicyFeatures:
    """
    Structured feature representation for decision policy models.
    Version: v1
    """

    # Context features
    classification_confidence: float = 0.0
    retrieval_score: float = 0.0
    retrieval_gap: float = 0.0
    number_of_documents: float = 0.0
    diagnosis_confidence: float = 0.0
    resolution_confidence: float = 0.0
    verification_score: float = 0.0
    ticket_priority: float = 0.5
    ticket_complexity: float = 0.5
    previous_attempts: float = 0.0
    historical_resolution_rate: float = 0.0

    # Memory features
    memory_historical_ticket_count: float = 0.0
    memory_historical_used: float = 0.0
    memory_conversation_count: float = 0.0
    memory_has_historical_solution: float = 0.0

    # Safety/State Binary Indicator features
    requires_human: float = 0.0
    verification_failed: float = 0.0
    missing_information: float = 0.0
    fallback_used: float = 0.0
    errors_present: float = 0.0
    safety_violation: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class DecisionPolicyFeatureExtractor:
    """
    Canonical feature extractor. Guaranteed consistent ordering and feature schema.
    """

    FEATURE_VERSION: str = "v1"

    FEATURE_NAMES: List[str] = [
        "classification_confidence",
        "retrieval_score",
        "retrieval_gap",
        "number_of_documents",
        "diagnosis_confidence",
        "resolution_confidence",
        "verification_score",
        "ticket_priority",
        "ticket_complexity",
        "previous_attempts",
        "historical_resolution_rate",
        "memory_historical_ticket_count",
        "memory_historical_used",
        "memory_conversation_count",
        "memory_has_historical_solution",
        "requires_human",
        "verification_failed",
        "missing_information",
        "fallback_used",
        "errors_present",
        "safety_violation",
    ]

    @classmethod
    def feature_dimension(cls) -> int:
        return len(cls.FEATURE_NAMES)

    @classmethod
    def extract_features(
        cls, state: Union[Dict[str, Any], Any]
    ) -> DecisionPolicyFeatures:
        """
        Extract DecisionPolicyFeatures from state dictionary or state object safely.
        """
        if hasattr(state, "model_dump"):
            s_dict = state.model_dump()
        elif hasattr(state, "dict") and callable(state.dict):
            s_dict = state.dict()
        elif isinstance(state, dict):
            s_dict = state
        else:
            s_dict = getattr(state, "__dict__", {})

        pf = s_dict.get("policy_features") or {}

        def get_val(key: str, default: float = 0.0) -> float:
            if key in pf and pf[key] is not None:
                try:
                    return float(pf[key])
                except (ValueError, TypeError):
                    pass
            if key in s_dict and s_dict[key] is not None:
                try:
                    return float(s_dict[key])
                except (ValueError, TypeError):
                    pass
            return float(default)

        class_conf = get_val("classification_confidence", get_val("confidence", 0.0))
        ret_score = get_val("retrieval_score", 0.0)
        ret_gap = get_val("retrieval_gap", 0.0)
        num_docs = get_val("number_of_documents", 0.0)
        diag_conf = get_val("diagnosis_confidence", get_val("confidence", 0.0))
        res_conf = get_val("resolution_confidence", get_val("confidence", 0.0))

        vr = s_dict.get("verification_result") or {}
        if isinstance(vr, dict) and vr.get("verification_score") is not None:
            ver_score = float(vr["verification_score"])
        else:
            ver_score = get_val(
                "verification_score", get_val("verification_confidence", 0.0)
            )

        if isinstance(vr, dict) and "verification_passed" in vr:
            ver_passed = bool(vr["verification_passed"])
        elif "verification_passed" in s_dict:
            ver_passed = bool(s_dict["verification_passed"])
        else:
            ver_passed = not bool(s_dict.get("verification_failed", False))

        p_str = str(s_dict.get("priority", pf.get("ticket_priority", "medium"))).lower()
        p_val = PRIORITY_MAP.get(p_str, 0.5)

        c_str = str(
            s_dict.get("complexity", pf.get("ticket_complexity", "medium"))
        ).lower()
        c_val = COMPLEXITY_MAP.get(c_str, 0.5)

        prev_attempts = get_val("previous_attempts", 0.0)
        hist_res_rate = get_val("historical_resolution_rate", 0.5)

        prev_tickets = s_dict.get("previous_tickets") or []
        if not isinstance(prev_tickets, list):
            prev_tickets = []
        conv_hist = s_dict.get("conversation_history") or []
        conv_count = len(conv_hist) if isinstance(conv_hist, (list, tuple)) else 0
        has_sol = any(
            bool(t.get("solution")) for t in prev_tickets if isinstance(t, dict)
        )

        mem_hist_count = (
            float(len(prev_tickets))
            if prev_tickets
            else get_val("memory_historical_ticket_count", 0.0)
        )
        mem_hist_used = (
            1.0 if (prev_tickets or get_val("memory_historical_used", 0.0) > 0) else 0.0
        )
        mem_conv_count = (
            float(conv_count)
            if conv_count
            else get_val("memory_conversation_count", 0.0)
        )
        mem_has_sol = (
            1.0
            if (has_sol or get_val("memory_has_historical_solution", 0.0) > 0)
            else 0.0
        )

        req_human = (
            1.0
            if bool(s_dict.get("requires_human"))
            or bool(s_dict.get("explicit_requires_human"))
            or bool(pf.get("requires_human"))
            else 0.0
        )
        v_failed = (
            1.0 if (not ver_passed) or bool(s_dict.get("verification_failed")) else 0.0
        )
        miss_info = (
            1.0
            if bool(s_dict.get("missing_information"))
            or bool(s_dict.get("diagnosis_missing_information"))
            or bool(pf.get("missing_information"))
            else 0.0
        )
        fb_used = (
            1.0
            if bool(s_dict.get("fallback_used")) or bool(pf.get("fallback_used"))
            else 0.0
        )
        err_present = (
            1.0
            if (
                bool(s_dict.get("errors"))
                or bool(s_dict.get("error"))
                or bool(s_dict.get("errors_present"))
            )
            else 0.0
        )
        saf_viol = (
            1.0
            if bool(s_dict.get("safety_violation"))
            or bool(s_dict.get("policy_violation"))
            else 0.0
        )

        return DecisionPolicyFeatures(
            classification_confidence=class_conf,
            retrieval_score=ret_score,
            retrieval_gap=ret_gap,
            number_of_documents=num_docs,
            diagnosis_confidence=diag_conf,
            resolution_confidence=res_conf,
            verification_score=ver_score,
            ticket_priority=p_val,
            ticket_complexity=c_val,
            previous_attempts=prev_attempts,
            historical_resolution_rate=hist_res_rate,
            memory_historical_ticket_count=mem_hist_count,
            memory_historical_used=mem_hist_used,
            memory_conversation_count=mem_conv_count,
            memory_has_historical_solution=mem_has_sol,
            requires_human=req_human,
            verification_failed=v_failed,
            missing_information=miss_info,
            fallback_used=fb_used,
            errors_present=err_present,
            safety_violation=saf_viol,
        )

    @classmethod
    def to_vector(cls, features: DecisionPolicyFeatures) -> np.ndarray:
        f_dict = features.to_dict()
        vec = [float(f_dict[name]) for name in cls.FEATURE_NAMES]
        return np.array(vec, dtype=np.float64)

    @classmethod
    def state_to_vector(cls, state: Union[Dict[str, Any], Any]) -> np.ndarray:
        features = cls.extract_features(state)
        return cls.to_vector(features)
