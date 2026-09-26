"""
heuristic.py — Baseline Heuristic Policy for ResolveX.

Refactor of the existing Phase 18/19 deterministic decision logic into a reusable policy object.
Reproduces expected deterministic decisions without safety regression.
"""

from typing import Dict, Any, Tuple, Union
from ai.policy.action_space import DecisionAction
from ai.policy.feature_extractor import (
    DecisionPolicyFeatures,
    DecisionPolicyFeatureExtractor,
)


class HeuristicDecisionPolicy:
    """
    Deterministic reference policy implementing ResolveX Phase 18/19 decision rules.
    """

    AUTO_RESOLVE_THRESHOLD: float = 0.75

    def __init__(self, auto_resolve_threshold: float = 0.75):
        self.auto_resolve_threshold = auto_resolve_threshold

    def predict(
        self, state_or_features: Union[Dict[str, Any], DecisionPolicyFeatures, Any]
    ) -> DecisionAction:
        action, _ = self.predict_with_metadata(state_or_features)
        return action

    def predict_with_metadata(
        self, state_or_features: Union[Dict[str, Any], DecisionPolicyFeatures, Any]
    ) -> Tuple[DecisionAction, Dict[str, Any]]:
        """
        Evaluate deterministic rules and return (action, metadata).
        """
        if isinstance(state_or_features, DecisionPolicyFeatures):
            features = state_or_features
            s_dict = features.to_dict()
        elif isinstance(state_or_features, dict):
            s_dict = state_or_features
            features = DecisionPolicyFeatureExtractor.extract_features(s_dict)
        elif hasattr(state_or_features, "dict") and callable(state_or_features.dict):
            s_dict = state_or_features.dict()
            features = DecisionPolicyFeatureExtractor.extract_features(s_dict)
        elif hasattr(state_or_features, "model_dump"):
            s_dict = state_or_features.model_dump()
            features = DecisionPolicyFeatureExtractor.extract_features(s_dict)
        else:
            s_dict = getattr(state_or_features, "__dict__", {})
            features = DecisionPolicyFeatureExtractor.extract_features(s_dict)

        # Helper to extract list fields safely
        def _get_list(raw_val: Any) -> list:
            if isinstance(raw_val, (list, tuple)):
                return list(raw_val)
            if isinstance(raw_val, str) and raw_val.strip():
                return [raw_val]
            return []

        errors = _get_list(s_dict.get("errors")) or _get_list(s_dict.get("error"))
        if features.errors_present > 0 or errors:
            err_msg = (
                "; ".join(str(e) for e in errors)
                if errors
                else "Processing errors present"
            )
            return DecisionAction.ESCALATE, {
                "decision": DecisionAction.ESCALATE.value,
                "requires_human": True,
                "reason": f"Automated processing encountered errors: {err_msg}",
                "gate_failed": "errors",
            }

        # 2. Explicit human requirement
        if (
            features.requires_human > 0
            or bool(s_dict.get("requires_human"))
            or bool(s_dict.get("explicit_requires_human"))
        ):
            return DecisionAction.HUMAN_REVIEW, {
                "decision": DecisionAction.HUMAN_REVIEW.value,
                "requires_human": True,
                "reason": "The generated resolution explicitly requires human intervention.",
                "gate_failed": "requires_human",
            }

        # 3. Missing information
        missing_info = _get_list(
            s_dict.get("diagnosis_missing_information")
        ) or _get_list(s_dict.get("missing_information"))
        if features.missing_information > 0 or missing_info:
            info_msg = (
                "; ".join(str(m) for m in missing_info)
                if missing_info
                else "Missing required diagnostic info"
            )
            return DecisionAction.ASK_CLARIFICATION, {
                "decision": DecisionAction.ASK_CLARIFICATION.value,
                "requires_human": False,
                "reason": f"Additional information is required: {info_msg}",
                "gate_failed": "missing_information",
            }

        # 4. Verification failure
        vr = s_dict.get("verification_result") or {}
        if isinstance(vr, dict) and "verification_passed" in vr:
            ver_passed = bool(vr["verification_passed"])
        elif "verification_passed" in s_dict:
            ver_passed = bool(s_dict["verification_passed"])
        else:
            ver_passed = not bool(s_dict.get("verification_failed", False))

        if features.verification_failed > 0 or not ver_passed:
            return DecisionAction.HUMAN_REVIEW, {
                "decision": DecisionAction.HUMAN_REVIEW.value,
                "requires_human": True,
                "reason": "The proposed resolution did not pass the verification gate.",
                "gate_failed": "verification_failed",
            }

        # 5. Verification confidence
        if features.verification_score < self.auto_resolve_threshold:
            return DecisionAction.HUMAN_REVIEW, {
                "decision": DecisionAction.HUMAN_REVIEW.value,
                "requires_human": True,
                "reason": f"Verification confidence ({features.verification_score:.2f}) is below auto-resolution threshold ({self.auto_resolve_threshold:.2f}).",
                "gate_failed": "low_verification_confidence",
            }

        # 6. Diagnosis confidence
        if features.diagnosis_confidence < self.auto_resolve_threshold:
            return DecisionAction.HUMAN_REVIEW, {
                "decision": DecisionAction.HUMAN_REVIEW.value,
                "requires_human": True,
                "reason": f"Diagnosis confidence ({features.diagnosis_confidence:.2f}) is below auto-resolution threshold ({self.auto_resolve_threshold:.2f}).",
                "gate_failed": "low_diagnosis_confidence",
            }

        # 7. Resolution confidence
        if features.resolution_confidence < self.auto_resolve_threshold:
            return DecisionAction.HUMAN_REVIEW, {
                "decision": DecisionAction.HUMAN_REVIEW.value,
                "requires_human": True,
                "reason": f"Resolution confidence ({features.resolution_confidence:.2f}) is below auto-resolution threshold ({self.auto_resolve_threshold:.2f}).",
                "gate_failed": "low_resolution_confidence",
            }

        # 8. Fallback protection
        if features.fallback_used > 0 or bool(s_dict.get("fallback_used")):
            return DecisionAction.HUMAN_REVIEW, {
                "decision": DecisionAction.HUMAN_REVIEW.value,
                "requires_human": True,
                "reason": "One or more AI pipeline components used a fallback path.",
                "gate_failed": "fallback_used",
            }

        # 9. Warnings
        warnings = _get_list(s_dict.get("warnings"))
        if warnings:
            warn_msg = "; ".join(str(w) for w in warnings)
            return DecisionAction.HUMAN_REVIEW, {
                "decision": DecisionAction.HUMAN_REVIEW.value,
                "requires_human": True,
                "reason": f"Pipeline produced warnings: {warn_msg}",
                "gate_failed": "warnings",
            }

        # 10. All safety gates passed -> Auto resolve
        return DecisionAction.AUTO_RESOLVE, {
            "decision": DecisionAction.AUTO_RESOLVE.value,
            "requires_human": False,
            "reason": "All safety gates passed.",
            "gate_failed": None,
        }
