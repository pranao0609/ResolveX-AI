"""
safety_guard.py — Authoritative Deterministic Safety Policy Guard for ResolveX.

CRITICAL DESIGN PRINCIPLE:
The contextual bandit recommends an action, but safety constraints MUST be able to override it.
The bandit MUST NOT be capable of bypassing deterministic safety constraints.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
from ai.policy.action_space import DecisionAction, ActionSpace
from ai.policy.feature_extractor import (
    DecisionPolicyFeatureExtractor,
    DecisionPolicyFeatures,
)


@dataclass
class SafetyGuardResult:
    recommended_action: str
    final_action: str
    was_overridden: bool
    override_reason: str
    safety_gate_passed: bool
    gate_failed: Optional[str] = None


class SafetyPolicyGuard:
    """
    Authoritative safety layer enforcing deterministic system guardrails.
    """

    AUTO_RESOLVE_THRESHOLD: float = 0.75

    def __init__(self, auto_resolve_threshold: float = 0.75):
        self.auto_resolve_threshold = auto_resolve_threshold

    def evaluate_safety(
        self,
        recommended_action: Union[str, DecisionAction],
        state: Dict[str, Any],
    ) -> SafetyGuardResult:
        """
        Evaluate recommended action against deterministic safety rules.
        """
        rec_str = (
            recommended_action.value
            if isinstance(recommended_action, DecisionAction)
            else str(recommended_action).lower()
        )

        if rec_str not in ActionSpace.all_action_names():
            return SafetyGuardResult(
                recommended_action=rec_str,
                final_action=DecisionAction.HUMAN_REVIEW.value,
                was_overridden=True,
                override_reason=f"Unknown or invalid recommendation '{rec_str}'. Failing closed to human_review.",
                safety_gate_passed=False,
                gate_failed="invalid_recommendation",
            )

        features = DecisionPolicyFeatureExtractor.extract_features(state)

        def _get_list(raw_val: Any) -> list:
            if isinstance(raw_val, (list, tuple)):
                return list(raw_val)
            if isinstance(raw_val, str) and raw_val.strip():
                return [raw_val]
            return []

        errors = _get_list(state.get("errors")) or _get_list(state.get("error"))
        missing_info = _get_list(
            state.get("diagnosis_missing_information")
        ) or _get_list(state.get("missing_information"))
        warnings = _get_list(state.get("warnings"))

        vr = state.get("verification_result") or {}
        if isinstance(vr, dict) and "verification_passed" in vr:
            ver_passed = bool(vr["verification_passed"])
        elif "verification_passed" in state:
            ver_passed = bool(state["verification_passed"])
        else:
            ver_passed = not bool(state.get("verification_failed", False))

        # ---------------------------------------------------------------
        # Rule 1: Critical execution errors -> ESCALATE
        # ---------------------------------------------------------------
        if errors or features.errors_present > 0:
            if rec_str != DecisionAction.ESCALATE.value:
                err_msg = (
                    "; ".join(str(e) for e in errors)
                    if errors
                    else "Processing errors present"
                )
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.ESCALATE.value,
                    was_overridden=True,
                    override_reason=f"Processing errors present: {err_msg}",
                    safety_gate_passed=False,
                    gate_failed="errors",
                )

        # ---------------------------------------------------------------
        # Rule 2: Explicit human requirement -> HUMAN_REVIEW
        # ---------------------------------------------------------------
        if (
            features.requires_human > 0
            or bool(state.get("requires_human"))
            or bool(state.get("explicit_requires_human"))
        ):
            if rec_str != DecisionAction.HUMAN_REVIEW.value:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason="State explicitly requires human intervention.",
                    safety_gate_passed=False,
                    gate_failed="requires_human",
                )

        # ---------------------------------------------------------------
        # Rule 3: Missing information -> ASK_CLARIFICATION
        # ---------------------------------------------------------------
        if missing_info or features.missing_information > 0:
            if rec_str != DecisionAction.ASK_CLARIFICATION.value:
                info_msg = (
                    "; ".join(str(m) for m in missing_info)
                    if missing_info
                    else "Missing information"
                )
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.ASK_CLARIFICATION.value,
                    was_overridden=True,
                    override_reason=f"Missing diagnostic information: {info_msg}",
                    safety_gate_passed=False,
                    gate_failed="missing_information",
                )

        # ---------------------------------------------------------------
        # Rule 4: Verification failure -> HUMAN_REVIEW
        # ---------------------------------------------------------------
        if not ver_passed or features.verification_failed > 0:
            if rec_str == DecisionAction.AUTO_RESOLVE.value:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason="Resolution failed verification gate.",
                    safety_gate_passed=False,
                    gate_failed="verification_failed",
                )

        # ---------------------------------------------------------------
        # Rule 5: Low confidence for AUTO_RESOLVE -> HUMAN_REVIEW
        # ---------------------------------------------------------------
        if rec_str == DecisionAction.AUTO_RESOLVE.value:
            if features.verification_score < self.auto_resolve_threshold:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason=f"Verification confidence ({features.verification_score:.2f}) below threshold ({self.auto_resolve_threshold:.2f}).",
                    safety_gate_passed=False,
                    gate_failed="low_verification_confidence",
                )
            if features.diagnosis_confidence < self.auto_resolve_threshold:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason=f"Diagnosis confidence ({features.diagnosis_confidence:.2f}) below threshold ({self.auto_resolve_threshold:.2f}).",
                    safety_gate_passed=False,
                    gate_failed="low_diagnosis_confidence",
                )
            if features.resolution_confidence < self.auto_resolve_threshold:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason=f"Resolution confidence ({features.resolution_confidence:.2f}) below threshold ({self.auto_resolve_threshold:.2f}).",
                    safety_gate_passed=False,
                    gate_failed="low_resolution_confidence",
                )

        # ---------------------------------------------------------------
        # Rule 6: Fallback used for AUTO_RESOLVE -> HUMAN_REVIEW
        # ---------------------------------------------------------------
        if features.fallback_used > 0 or bool(state.get("fallback_used")):
            if rec_str == DecisionAction.AUTO_RESOLVE.value:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason="Fallback path was used by AI components.",
                    safety_gate_passed=False,
                    gate_failed="fallback_used",
                )

        # ---------------------------------------------------------------
        # Rule 7: Warnings present for AUTO_RESOLVE -> HUMAN_REVIEW
        # ---------------------------------------------------------------
        if warnings and rec_str == DecisionAction.AUTO_RESOLVE.value:
            warn_msg = "; ".join(str(w) for w in warnings)
            return SafetyGuardResult(
                recommended_action=rec_str,
                final_action=DecisionAction.HUMAN_REVIEW.value,
                was_overridden=True,
                override_reason=f"Pipeline warnings present: {warn_msg}",
                safety_gate_passed=False,
                gate_failed="warnings",
            )

        # ---------------------------------------------------------------
        # Rule 8: Policy / Safety violation -> ESCALATE or HUMAN_REVIEW
        # ---------------------------------------------------------------
        if features.safety_violation > 0 or bool(state.get("safety_violation")):
            if rec_str == DecisionAction.AUTO_RESOLVE.value:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason="Safety policy violation detected.",
                    safety_gate_passed=False,
                    gate_failed="safety_violation",
                )

        # ---------------------------------------------------------------
        # Rule 9: Unsafe Action / Destructive Operation Guard -> HUMAN_REVIEW
        # ---------------------------------------------------------------
        combined_text = (
            str(state.get("ticket_text", ""))
            + " "
            + str(state.get("diagnosis", ""))
            + " "
            + " ".join(str(s) for s in state.get("resolution_steps", []))
        ).lower()

        unsafe_patterns = [
            "drop database",
            "drop table",
            "rm -rf",
            "format c:",
            "delete production",
            "sudo chmod 777",
            "grant root",
            "disable firewall",
            "disable security",
            "bypass authentication",
            "override privilege",
            "export secrets",
            "print api key",
        ]

        if any(pat in combined_text for pat in unsafe_patterns):
            if rec_str == DecisionAction.AUTO_RESOLVE.value:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason="Unsafe action, privilege escalation, or destructive operation detected.",
                    safety_gate_passed=False,
                    gate_failed="unsafe_action_guard",
                )

        # ---------------------------------------------------------------
        # Rule 10: Secret Leakage Guard -> HUMAN_REVIEW
        # ---------------------------------------------------------------
        secret_patterns = ["gsk_", "sk-", "lsv2_", "bearer "]
        if any(pat in combined_text for pat in secret_patterns):
            if rec_str == DecisionAction.AUTO_RESOLVE.value:
                return SafetyGuardResult(
                    recommended_action=rec_str,
                    final_action=DecisionAction.HUMAN_REVIEW.value,
                    was_overridden=True,
                    override_reason="Potential secret key or credential leakage detected in output.",
                    safety_gate_passed=False,
                    gate_failed="secret_leakage_guard",
                )

        return SafetyGuardResult(
            recommended_action=rec_str,
            final_action=rec_str,
            was_overridden=False,
            override_reason="",
            safety_gate_passed=True,
            gate_failed=None,
        )
