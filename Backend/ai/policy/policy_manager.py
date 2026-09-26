"""
policy_manager.py — Central Policy Manager & Integration Layer for ResolveX.

Connects:
- DecisionPolicyFeatureExtractor
- Baseline Heuristic Policy
- LinUCB / Thompson Sampling Contextual Bandit Policies
- SafetyPolicyGuard
- Policy Modes (HEURISTIC, SHADOW, BANDIT)
- Telemetry & Metadata logging
"""

import time
from typing import Dict, Any, Optional, Union, Tuple
from enum import Enum

from ai.policy.action_space import ActionSpace, DecisionAction
from ai.policy.feature_extractor import (
    DecisionPolicyFeatureExtractor,
    DecisionPolicyFeatures,
)
from ai.policy.heuristic import HeuristicDecisionPolicy
from ai.policy.linucb import LinUCBPolicy
from ai.policy.thompson import ThompsonSamplingPolicy
from ai.policy.safety_guard import SafetyPolicyGuard, SafetyGuardResult
from app.config import settings
from app.core.logger import logger


class PolicyMode(str, Enum):
    HEURISTIC = "heuristic"
    SHADOW = "shadow"
    BANDIT = "bandit"


class PolicyManager:
    """
    Production-safe policy orchestrator.
    Guarantees deterministic safety guard override over contextual bandit recommendations.
    """

    def __init__(
        self,
        mode: Union[str, PolicyMode] = PolicyMode.HEURISTIC,
        bandit_type: str = "linucb",
        auto_resolve_threshold: float = 0.75,
        linucb_alpha: float = 0.5,
        random_seed: int = 42,
    ):
        self.mode = PolicyMode(mode.lower()) if isinstance(mode, str) else mode
        self.bandit_type = bandit_type.lower()
        self.auto_resolve_threshold = auto_resolve_threshold
        self.random_seed = random_seed

        self.feature_extractor = DecisionPolicyFeatureExtractor()
        self.baseline_policy = HeuristicDecisionPolicy(
            auto_resolve_threshold=auto_resolve_threshold
        )
        self.safety_guard = SafetyPolicyGuard(
            auto_resolve_threshold=auto_resolve_threshold
        )

        if self.bandit_type == "thompson_sampling":
            self.bandit_policy = ThompsonSamplingPolicy(random_seed=random_seed)
        else:
            self.bandit_policy = LinUCBPolicy(
                alpha=linucb_alpha, random_seed=random_seed
            )

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate policy given state dict and return decision response dictionary.
        """
        start_time = time.perf_counter()

        features = self.feature_extractor.extract_features(state)
        feature_vec = self.feature_extractor.to_vector(features)

        baseline_action, baseline_meta = self.baseline_policy.predict_with_metadata(
            state
        )

        bandit_scores = {}
        if self.mode in (PolicyMode.SHADOW, PolicyMode.BANDIT):
            try:
                bandit_rec, bandit_telemetry = self.bandit_policy.predict(feature_vec)
                bandit_action = bandit_rec.value
                bandit_scores = bandit_telemetry.get("scores", {})
            except Exception as e:
                logger.error(
                    f"Contextual bandit prediction error: {e}. Falling back to baseline."
                )
                bandit_action = baseline_action.value
                bandit_scores = {}
        else:
            bandit_action = baseline_action.value

        if self.mode == PolicyMode.HEURISTIC:
            recommended_action = baseline_action.value
            sg_result = self.safety_guard.evaluate_safety(recommended_action, state)
            final_action = sg_result.final_action

        elif self.mode == PolicyMode.SHADOW:
            recommended_action = bandit_action
            sg_result = self.safety_guard.evaluate_safety(baseline_action.value, state)
            final_action = sg_result.final_action

        elif self.mode == PolicyMode.BANDIT:
            recommended_action = bandit_action
            sg_result = self.safety_guard.evaluate_safety(recommended_action, state)
            final_action = sg_result.final_action

        else:
            recommended_action = DecisionAction.HUMAN_REVIEW.value
            sg_result = self.safety_guard.evaluate_safety(recommended_action, state)
            final_action = sg_result.final_action

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        requires_human = final_action in (
            DecisionAction.HUMAN_REVIEW.value,
            DecisionAction.ESCALATE.value,
        )

        # Build escalation reason (empty string for auto_resolve to match exact Phase 18 contract)
        if final_action == DecisionAction.AUTO_RESOLVE.value:
            reason = ""
        elif sg_result.was_overridden:
            reason = sg_result.override_reason
        else:
            reason = baseline_meta.get("reason", f"Decision set to {final_action}")

        # Preserve existing policy_features dictionary keys while extending
        existing_pf = dict(state.get("policy_features") or {})
        pf_dict = features.to_dict()
        pf_dict["verification_confidence"] = features.verification_score

        combined_policy_features = {**existing_pf, **pf_dict}

        previous_tickets = state.get("previous_tickets") or []
        if not isinstance(previous_tickets, list):
            previous_tickets = []
        conversation_history = state.get("conversation_history") or []
        conversation_count = (
            len(conversation_history)
            if isinstance(conversation_history, (list, tuple))
            else 0
        )
        has_historical_solution = any(
            bool(t.get("solution")) for t in previous_tickets if isinstance(t, dict)
        )

        memory_telemetry = {
            "historical_ticket_count": len(previous_tickets),
            "historical_memory_used": bool(previous_tickets),
            "conversation_memory_count": conversation_count,
            "historical_solution_available": has_historical_solution,
        }

        policy_telemetry = {
            "policy_name": (
                self.bandit_policy.MODEL_TYPE
                if self.mode != PolicyMode.HEURISTIC
                else "heuristic_baseline"
            ),
            "policy_version": self.bandit_policy.MODEL_VERSION,
            "policy_mode": self.mode.value,
            "feature_version": self.feature_extractor.FEATURE_VERSION,
            "recommended_action": recommended_action,
            "baseline_action": baseline_action.value,
            "final_action": final_action,
            "agreement": bool(recommended_action == baseline_action.value),
            "override": bool(sg_result.was_overridden),
            "override_reason": sg_result.override_reason,
            "inference_latency_ms": round(elapsed_ms, 3),
            "action_scores": bandit_scores,
            "safety_gate_passed": sg_result.safety_gate_passed,
            "gate_failed": sg_result.gate_failed,
            "memory": memory_telemetry,
        }

        existing_meta = dict(state.get("metadata") or {})

        return {
            "decision": final_action,
            "policy_action": final_action,
            "policy_features": combined_policy_features,
            "policy_metadata": policy_telemetry,
            "requires_human": requires_human,
            "escalation_reason": reason,
            "metadata": {
                **existing_meta,
                "policy": policy_telemetry,
                "memory": memory_telemetry,
            },
        }
