"""
runner.py — Evaluation Runner for ResolveX (Phase 20 Evaluation Runner).

Orchestrates:
1. Dataset loading (synthetic, historical, live)
2. Policy evaluation (Heuristic, LinUCB, Thompson Sampling)
3. Safety Policy Guard enforcement
4. Metric evaluation (Agent, Resolution, Policy, Safety)
5. HITL feedback recording & outcome reward mapping
6. Comparison report generation
"""

from typing import Dict, Any, List, Optional
import os
import json

from ai.evaluation.models import EvaluationRecord, MetricValue
from ai.evaluation.hitl_feedback import (
    HumanFeedback,
    HumanFeedbackStore,
    HITLRewardIntegrator,
)
from ai.evaluation.comparison import PolicyComparisonEvaluator, PolicyComparisonReport
from ai.policy.dataset import DecisionPolicyDataset, DecisionPolicySample
from ai.policy.feature_extractor import DecisionPolicyFeatureExtractor
from ai.policy.heuristic import HeuristicDecisionPolicy
from ai.policy.linucb import LinUCBPolicy
from ai.policy.thompson import ThompsonSamplingPolicy
from ai.policy.safety_guard import SafetyPolicyGuard
from ai.policy.policy_manager import PolicyManager, PolicyMode


class EvaluationRunner:
    """
    Offline-first evaluation runner for ResolveX decision policy subsystem.
    """

    def __init__(
        self,
        auto_resolve_threshold: float = 0.75,
        feedback_store: Optional[HumanFeedbackStore] = None,
    ):
        self.auto_resolve_threshold = auto_resolve_threshold
        self.feedback_store = feedback_store or HumanFeedbackStore()
        self.reward_integrator = HITLRewardIntegrator()
        self.comparison_evaluator = PolicyComparisonEvaluator(auto_resolve_threshold)
        self.safety_guard = SafetyPolicyGuard(auto_resolve_threshold)
        self.heuristic_policy = HeuristicDecisionPolicy(auto_resolve_threshold)

    def record_human_feedback(self, feedback: HumanFeedback) -> None:
        """
        Record HITL feedback without automatically updating online production policy weights.
        """
        self.feedback_store.save_feedback(feedback)

    def evaluate_state(
        self,
        state: Dict[str, Any],
        mode: str = "heuristic",
        feedback: Optional[HumanFeedback] = None,
    ) -> EvaluationRecord:
        """
        Evaluate single live or historical ResolveX state.
        """
        ticket_id = str(state.get("ticket_id") or "ticket_001")
        features = DecisionPolicyFeatureExtractor.extract_features(state)
        feature_dict = features.to_dict()

        manager = PolicyManager(
            mode=mode,
            auto_resolve_threshold=self.auto_resolve_threshold,
        )

        res = manager.evaluate(state)
        meta = res.get("policy_metadata") or {}

        # If HITL feedback passed directly or in store
        fb = feedback or self.feedback_store.get_feedback(ticket_id)
        reward_val, reward_src = self.reward_integrator.compute_reward_from_feedback(
            action=res["decision"],
            state=state,
            feedback=fb,
        )

        is_auto = res["decision"] == "auto_resolve"

        return EvaluationRecord(
            evaluation_id=f"eval_{ticket_id}",
            ticket_id=ticket_id,
            evaluation_source="hitl" if fb else "live_execution",
            policy_mode=mode,
            policy_name=meta.get("policy_name", "heuristic_baseline"),
            policy_version=meta.get("policy_version", "v1"),
            heuristic_action=meta.get("baseline_action", res["decision"]),
            rl_action=meta.get("recommended_action"),
            final_action=res["decision"],
            auto_resolved=is_auto,
            reward=reward_val,
            reward_source=reward_src,
            decision_latency_ms=meta.get("inference_latency_ms", 0.0),
            human_reviewed=bool(fb or res.get("requires_human")),
            human_accepted=fb.accepted if fb else None,
            human_action=fb.corrected_action if fb else None,
            human_correction=(not fb.accepted) if fb else None,
            human_feedback=fb.feedback_notes if fb else None,
            was_overridden=meta.get("override", False),
            override_reason=meta.get("override_reason", ""),
            safety_gate_passed=meta.get("safety_gate_passed", True),
            gate_failed=meta.get("gate_failed"),
            context_features=feature_dict,
        )

    def evaluate_dataset(
        self,
        dataset: DecisionPolicyDataset,
        linucb_policy: Optional[LinUCBPolicy] = None,
        thompson_policy: Optional[ThompsonSamplingPolicy] = None,
        dataset_name: str = "Evaluation Dataset",
        dataset_source: str = "synthetic",
    ) -> PolicyComparisonReport:
        """
        Evaluate and compare Heuristic, LinUCB, and Thompson Sampling policies on dataset.
        """
        return self.comparison_evaluator.compare_policies(
            dataset=dataset,
            linucb_policy=linucb_policy,
            thompson_policy=thompson_policy,
            dataset_name=dataset_name,
            dataset_source=dataset_source,
        )
