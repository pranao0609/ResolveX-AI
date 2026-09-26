"""
evaluation.py — Offline Policy Evaluation (OPE) Framework for ResolveX.

Evaluates decision policies (Heuristic, LinUCB, Thompson Sampling) on a DecisionPolicyDataset.
Calculates reward metrics, safety metrics, baseline agreement rates, action distributions, and OPE (IPS/SNIPS) when propensities are available.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import numpy as np
from ai.policy.action_space import ActionSpace, DecisionAction
from ai.policy.dataset import DecisionPolicyDataset, DecisionPolicySample
from ai.policy.heuristic import HeuristicDecisionPolicy
from ai.policy.safety_guard import SafetyPolicyGuard


@dataclass
class EvaluationReport:
    policy_name: str
    sample_count: int
    reward_metrics: Dict[str, float]
    safety_metrics: Dict[str, Any]
    action_distribution: Dict[str, int]
    baseline_agreement_rate: float
    ope_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "sample_count": self.sample_count,
            "reward_metrics": self.reward_metrics,
            "safety_metrics": self.safety_metrics,
            "action_distribution": self.action_distribution,
            "baseline_agreement_rate": float(self.baseline_agreement_rate),
            "ope_metrics": self.ope_metrics,
            "metadata": self.metadata,
        }


class OfflinePolicyEvaluator:
    """
    Evaluator for comparing policies against logged offline datasets.
    """

    def __init__(
        self,
        auto_resolve_threshold: float = 0.75,
        safety_guard: Optional[SafetyPolicyGuard] = None,
    ):
        self.heuristic_baseline = HeuristicDecisionPolicy(auto_resolve_threshold)
        self.safety_guard = safety_guard or SafetyPolicyGuard(auto_resolve_threshold)

    def evaluate(self, policy: Any, dataset: DecisionPolicyDataset) -> EvaluationReport:
        """
        Evaluate given policy object (must implement predict(context)) on dataset.
        """
        policy_name = getattr(
            policy, "MODEL_TYPE", getattr(policy, "__class__", type(policy)).__name__
        )
        dataset.sort_deterministically()

        if len(dataset) == 0:
            return EvaluationReport(
                policy_name=policy_name,
                sample_count=0,
                reward_metrics={
                    "mean_observed_reward": 0.0,
                    "mean_proxy_reward": 0.0,
                    "total_reward": 0.0,
                },
                safety_metrics={
                    "safety_violation_count": 0,
                    "unsafe_recommendation_count": 0,
                    "override_count": 0,
                    "override_rate": 0.0,
                },
                action_distribution={a: 0 for a in ActionSpace.all_action_names()},
                baseline_agreement_rate=1.0,
                ope_metrics={
                    "ope_available": False,
                    "ips_reward": None,
                    "snips_reward": None,
                    "reason": "Empty dataset",
                },
            )

        action_counts: Dict[str, int] = {a: 0 for a in ActionSpace.all_action_names()}
        agreements = 0
        unsafe_recs = 0
        overrides = 0
        safety_violations = 0

        total_obs_reward = 0.0
        total_proxy_reward = 0.0

        ips_num = 0.0
        ips_den = 0.0
        propensities_valid = True

        for sample in dataset.samples:
            ctx = sample.context
            ctx_dict = ctx.to_dict()

            if hasattr(policy, "predict"):
                res = policy.predict(ctx)
                if isinstance(res, tuple):
                    res_act = res[0]
                else:
                    res_act = res
                pred_action = (
                    res_act.value
                    if isinstance(res_act, DecisionAction)
                    else str(res_act)
                )
            else:
                pred_action = DecisionAction.HUMAN_REVIEW.value

            pred_action = pred_action.lower()
            if pred_action not in action_counts:
                pred_action = DecisionAction.HUMAN_REVIEW.value
            action_counts[pred_action] += 1

            baseline_action = self.heuristic_baseline.predict(ctx).value

            if pred_action == baseline_action:
                agreements += 1

            sg_res = self.safety_guard.evaluate_safety(pred_action, ctx_dict)
            if sg_res.was_overridden:
                overrides += 1
                if pred_action == DecisionAction.AUTO_RESOLVE.value and (
                    ctx.verification_failed > 0 or ctx.errors_present > 0
                ):
                    unsafe_recs += 1

            if ctx.safety_violation > 0:
                safety_violations += 1

            obs_r = (
                sample.observed_reward if sample.observed_reward is not None else 0.0
            )
            proxy_r = sample.proxy_reward if sample.proxy_reward is not None else 0.0
            total_obs_reward += obs_r
            total_proxy_reward += proxy_r

            if sample.propensity is None or sample.propensity <= 0.0:
                propensities_valid = False
            elif propensities_valid:
                p_i = sample.propensity
                if pred_action == sample.action:
                    weight = 1.0 / p_i
                    ips_num += weight * obs_r
                    ips_den += weight

        N = len(dataset)
        agreement_rate = agreements / float(N)
        override_rate = overrides / float(N)

        reward_metrics = {
            "mean_observed_reward": float(total_obs_reward / N),
            "mean_proxy_reward": float(total_proxy_reward / N),
            "total_observed_reward": float(total_obs_reward),
            "total_proxy_reward": float(total_proxy_reward),
        }

        safety_metrics = {
            "safety_violation_count": safety_violations,
            "unsafe_recommendation_count": unsafe_recs,
            "override_count": overrides,
            "override_rate": float(override_rate),
        }

        if propensities_valid and N > 0:
            ips_val = float(ips_num / N)
            snips_val = float(ips_num / ips_den) if ips_den > 0 else 0.0
            ope_metrics = {
                "ope_available": True,
                "ips_reward": ips_val,
                "snips_reward": snips_val,
                "reason": "Propensities available",
            }
        else:
            ope_metrics = {
                "ope_available": False,
                "ips_reward": None,
                "snips_reward": None,
                "reason": "Propensities missing or non-positive in dataset samples",
            }

        return EvaluationReport(
            policy_name=policy_name,
            sample_count=N,
            reward_metrics=reward_metrics,
            safety_metrics=safety_metrics,
            action_distribution=action_counts,
            baseline_agreement_rate=agreement_rate,
            ope_metrics=ope_metrics,
            metadata={
                "auto_resolve_threshold": self.heuristic_baseline.auto_resolve_threshold
            },
        )
