"""
comparison.py — Heuristic vs RL Policy Comparison Framework (Phase 20.5).

Evaluates Heuristic, LinUCB, and Thompson Sampling policies on the EXACT SAME dataset.
Generates structured PolicyComparisonReport containing metrics, agreement/disagreement rates,
action distributions, confusion matrices, and safety override analyses.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ai.evaluation.models import EvaluationRecord, MetricValue
from ai.evaluation.agent_metrics import AgentMetricsEvaluator
from ai.evaluation.resolution_metrics import ResolutionMetricsEvaluator
from ai.evaluation.policy_metrics import PolicyMetricsEvaluator
from ai.policy.action_space import ActionSpace, DecisionAction
from ai.policy.dataset import DecisionPolicyDataset, DecisionPolicySample
from ai.policy.heuristic import HeuristicDecisionPolicy
from ai.policy.linucb import LinUCBPolicy
from ai.policy.thompson import ThompsonSamplingPolicy
from ai.policy.safety_guard import SafetyPolicyGuard


@dataclass
class PolicyComparisonReport:
    """
    Structured comparison report comparing multiple decision policies on the same dataset.
    """

    dataset_info: Dict[str, Any]
    policy_summaries: Dict[str, Dict[str, Any]]
    comparison: Dict[str, Any]
    action_distributions: Dict[str, Dict[str, int]]
    confusion_matrices: Dict[str, Dict[str, Dict[str, int]]]
    divergence_cases: Dict[str, Any]
    safety_overrides: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PolicyComparisonEvaluator:
    """
    Orchestrates comparative evaluation of Heuristic, LinUCB, and Thompson Sampling policies.
    """

    def __init__(self, auto_resolve_threshold: float = 0.75):
        self.auto_resolve_threshold = auto_resolve_threshold
        self.heuristic_policy = HeuristicDecisionPolicy(auto_resolve_threshold)
        self.safety_guard = SafetyPolicyGuard(auto_resolve_threshold)

    def compare_policies(
        self,
        dataset: DecisionPolicyDataset,
        linucb_policy: Optional[LinUCBPolicy] = None,
        thompson_policy: Optional[ThompsonSamplingPolicy] = None,
        dataset_name: str = "Evaluation Dataset",
        dataset_source: str = "synthetic",
    ) -> PolicyComparisonReport:
        """
        Evaluate and compare policies on the exact same dataset samples.
        """
        dataset.sort_deterministically()
        sample_count = len(dataset)

        linucb = linucb_policy or LinUCBPolicy(alpha=0.5, random_seed=42)
        thompson = thompson_policy or ThompsonSamplingPolicy(random_seed=42)

        # Container for policy evaluation records
        records_by_policy: Dict[str, List[EvaluationRecord]] = {
            "heuristic": [],
            "linucb": [],
            "thompson_sampling": [],
        }

        action_dists: Dict[str, Dict[str, int]] = {
            "heuristic": {a: 0 for a in ActionSpace.all_action_names()},
            "linucb": {a: 0 for a in ActionSpace.all_action_names()},
            "thompson_sampling": {a: 0 for a in ActionSpace.all_action_names()},
        }

        # Detailed tracking
        agreements_h_l = 0
        agreements_h_t = 0

        rl_autoresolve_h_not = 0
        h_autoresolve_rl_not = 0

        override_reasons: Dict[str, int] = {}
        total_overrides = 0

        # Run evaluation sample by sample
        for sample in dataset.samples:
            ctx = sample.context
            ctx_dict = ctx.to_dict()

            # 1. Heuristic prediction
            h_action_enum, h_meta = self.heuristic_policy.predict_with_metadata(
                ctx_dict
            )
            h_act = h_action_enum.value
            action_dists["heuristic"][h_act] += 1

            # 2. LinUCB prediction & safety guard
            l_rec_enum, _ = linucb.predict(ctx)
            l_rec_act = l_rec_enum.value
            sg_l = self.safety_guard.evaluate_safety(l_rec_act, ctx_dict)
            l_final = sg_l.final_action
            action_dists["linucb"][l_final] += 1

            # 3. Thompson Sampling prediction & safety guard
            t_rec_enum, _ = thompson.predict(ctx)
            t_rec_act = t_rec_enum.value
            sg_t = self.safety_guard.evaluate_safety(t_rec_act, ctx_dict)
            t_final = sg_t.final_action
            action_dists["thompson_sampling"][t_final] += 1

            # Track agreements
            if h_act == l_final:
                agreements_h_l += 1
            if h_act == t_final:
                agreements_h_t += 1

            # Track divergences
            if (
                l_final == DecisionAction.AUTO_RESOLVE.value
                and h_act != DecisionAction.AUTO_RESOLVE.value
            ):
                rl_autoresolve_h_not += 1
            if (
                h_act == DecisionAction.AUTO_RESOLVE.value
                and l_final != DecisionAction.AUTO_RESOLVE.value
            ):
                h_autoresolve_rl_not += 1

            if sg_l.was_overridden:
                total_overrides += 1
                reason = sg_l.override_reason
                override_reasons[reason] = override_reasons.get(reason, 0) + 1

            # Build records for each policy
            r_val = sample.get_reward()
            is_obs = sample.observed_reward is not None

            # Heuristic record
            records_by_policy["heuristic"].append(
                EvaluationRecord(
                    evaluation_id=f"eval_h_{sample.sample_id}",
                    ticket_id=sample.sample_id,
                    evaluation_source=dataset_source,
                    policy_mode="heuristic",
                    policy_name="heuristic_baseline",
                    heuristic_action=h_act,
                    final_action=h_act,
                    auto_resolved=(h_act == DecisionAction.AUTO_RESOLVE.value),
                    reward=r_val,
                    reward_source="observed" if is_obs else "proxy",
                    human_escalation=(
                        h_act
                        in (
                            DecisionAction.HUMAN_REVIEW.value,
                            DecisionAction.ESCALATE.value,
                        )
                    ),
                    context_features=ctx_dict,
                )
            )

            # LinUCB record
            records_by_policy["linucb"].append(
                EvaluationRecord(
                    evaluation_id=f"eval_l_{sample.sample_id}",
                    ticket_id=sample.sample_id,
                    evaluation_source=dataset_source,
                    policy_mode="bandit",
                    policy_name="linucb",
                    heuristic_action=h_act,
                    rl_action=l_rec_act,
                    final_action=l_final,
                    auto_resolved=(l_final == DecisionAction.AUTO_RESOLVE.value),
                    reward=r_val,
                    reward_source="observed" if is_obs else "proxy",
                    human_escalation=(
                        l_final
                        in (
                            DecisionAction.HUMAN_REVIEW.value,
                            DecisionAction.ESCALATE.value,
                        )
                    ),
                    was_overridden=sg_l.was_overridden,
                    override_reason=sg_l.override_reason,
                    safety_gate_passed=sg_l.safety_gate_passed,
                    gate_failed=sg_l.gate_failed,
                    context_features=ctx_dict,
                )
            )

            # Thompson Sampling record
            records_by_policy["thompson_sampling"].append(
                EvaluationRecord(
                    evaluation_id=f"eval_t_{sample.sample_id}",
                    ticket_id=sample.sample_id,
                    evaluation_source=dataset_source,
                    policy_mode="bandit",
                    policy_name="thompson_sampling",
                    heuristic_action=h_act,
                    rl_action=t_rec_act,
                    final_action=t_final,
                    auto_resolved=(t_final == DecisionAction.AUTO_RESOLVE.value),
                    reward=r_val,
                    reward_source="observed" if is_obs else "proxy",
                    human_escalation=(
                        t_final
                        in (
                            DecisionAction.HUMAN_REVIEW.value,
                            DecisionAction.ESCALATE.value,
                        )
                    ),
                    was_overridden=sg_t.was_overridden,
                    override_reason=sg_t.override_reason,
                    safety_gate_passed=sg_t.safety_gate_passed,
                    gate_failed=sg_t.gate_failed,
                    context_features=ctx_dict,
                )
            )

        # Summarize policy metrics
        policy_summaries = {}
        for pol_name, recs in records_by_policy.items():
            pm = PolicyMetricsEvaluator.evaluate(recs)
            rm = ResolutionMetricsEvaluator.evaluate(recs)
            am = AgentMetricsEvaluator.evaluate(recs)
            combined = {
                **{k: v.to_dict() for k, v in pm.items()},
                **{k: v.to_dict() for k, v in rm.items()},
                **{k: v.to_dict() for k, v in am.items()},
            }
            policy_summaries[pol_name] = combined

        # Build confusion matrices (Heuristic vs LinUCB, Heuristic vs Thompson)
        cm_linucb = self._build_confusion_matrix(
            records_by_policy["heuristic"], records_by_policy["linucb"]
        )
        cm_thompson = self._build_confusion_matrix(
            records_by_policy["heuristic"], records_by_policy["thompson_sampling"]
        )

        N = float(max(1, sample_count))

        comparison_dict = {
            "agreement_rate_heuristic_linucb": float(agreements_h_l / N),
            "disagreement_rate_heuristic_linucb": float(
                (sample_count - agreements_h_l) / N
            ),
            "agreement_rate_heuristic_thompson": float(agreements_h_t / N),
            "disagreement_rate_heuristic_thompson": float(
                (sample_count - agreements_h_t) / N
            ),
            "safety_override_rate_linucb": float(total_overrides / N),
        }

        divergence_dict = {
            "linucb_autoresolve_heuristic_not": rl_autoresolve_h_not,
            "heuristic_autoresolve_linucb_not": h_autoresolve_rl_not,
        }

        safety_overrides_dict = {
            "total_overrides": total_overrides,
            "override_rate": float(total_overrides / N),
            "override_reason_distribution": override_reasons,
        }

        dataset_info = {
            "name": dataset_name,
            "source": dataset_source,
            "sample_count": sample_count,
        }

        return PolicyComparisonReport(
            dataset_info=dataset_info,
            policy_summaries=policy_summaries,
            comparison=comparison_dict,
            action_distributions=action_dists,
            confusion_matrices={
                "heuristic_vs_linucb": cm_linucb,
                "heuristic_vs_thompson": cm_thompson,
            },
            divergence_cases=divergence_dict,
            safety_overrides=safety_overrides_dict,
        )

    def _build_confusion_matrix(
        self, recs_base: List[EvaluationRecord], recs_target: List[EvaluationRecord]
    ) -> Dict[str, Dict[str, int]]:
        actions = ActionSpace.all_action_names()
        cm = {a_base: {a_target: 0 for a_target in actions} for a_base in actions}

        for r_b, r_t in zip(recs_base, recs_target):
            act_b = r_b.final_action
            act_t = r_t.final_action
            if act_b in cm and act_t in cm[act_b]:
                cm[act_b][act_t] += 1

        return cm
