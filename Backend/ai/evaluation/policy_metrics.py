"""
policy_metrics.py — Policy / RL Metrics Evaluator for ResolveX (Phase 20.4).

Evaluates Heuristic, LinUCB, and Thompson Sampling policies on:
- Average Reward (observed vs proxy)
- Policy Success Rate
- Auto-Resolution Success Rate
- Human Escalation Rate (human_review + escalate)
- Human Review Rate & Escalation Rate
- Escalation Precision
- Decision Latency (mean, median, p95)
- Cost Per Ticket & Token Usage
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ai.evaluation.models import EvaluationRecord, MetricValue
from ai.policy.action_space import DecisionAction


class PolicyMetricsEvaluator:
    """
    Evaluator for policy decision quality, reward, latency, and resource costs.
    """

    @classmethod
    def evaluate(cls, records: List[EvaluationRecord]) -> Dict[str, MetricValue]:
        if not records:
            return {
                "average_reward": MetricValue.unavailable("Average Reward"),
                "policy_success_rate": MetricValue.unavailable("Policy Success Rate"),
                "auto_resolution_success_rate": MetricValue.unavailable(
                    "Auto-Resolution Success Rate"
                ),
                "human_escalation_rate": MetricValue.unavailable(
                    "Human Escalation Rate"
                ),
                "human_review_rate": MetricValue.unavailable("Human Review Rate"),
                "escalation_rate": MetricValue.unavailable("Escalation Rate"),
                "escalation_precision": MetricValue.unavailable("Escalation Precision"),
                "mean_decision_latency_ms": MetricValue.unavailable(
                    "Mean Decision Latency (ms)"
                ),
                "median_decision_latency_ms": MetricValue.unavailable(
                    "Median Decision Latency (ms)"
                ),
                "p95_decision_latency_ms": MetricValue.unavailable(
                    "P95 Decision Latency (ms)"
                ),
                "cost_per_ticket": MetricValue.unavailable("Cost Per Ticket"),
            }

        sample_count = len(records)

        # 1. Average Reward
        rewards = [r.reward for r in records if r.reward is not None]
        sources = [r.reward_source for r in records if r.reward_source is not None]
        primary_source = sources[0] if sources else "proxy"

        if rewards:
            avg_reward = MetricValue(
                value=float(np.mean(rewards)),
                available=True,
                source=primary_source,
                sample_count=len(rewards),
                description="Average Reward",
            )
        else:
            avg_reward = MetricValue.unavailable("Average Reward")

        # 2. Policy Success Rate
        pol_success_vals = [
            r.policy_success for r in records if r.policy_success is not None
        ]
        if pol_success_vals:
            succ_count = sum(1 for s in pol_success_vals if s)
            policy_success_rate = MetricValue(
                value=float(succ_count / len(pol_success_vals)),
                available=True,
                source="ground_truth",
                sample_count=len(pol_success_vals),
                numerator=succ_count,
                denominator=len(pol_success_vals),
                description="Policy Success Rate",
            )
        else:
            policy_success_rate = MetricValue.unavailable("Policy Success Rate")

        # 3. Auto-Resolution Success Rate
        auto_records = [
            r
            for r in records
            if r.final_action == DecisionAction.AUTO_RESOLVE.value or r.auto_resolved
        ]
        auto_success_vals = [
            r.auto_resolution_success
            for r in auto_records
            if r.auto_resolution_success is not None
        ]

        if auto_records and auto_success_vals:
            auto_succ_count = sum(1 for s in auto_success_vals if s)
            auto_res_success_rate = MetricValue(
                value=float(auto_succ_count / len(auto_success_vals)),
                available=True,
                source="hitl_or_outcome",
                sample_count=len(auto_success_vals),
                numerator=auto_succ_count,
                denominator=len(auto_success_vals),
                description="Auto-Resolution Success Rate",
            )
        else:
            auto_res_success_rate = MetricValue.unavailable(
                "Auto-Resolution Success Rate"
            )

        # 4. Human Escalation Rate, Human Review Rate, Escalation Rate
        human_review_count = sum(
            1 for r in records if r.final_action == DecisionAction.HUMAN_REVIEW.value
        )
        escalate_count = sum(
            1 for r in records if r.final_action == DecisionAction.ESCALATE.value
        )
        human_escalation_count = human_review_count + escalate_count

        human_escalation_rate = MetricValue(
            value=float(human_escalation_count / sample_count),
            available=True,
            source="final_action_telemetry",
            sample_count=sample_count,
            numerator=human_escalation_count,
            denominator=sample_count,
            description="Human Escalation Rate (human_review + escalate)",
        )

        human_review_rate = MetricValue(
            value=float(human_review_count / sample_count),
            available=True,
            source="final_action_telemetry",
            sample_count=sample_count,
            numerator=human_review_count,
            denominator=sample_count,
            description="Human Review Rate",
        )

        escalation_rate = MetricValue(
            value=float(escalate_count / sample_count),
            available=True,
            source="final_action_telemetry",
            sample_count=sample_count,
            numerator=escalate_count,
            denominator=sample_count,
            description="Escalation Rate",
        )

        # 5. Escalation Precision
        escalated_records = [
            r for r in records if r.final_action == DecisionAction.ESCALATE.value
        ]
        esc_precision_vals = [
            r.escalation_correct
            for r in escalated_records
            if r.escalation_correct is not None
        ]
        if escalated_records and esc_precision_vals:
            correct_esc = sum(1 for c in esc_precision_vals if c)
            escalation_precision = MetricValue(
                value=float(correct_esc / len(esc_precision_vals)),
                available=True,
                source="hitl_or_ground_truth",
                sample_count=len(esc_precision_vals),
                numerator=correct_esc,
                denominator=len(esc_precision_vals),
                description="Escalation Precision",
            )
        else:
            escalation_precision = MetricValue.unavailable("Escalation Precision")

        # 6. Decision Latency (ms)
        latencies = [
            r.decision_latency_ms for r in records if r.decision_latency_ms is not None
        ]
        if latencies:
            arr = np.array(latencies, dtype=np.float64)
            mean_lat = MetricValue(
                value=float(np.mean(arr)),
                available=True,
                source="telemetry",
                sample_count=len(arr),
                description="Mean Decision Latency (ms)",
            )
            med_lat = MetricValue(
                value=float(np.median(arr)),
                available=True,
                source="telemetry",
                sample_count=len(arr),
                description="Median Decision Latency (ms)",
            )
            p95_lat = MetricValue(
                value=float(np.percentile(arr, 95)),
                available=True,
                source="telemetry",
                sample_count=len(arr),
                description="P95 Decision Latency (ms)",
            )
        else:
            mean_lat = MetricValue.unavailable("Mean Decision Latency (ms)")
            med_lat = MetricValue.unavailable("Median Decision Latency (ms)")
            p95_lat = MetricValue.unavailable("P95 Decision Latency (ms)")

        # 7. Cost Per Ticket
        costs = [r.cost for r in records if r.cost is not None]
        if costs:
            cost_per_ticket = MetricValue(
                value=float(np.mean(costs)),
                available=True,
                source="telemetry",
                sample_count=len(costs),
                description="Cost Per Ticket ($)",
            )
        else:
            cost_per_ticket = MetricValue.unavailable("Cost Per Ticket ($)")

        return {
            "average_reward": avg_reward,
            "policy_success_rate": policy_success_rate,
            "auto_resolution_success_rate": auto_res_success_rate,
            "human_escalation_rate": human_escalation_rate,
            "human_review_rate": human_review_rate,
            "escalation_rate": escalation_rate,
            "escalation_precision": escalation_precision,
            "mean_decision_latency_ms": mean_lat,
            "median_decision_latency_ms": med_lat,
            "p95_decision_latency_ms": p95_lat,
            "cost_per_ticket": cost_per_ticket,
        }
