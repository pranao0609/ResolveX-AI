"""
resolution_metrics.py — Resolution Metrics Evaluator for ResolveX (Phase 20.3).

Computes:
- Resolution Success Rate (marked unavailable if outcome ground truth absent)
- Human Acceptance Rate (computed on HITL-reviewed subset)
- Escalation Accuracy (marked unavailable if ground truth absent)
- Auto-Resolution Rate (calculated directly from final actions)
- Unsafe Auto-Resolution Rate (calculated from explicit verification failures, human rejections, or policy violations)
"""

from typing import List, Dict, Any, Optional
from ai.evaluation.models import EvaluationRecord, MetricValue
from ai.policy.action_space import DecisionAction


class ResolutionMetricsEvaluator:
    """
    Evaluator for ticket resolution outcomes and safety metrics.
    """

    @classmethod
    def evaluate(cls, records: List[EvaluationRecord]) -> Dict[str, MetricValue]:
        if not records:
            return {
                "resolution_success_rate": MetricValue.unavailable(
                    "Resolution Success Rate"
                ),
                "human_acceptance_rate": MetricValue.unavailable(
                    "Human Acceptance Rate"
                ),
                "escalation_accuracy": MetricValue.unavailable("Escalation Accuracy"),
                "auto_resolution_rate": MetricValue.unavailable("Auto-Resolution Rate"),
                "unsafe_auto_resolution_rate": MetricValue.unavailable(
                    "Unsafe Auto-Resolution Rate"
                ),
            }

        sample_count = len(records)

        # 1. Resolution Success Rate (requires explicit outcome evidence)
        res_success_vals = [
            r.resolution_success for r in records if r.resolution_success is not None
        ]
        if res_success_vals:
            num_success = sum(1 for s in res_success_vals if s)
            res_success_rate = MetricValue(
                value=float(num_success / len(res_success_vals)),
                available=True,
                source="hitl_or_outcome",
                sample_count=len(res_success_vals),
                numerator=num_success,
                denominator=len(res_success_vals),
                description="Resolution Success Rate",
            )
        else:
            res_success_rate = MetricValue.unavailable("Resolution Success Rate")

        # 2. Human Acceptance Rate (evaluated on HITL-reviewed samples)
        hitl_records = [
            r for r in records if r.human_reviewed and r.human_accepted is not None
        ]
        if hitl_records:
            accepted_count = sum(1 for r in hitl_records if r.human_accepted)
            human_acceptance_rate = MetricValue(
                value=float(accepted_count / len(hitl_records)),
                available=True,
                source="hitl",
                sample_count=len(hitl_records),
                numerator=accepted_count,
                denominator=len(hitl_records),
                description="Human Acceptance Rate",
            )
        else:
            human_acceptance_rate = MetricValue.unavailable("Human Acceptance Rate")

        # 3. Escalation Accuracy
        escalated_records = [
            r for r in records if r.final_action == DecisionAction.ESCALATE.value
        ]
        escalation_truth = [
            r.escalation_correct
            for r in escalated_records
            if r.escalation_correct is not None
        ]
        if escalated_records and escalation_truth:
            correct_count = sum(1 for c in escalation_truth if c)
            escalation_accuracy = MetricValue(
                value=float(correct_count / len(escalation_truth)),
                available=True,
                source="hitl_or_ground_truth",
                sample_count=len(escalation_truth),
                numerator=correct_count,
                denominator=len(escalation_truth),
                description="Escalation Accuracy",
            )
        else:
            escalation_accuracy = MetricValue.unavailable("Escalation Accuracy")

        # 4. Auto-Resolution Rate
        auto_resolved_count = sum(
            1
            for r in records
            if r.final_action == DecisionAction.AUTO_RESOLVE.value or r.auto_resolved
        )
        auto_resolution_rate = MetricValue(
            value=float(auto_resolved_count / sample_count),
            available=True,
            source="final_action_telemetry",
            sample_count=sample_count,
            numerator=auto_resolved_count,
            denominator=sample_count,
            description="Auto-Resolution Rate",
        )

        # 5. Unsafe Auto-Resolution Rate (unsafe = auto_resolved AND (unsafe_auto_resolution or failed_verification or human_rejected))
        auto_resolved_records = [
            r
            for r in records
            if r.final_action == DecisionAction.AUTO_RESOLVE.value or r.auto_resolved
        ]
        if auto_resolved_records:
            unsafe_count = sum(
                1
                for r in auto_resolved_records
                if (
                    r.unsafe_auto_resolution is True
                    or (r.human_accepted is False)
                    or (
                        r.was_overridden
                        and r.gate_failed in ("verification_failed", "errors")
                    )
                )
            )
            unsafe_rate = MetricValue(
                value=float(unsafe_count / len(auto_resolved_records)),
                available=True,
                source="safety_guard_and_hitl",
                sample_count=len(auto_resolved_records),
                numerator=unsafe_count,
                denominator=len(auto_resolved_records),
                description="Unsafe Auto-Resolution Rate",
            )
        else:
            unsafe_rate = MetricValue(
                value=0.0,
                available=True,
                source="final_action_telemetry",
                sample_count=0,
                numerator=0,
                denominator=0,
                description="Unsafe Auto-Resolution Rate",
            )

        return {
            "resolution_success_rate": res_success_rate,
            "human_acceptance_rate": human_acceptance_rate,
            "escalation_accuracy": escalation_accuracy,
            "auto_resolution_rate": auto_resolution_rate,
            "unsafe_auto_resolution_rate": unsafe_rate,
        }
