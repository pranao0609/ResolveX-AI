"""
agent_metrics.py — Agent Execution Metrics Evaluator for ResolveX (Phase 20.2).

Computes:
- Task Success Rate
- Tool Call Accuracy (returns unavailable if ground truth absent)
- Unnecessary Tool Calls & Unnecessary Tool Call Rate (returns unavailable if ground truth absent)
- Agent Latency (mean, median, p95)
- Agent Failure Rate
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ai.evaluation.models import EvaluationRecord, MetricValue


class AgentMetricsEvaluator:
    """
    Evaluator for agent execution performance telemetry.
    """

    @classmethod
    def evaluate(cls, records: List[EvaluationRecord]) -> Dict[str, MetricValue]:
        if not records:
            return {
                "task_success_rate": MetricValue.unavailable("Task Success Rate"),
                "tool_call_accuracy": MetricValue.unavailable("Tool Call Accuracy"),
                "unnecessary_tool_call_rate": MetricValue.unavailable(
                    "Unnecessary Tool Call Rate"
                ),
                "agent_failure_rate": MetricValue.unavailable("Agent Failure Rate"),
                "mean_agent_latency_ms": MetricValue.unavailable(
                    "Mean Agent Latency (ms)"
                ),
                "median_agent_latency_ms": MetricValue.unavailable(
                    "Median Agent Latency (ms)"
                ),
                "p95_agent_latency_ms": MetricValue.unavailable(
                    "P95 Agent Latency (ms)"
                ),
            }

        sample_count = len(records)

        # 1. Task Success Rate
        task_success_vals = [
            r.task_success for r in records if r.task_success is not None
        ]
        if task_success_vals:
            num_success = sum(1 for v in task_success_vals if v)
            task_success_rate = MetricValue(
                value=float(num_success / len(task_success_vals)),
                available=True,
                source="telemetry",
                sample_count=len(task_success_vals),
                numerator=num_success,
                denominator=len(task_success_vals),
                description="Task Success Rate",
            )
        else:
            task_success_rate = MetricValue.unavailable("Task Success Rate")

        # 2. Tool Call Accounting
        total_tool_calls = sum(r.tool_calls for r in records)
        accurate_tool_calls_vals = [
            r.accurate_tool_calls for r in records if r.accurate_tool_calls is not None
        ]
        unnecessary_tool_calls_vals = [
            r.unnecessary_tool_calls
            for r in records
            if r.unnecessary_tool_calls is not None
        ]

        if total_tool_calls > 0 and accurate_tool_calls_vals:
            acc_sum = sum(accurate_tool_calls_vals)
            tool_call_accuracy = MetricValue(
                value=float(acc_sum / total_tool_calls),
                available=True,
                source="telemetry",
                sample_count=sample_count,
                numerator=acc_sum,
                denominator=total_tool_calls,
                description="Tool Call Accuracy",
            )
        else:
            tool_call_accuracy = MetricValue.unavailable("Tool Call Accuracy")

        if total_tool_calls > 0 and unnecessary_tool_calls_vals:
            unnecessary_sum = sum(unnecessary_tool_calls_vals)
            unnecessary_rate = MetricValue(
                value=float(unnecessary_sum / total_tool_calls),
                available=True,
                source="telemetry",
                sample_count=sample_count,
                numerator=unnecessary_sum,
                denominator=total_tool_calls,
                description="Unnecessary Tool Call Rate",
            )
        else:
            unnecessary_rate = MetricValue.unavailable("Unnecessary Tool Call Rate")

        # 3. Agent Failure Rate
        failure_vals = [r.agent_failure for r in records if r.agent_failure is not None]
        if failure_vals:
            num_failures = sum(1 for f in failure_vals if f)
            failure_rate = MetricValue(
                value=float(num_failures / len(failure_vals)),
                available=True,
                source="telemetry",
                sample_count=len(failure_vals),
                numerator=num_failures,
                denominator=len(failure_vals),
                description="Agent Failure Rate",
            )
        else:
            # Fallback to checking error logs
            num_failures = sum(
                1 for r in records if r.was_overridden and r.gate_failed == "errors"
            )
            failure_rate = MetricValue(
                value=float(num_failures / sample_count),
                available=True,
                source="telemetry",
                sample_count=sample_count,
                numerator=num_failures,
                denominator=sample_count,
                description="Agent Failure Rate",
            )

        # 4. Agent Latency (mean, median, p95)
        latencies = [
            r.agent_latency_ms for r in records if r.agent_latency_ms is not None
        ]
        if latencies:
            arr = np.array(latencies, dtype=np.float64)
            mean_lat = MetricValue(
                value=float(np.mean(arr)),
                available=True,
                source="telemetry",
                sample_count=len(arr),
                description="Mean Agent Latency (ms)",
            )
            med_lat = MetricValue(
                value=float(np.median(arr)),
                available=True,
                source="telemetry",
                sample_count=len(arr),
                description="Median Agent Latency (ms)",
            )
            p95_lat = MetricValue(
                value=float(np.percentile(arr, 95)),
                available=True,
                source="telemetry",
                sample_count=len(arr),
                description="P95 Agent Latency (ms)",
            )
        else:
            mean_lat = MetricValue.unavailable("Mean Agent Latency (ms)")
            med_lat = MetricValue.unavailable("Median Agent Latency (ms)")
            p95_lat = MetricValue.unavailable("P95 Agent Latency (ms)")

        return {
            "task_success_rate": task_success_rate,
            "tool_call_accuracy": tool_call_accuracy,
            "unnecessary_tool_call_rate": unnecessary_rate,
            "agent_failure_rate": failure_rate,
            "mean_agent_latency_ms": mean_lat,
            "median_agent_latency_ms": med_lat,
            "p95_agent_latency_ms": p95_lat,
        }
