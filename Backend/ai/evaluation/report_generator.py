"""
report_generator.py — Report Generator for Phase 20 Evaluation Framework.

Formats PolicyComparisonReport into JSON and clean GitHub-Flavored Markdown report tables.
"""

from typing import Dict, Any, Optional, Tuple
import json

from ai.evaluation.comparison import PolicyComparisonReport


class EvaluationReportGenerator:
    """
    Generates structured JSON and Markdown summary reports for evaluation results.
    """

    @classmethod
    def to_json(cls, report: PolicyComparisonReport, indent: int = 2) -> str:
        return json.dumps(report.to_dict(), indent=indent)

    @classmethod
    def to_markdown(cls, report: PolicyComparisonReport) -> str:
        """
        Generate neutral, structured GitHub-Flavored Markdown evaluation report.
        """
        ds_info = report.dataset_info
        pol_sums = report.policy_summaries
        comp = report.comparison
        action_dists = report.action_distributions
        overrides = report.safety_overrides

        lines = []
        lines.append("# ResolveX Policy & Agent Evaluation Report (Phase 20)")
        lines.append("")
        lines.append(f"**Dataset Name:** `{ds_info.get('name')}`  ")
        lines.append(f"**Source:** `{ds_info.get('source')}`  ")
        lines.append(f"**Sample Count:** `{ds_info.get('sample_count')}`  ")
        lines.append("")

        lines.append("## 1. Policy Performance Comparison Table")
        lines.append("")
        lines.append(
            "| Metric | Heuristic Baseline | LinUCB | Thompson Sampling | Available | Source |"
        )
        lines.append("|---|---|---|---|---|---|")

        metrics_to_show = [
            ("Average Reward", "average_reward"),
            ("Policy Success Rate", "policy_success_rate"),
            ("Resolution Success Rate", "resolution_success_rate"),
            ("Auto-Resolution Rate", "auto_resolution_rate"),
            ("Auto-Resolution Success Rate", "auto_resolution_success_rate"),
            ("Unsafe Auto-Resolution Rate", "unsafe_auto_resolution_rate"),
            ("Human Escalation Rate", "human_escalation_rate"),
            ("Escalation Precision", "escalation_precision"),
            ("Mean Decision Latency (ms)", "mean_decision_latency_ms"),
            ("Cost Per Ticket ($)", "cost_per_ticket"),
        ]

        def _format_m(m_dict: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
            if not m_dict or not m_dict.get("available"):
                return (
                    "N/A",
                    "False",
                    m_dict.get("source", "missing") if m_dict else "missing",
                )
            val = m_dict.get("value")
            if val is None:
                return "N/A", "False", m_dict.get("source", "missing")
            if isinstance(val, float):
                formatted = f"{val:.4f}"
            else:
                formatted = str(val)
            return formatted, "True", str(m_dict.get("source"))

        for label, m_key in metrics_to_show:
            h_m = pol_sums.get("heuristic", {}).get(m_key)
            l_m = pol_sums.get("linucb", {}).get(m_key)
            t_m = pol_sums.get("thompson_sampling", {}).get(m_key)

            h_val, avail, src = _format_m(h_m)
            l_val, _, _ = _format_m(l_m)
            t_val, _, _ = _format_m(t_m)

            lines.append(
                f"| **{label}** | {h_val} | {l_val} | {t_val} | {avail} | {src} |"
            )

        lines.append("")
        lines.append("## 2. Policy Agreement & Safety Override Metrics")
        lines.append("")
        lines.append(
            f"- **Agreement (Heuristic vs LinUCB):** `{comp.get('agreement_rate_heuristic_linucb', 0.0):.2%}`"
        )
        lines.append(
            f"- **Disagreement (Heuristic vs LinUCB):** `{comp.get('disagreement_rate_heuristic_linucb', 0.0):.2%}`"
        )
        lines.append(
            f"- **Agreement (Heuristic vs Thompson):** `{comp.get('agreement_rate_heuristic_thompson', 0.0):.2%}`"
        )
        lines.append(
            f"- **Safety Override Rate (LinUCB):** `{comp.get('safety_override_rate_linucb', 0.0):.2%}`"
        )
        lines.append(
            f"- **Total Safety Overrides:** `{overrides.get('total_overrides', 0)}`"
        )
        lines.append("")

        lines.append("## 3. Action Distributions")
        lines.append("")
        lines.append("| Action | Heuristic Count | LinUCB Count | Thompson Count |")
        lines.append("|---|---|---|---|")

        actions = ["auto_resolve", "ask_clarification", "human_review", "escalate"]
        for act in actions:
            h_c = action_dists.get("heuristic", {}).get(act, 0)
            l_c = action_dists.get("linucb", {}).get(act, 0)
            t_c = action_dists.get("thompson_sampling", {}).get(act, 0)
            lines.append(f"| `{act}` | {h_c} | {l_c} | {t_c} |")

        lines.append("")
        lines.append("## 4. Limitations & Ground Truth Availability")
        lines.append("")
        lines.append(
            "- Metrics marked `N/A` with `Available=False` indicate missing outcome/HITL ground truth."
        )
        lines.append(
            "- Performance on synthetic datasets is labeled source=`synthetic` and not represented as production claims."
        )
        lines.append(
            "- Safety Policy Guard remains authoritative over all bandit recommendations."
        )

        return "\n".join(lines)
