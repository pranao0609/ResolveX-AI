"""
smoke_test_langsmith.py — Manual smoke test for LangSmith observability in ResolveX.

USAGE (do NOT commit your API key):
  set LANGSMITH_TRACING=true
  set LANGSMITH_API_KEY=lsv2_pt_your_real_key_here
  set LANGSMITH_PROJECT=ResolveX-SmokeTest
  python Backend/scripts/smoke_test_langsmith.py

This script executes a sample ticket through the ResolveX LangGraph executor
and prints the resulting decision and execution metadata. If LangSmith tracing
is enabled in the environment, the trace will be visible in your LangSmith dashboard.
"""

import os
import sys

# Ensure Backend is in python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ai.graph.executor import execute_resolvex_graph
from ai.observability import is_langsmith_enabled


class SmokeTestTicket:
    def __init__(
        self,
        ticket_id=9001,
        title="Database connection failure",
        description="Application cannot connect to PostgreSQL database after network maintenance.",
    ):
        self.id = ticket_id
        self.title = title
        self.description = description
        self.attachment_paths = []


def run_smoke_test():
    print("=" * 60)
    print("RESOLVEX LANGSMITH OBSERVABILITY SMOKE TEST")
    print("=" * 60)

    enabled = is_langsmith_enabled()
    api_key_present = bool(
        os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    )

    print(f"LANGSMITH_TRACING enabled: {enabled}")
    print(f"API Key present:            {api_key_present}")

    if not enabled or not api_key_present:
        print("\nNOTE: Tracing is currently DISABLED or missing API key.")
        print("To test live LangSmith tracing, set environment variables:")
        print("  set LANGSMITH_TRACING=true")
        print("  set LANGSMITH_API_KEY=your_langsmith_api_key")
        print("  set LANGSMITH_PROJECT=ResolveX")

    print("\nExecuting sample ticket through ResolveX LangGraph...")
    ticket = SmokeTestTicket()
    result = execute_resolvex_graph(ticket, include_evaluation_details=True)

    print("\n" + "=" * 60)
    print("EXECUTION RESULT SUMMARY")
    print("=" * 60)
    print(f"Decision:            {result.get('decision')}")
    print(f"Auto Resolved:       {result.get('auto_resolved')}")
    print(f"Category:            {result.get('category')}")
    print(f"Confidence:          {result.get('confidence'):.2f}")
    print(f"Graph Run ID:        {result.get('graph_run_id')}")

    eval_details = result.get("evaluation", {})
    print(
        f"Retrieval Strategy:  {eval_details.get('retrieval_metadata', {}).get('strategy', 'N/A')}"
    )
    print(f"Errors Recorded:     {len(result.get('errors', []))}")
    print(f"Warnings Recorded:   {len(result.get('warnings', []))}")

    print("\n" + "=" * 60)
    if enabled and api_key_present:
        print("SMOKE TEST COMPLETE: Check your LangSmith dashboard for trace tree.")
    else:
        print("SMOKE TEST COMPLETE: Executed successfully in offline/disabled mode.")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
