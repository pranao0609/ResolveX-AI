"""
run_final_benchmark.py — Command-line interface to execute ResolveX Phase 24 Final Evaluation.

Usage:
    python -m Backend.scripts.run_final_benchmark
    OR
    python Backend/scripts/run_final_benchmark.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure Backend directory is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure Backend/.env is loaded if available
try:
    from dotenv import load_dotenv

    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

from ai.evaluation.final_benchmark_engine import FinalBenchmarkEngine
from app.core.logger import logger


def main() -> int:
    """Run full final evaluation suite and exit with code 0 on success."""
    print("=" * 70)
    print("      RESOLVEX 2.0 — FINAL BENCHMARK & EVALUATION SYSTEM (PHASE 24)      ")
    print("=" * 70)

    try:
        engine = FinalBenchmarkEngine()
        report = engine.run_full_benchmark()

        meta = report.get("metadata", {})
        ret = report.get("retrieval", {})
        rer = report.get("reranking", {})
        ver = report.get("verification", {})
        pol = report.get("policy", {})
        safe = report.get("safety", {})
        e2e = report.get("end_to_end", {})
        lat = report.get("latency", {})

        print("\n" + "-" * 70)
        print("SUMMARY RESULTS:")
        print(f"Timestamp:                    {meta.get('timestamp')}")
        print(f"Git Commit SHA:               {meta.get('git_commit_sha')}")
        print(
            f"Total Test Cases Evaluated:   {report.get('dataset', {}).get('total_cases')}"
        )
        print(
            f"Retrieval Recall@5 (Hybrid):  {ret.get('hybrid', {}).get('recall_at_5', 'N/A')}"
        )
        print(
            f"Reranked Recall@5 (CrossEnc): {ret.get('hybrid_reranker', {}).get('recall_at_5', 'N/A')}"
        )
        print(
            f"Verification Pass Rate:       {ver.get('verification_pass_rate', 0.0) * 100:.1f}%"
        )
        print(
            f"Auto-Resolution Rate:         {pol.get('auto_resolution_rate', 0.0) * 100:.1f}%"
        )
        print(
            f"Safety Routing Accuracy:      {safe.get('safety_routing_accuracy', 0.0) * 100:.1f}%"
        )
        print(
            f"Unsafe Auto-Resolutions:      {safe.get('unsafe_auto_resolution_count', 0)} (TARGET: 0)"
        )
        print(
            f"Workflow P50 Latency:         {lat.get('total_workflow', {}).get('p50', 0.0)} ms"
        )
        print("-" * 70)
        print("Report saved to:")
        print("  - Backend/data/evaluation/resolveX_final_benchmark.json")
        print("  - Backend/data/evaluation/resolveX_final_benchmark.md")
        print("=" * 70 + "\n")

        # Exit code: nonzero if unsafe auto resolutions > 0
        if safe.get("unsafe_auto_resolution_count", 0) > 0:
            logger.error("[FINAL BENCHMARK FAILED] Safety violation detected!")
            return 1

        return 0

    except Exception as exc:
        logger.error(f"[FINAL BENCHMARK ERROR] Execution failed: {exc}", exc_info=True)
        print(f"\n[ERROR] Benchmark execution failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
