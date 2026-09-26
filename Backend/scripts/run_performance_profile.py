"""
run_performance_profile.py — CLI entrypoint for Phase 25 Performance Profiling.

Usage:
    python -m Backend.scripts.run_performance_profile
    OR
    python Backend/scripts/run_performance_profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv

    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

from ai.evaluation.performance_profiler import PerformanceProfilerEngine
from app.core.logger import logger


def main() -> int:
    print("=" * 70)
    print("        RESOLVEX 2.0 — PERFORMANCE PROFILING SYSTEM (PHASE 25)        ")
    print("=" * 70)

    try:
        profiler = PerformanceProfilerEngine()
        result = profiler.run_profiler()

        ret_sub = result["retrieval_subcomponents"]
        wf = result["workflow_profiles"]["component_profiles"]

        print("\n" + "-" * 70)
        print("MEASURED COMPONENT TIMINGS:")
        print(
            f"BM25 Search Mean:              {ret_sub['bm25']['mean']} ms (P50: {ret_sub['bm25']['p50']} ms)"
        )
        print(
            f"FAISS Dense Search Mean:       {ret_sub['faiss']['mean']} ms (P50: {ret_sub['faiss']['p50']} ms)"
        )
        print(
            f"Hybrid Fusion Merge Mean:      {ret_sub['hybrid_merge']['mean']} ms (P50: {ret_sub['hybrid_merge']['p50']} ms)"
        )
        print(
            f"Cross-Encoder Reranker Mean:   {ret_sub['reranker']['mean']} ms (P50: {ret_sub['reranker']['p50']} ms)"
        )
        print(f"Total Workflow P50 Latency:    {wf['total_workflow']['p50']} ms")
        print(f"Total Workflow Mean Latency:   {wf['total_workflow']['mean']} ms")
        print("-" * 70)
        print(
            f"Provider Rate-Limited (HTTP 429): {result['workflow_profiles']['provider_rate_limited']}"
        )
        print("Report saved to:")
        print(f"  - {result['json_report']}")
        print(f"  - {result['md_report']}")
        print("=" * 70 + "\n")

        return 0
    except Exception as exc:
        logger.error(f"[PerformanceProfile Error] {exc}", exc_info=True)
        print(f"\n[ERROR] Profiling failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
