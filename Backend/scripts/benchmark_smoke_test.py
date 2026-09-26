"""
benchmark_smoke_test.py — Deterministic CI smoke test for ResolveX benchmark engine and dataset.

Validates benchmark initialization, dataset schema, retrieval evaluation,
and summary metric calculations without requiring external LLM provider API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure Backend directory is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai.evaluation.final_benchmark_engine import FinalBenchmarkEngine
from app.core.logger import logger


def run_benchmark_smoke_test() -> int:
    """Run offline deterministic benchmark smoke test."""
    print("=" * 70)
    print("      RESOLVEX 2.0 -- BENCHMARK ENGINE SMOKE TEST (CI VALIDATION)      ")
    print("=" * 70)

    try:
        # 1. Initialize benchmark engine
        print("[1/5] Initializing FinalBenchmarkEngine...")
        engine = FinalBenchmarkEngine()

        # 2. Validate dataset loading and schema
        print("[2/5] Loading and validating benchmark dataset schema...")
        dataset = engine.load_dataset()
        if not dataset:
            raise ValueError("Benchmark dataset is empty!")

        required_fields = {
            "case_id",
            "title",
            "description",
            "expected_category",
            "expected_final_route",
        }

        for idx, case in enumerate(dataset):
            missing = required_fields - set(case.keys())
            if missing:
                raise ValueError(
                    f"Benchmark case #{idx} ({case.get('case_id')}) missing fields: {sorted(missing)}"
                )

        print(
            f"      [OK] Dataset loaded successfully: {len(dataset)} valid benchmark cases"
        )

        # 3. Evaluate deterministic retrieval pipeline
        print("[3/5] Running offline retrieval evaluation...")
        retrieval_metrics = engine.evaluate_retrieval(dataset)
        if (
            "hybrid" not in retrieval_metrics
            or "hybrid_reranker" not in retrieval_metrics
        ):
            raise ValueError("Retrieval metrics output missing expected keys!")

        hybrid_r5 = retrieval_metrics["hybrid"].get("recall_at_5", 0.0)
        reranked_r5 = retrieval_metrics["hybrid_reranker"].get("recall_at_5", 0.0)
        print(f"      [OK] Hybrid Recall@5: {hybrid_r5:.4f}")
        print(f"      [OK] Reranked Recall@5: {reranked_r5:.4f}")

        # 4. Validate metric computation logic
        print("[4/5] Testing metric aggregation logic...")
        mock_case_results = [
            {
                "case_id": "bench_smoke_01",
                "decision": "auto_resolve",
                "verification_passed": True,
                "evidence": ["KB-001"],
                "is_unsafe_auto_resolve": False,
                "routing_correct": True,
                "fallback_used": False,
            },
            {
                "case_id": "bench_smoke_02",
                "decision": "ask_clarification",
                "verification_passed": False,
                "evidence": [],
                "is_unsafe_auto_resolve": False,
                "routing_correct": True,
                "fallback_used": False,
            },
        ]
        summaries = engine.compute_summary_metrics(mock_case_results)
        if summaries["safety"]["unsafe_auto_resolution_count"] != 0:
            raise ValueError("Summary metric calculation incorrect!")
        print("      [OK] Summary metric aggregation verified")

        # 5. Output success
        print("[5/5] Smoke test complete.")
        print("-" * 70)
        print("BENCHMARK SMOKE TEST PASSED: Engine & dataset structure valid.")
        print("=" * 70)
        return 0

    except Exception as exc:
        logger.error(
            f"[BENCHMARK SMOKE TEST ERROR] Validation failed: {exc}",
            exc_info=True,
        )
        print(f"\n[ERROR] Benchmark smoke test failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_benchmark_smoke_test())
