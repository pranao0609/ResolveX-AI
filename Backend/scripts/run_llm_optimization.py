"""
run_llm_optimization.py — CLI script for Phase 26 LLM Efficiency Optimization.

Usage:
    python -m Backend.scripts.run_llm_optimization
    OR
    python Backend/scripts/run_llm_optimization.py
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

from ai.evaluation.llm_optimizer import LLMOptimizerEngine
from app.core.logger import logger


def main() -> int:
    print("=" * 70)
    print("      RESOLVEX 2.0 — LLM EFFICIENCY OPTIMIZATION SYSTEM (PHASE 26)      ")
    print("=" * 70)

    try:
        engine = LLMOptimizerEngine()
        result = engine.run_optimization_suite()

        analysis = result["prompt_analysis"]

        print("\n" + "-" * 70)
        print("PROMPT OVERHEAD REDUCTION SUMMARY:")
        for agent, stats in analysis.items():
            print(
                f"  [{agent.upper()}] v1 Tokens: ~{stats['v1_est_tokens']} | v2 Tokens: ~{stats['v2_est_tokens']} | Reduction: {stats['pct_prompt_reduction']}%"
            )
        print("-" * 70)
        print("Report saved to:")
        print(f"  - {result['json_report']}")
        print(f"  - {result['md_report']}")
        print("=" * 70 + "\n")

        return 0
    except Exception as exc:
        logger.error(f"[LLM Optimization Error] {exc}", exc_info=True)
        print(f"\n[ERROR] LLM Optimization analysis failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
