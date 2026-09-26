"""
Run ResolveX generation evaluation.
"""

import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(BACKEND_ROOT),
)


from evaluation.generation.evaluator import (
    GenerationEvaluator,
    RESULTS_PATH,
)


async def main():
    print("=" * 100)
    print("RESOLVEX GENERATION EVALUATION")
    print("=" * 100)

    evaluator = GenerationEvaluator()

    cases = evaluator.load_cases()

    print(f"Evaluation cases: {len(cases)}")
    print()

    results = await evaluator.evaluate_all()

    evaluator.save_results(
        results,
        RESULTS_PATH,
    )

    successful = [result for result in results if "error" not in result]

    failed = [result for result in results if "error" in result]

    fallback_count = sum(
        result.get(
            "fallback_used",
            False,
        )
        for result in successful
    )

    print()
    print("=" * 100)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 100)

    print(f"Total cases:       {len(results)}")

    print(f"Successful:        {len(successful)}")

    print(f"Errors:            {len(failed)}")

    print(f"LLM fallbacks:     {fallback_count}")

    print()
    print("Results saved to:")
    print(RESULTS_PATH)

    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
