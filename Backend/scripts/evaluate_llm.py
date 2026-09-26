import asyncio
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import path
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from evaluation.generation.llm_benchmark import LLMBenchmark

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_PATH = BACKEND_ROOT / "evaluation" / "reports" / "phase12_llm_evaluation.json"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def calculate_summary(
    results: list[dict],
) -> dict:

    successful = [result for result in results if "error" not in result]

    metric_names = [
        "correctness",
        "faithfulness",
        "relevance",
        "hallucination",
        "instruction_adherence",
        "structured_output_validity",
    ]

    metrics = {}

    for metric_name in metric_names:
        values = [float(result["metrics"][metric_name]) for result in successful]

        metrics[metric_name] = sum(values) / len(values) if values else 0.0

    fallback_count = sum(
        bool(result.get("fallback_used", False)) for result in successful
    )

    return {
        "total_cases": len(results),
        "successful_cases": len(successful),
        "error_cases": (len(results) - len(successful)),
        "metrics": metrics,
        "fallback_count": fallback_count,
        "fallback_rate": (fallback_count / len(successful) if successful else 0.0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    benchmark = LLMBenchmark()

    print("=" * 90)
    print("RESOLVEX PHASE 12 — " "LLM EVALUATION BENCHMARK")
    print("=" * 90)
    print()

    results = await benchmark.evaluate_all(
        prompt_version=2,
    )

    summary = calculate_summary(results)

    report = {
        "benchmark": ("ResolveX Phase 12 LLM Evaluation"),
        "prompt_version": 2,
        "dataset": "evaluation_dataset.jsonl",
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print()
    print("=" * 90)
    print("PHASE 12 RESULTS")
    print("=" * 90)

    print(f"Cases: {summary['total_cases']}")

    print(f"Successful: " f"{summary['successful_cases']}")

    print(f"Errors: " f"{summary['error_cases']}")

    print()

    for name, value in summary["metrics"].items():
        print(f"{name:25s}: {value:.4f}")

    print()

    print(f"Fallback rate            : " f"{summary['fallback_rate']:.2%}")

    print()
    print(f"Report saved to: " f"{OUTPUT_PATH}")

    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
