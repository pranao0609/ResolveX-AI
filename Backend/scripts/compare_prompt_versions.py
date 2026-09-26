import json
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = BACKEND_ROOT / "data" / "evaluation" / "prompt_experiments"

REPORT_DIR = BACKEND_ROOT / "evaluation" / "reports"

REPORT_PATH = REPORT_DIR / "phase11_prompt_comparison.json"


def load_results(prompt_version: int) -> list[dict]:
    path = EXPERIMENT_DIR / f"prompt_v{prompt_version}_results.json"

    if not path.exists():
        raise FileNotFoundError(f"Prompt V{prompt_version} results not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def calculate_summary(
    prompt_version: int,
    results: list[dict],
) -> dict:

    successful = [result for result in results if not result.get("error")]

    total = len(results)

    metric_names = [
        "faithfulness",
        "answer_relevance",
        "context_relevance",
        "answer_correctness",
        "evidence_support",
    ]

    metrics = {}

    for metric_name in metric_names:
        values = [
            float(
                result.get("metrics", {}).get(
                    metric_name,
                    0.0,
                )
            )
            for result in successful
        ]

        metrics[metric_name] = average(values)

    fallback_count = sum(
        bool(result.get("fallback_used", False)) for result in successful
    )

    pipeline_confidences = [
        float(result.get("confidence", 0.0)) for result in successful
    ]

    return {
        "prompt_version": prompt_version,
        "total_cases": total,
        "successful_cases": len(successful),
        "error_cases": total - len(successful),
        "metrics": metrics,
        "pipeline_confidence": average(pipeline_confidences),
        "fallback_count": fallback_count,
        "fallback_rate": (fallback_count / len(successful) if successful else 0.0),
    }


def main() -> None:
    summaries = []

    for prompt_version in (1, 2, 3):
        results = load_results(prompt_version)

        summary = calculate_summary(
            prompt_version,
            results,
        )

        summaries.append(summary)

    report = {
        "experiment": ("ResolveX Phase 11 " "Prompt Version Comparison"),
        "prompt_versions": [1, 2, 3],
        "cases_per_version": 10,
        "total_evaluations": 30,
        "metric_definition": (
            "Semantic generation metrics computed "
            "using the Phase 9 evaluation subsystem."
        ),
        "summaries": summaries,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 90)
    print("RESOLVEX PHASE 11 — " "PROMPT VERSION COMPARISON")
    print("=" * 90)

    for summary in summaries:
        version = summary["prompt_version"]

        print()
        print(f"Prompt V{version}")
        print("-" * 50)

        for name, value in summary["metrics"].items():
            print(f"{name:25s}: {value:.4f}")

        print(f"{'Pipeline confidence':25s}: " f"{summary['pipeline_confidence']:.4f}")

        print(f"{'Fallback count':25s}: " f"{summary['fallback_count']}")

        print(f"{'Fallback rate':25s}: " f"{summary['fallback_rate']:.2%}")

    print()
    print("=" * 90)
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()
