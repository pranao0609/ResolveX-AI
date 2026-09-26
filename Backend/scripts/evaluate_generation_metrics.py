"""
Calculate generation metrics from existing ResolveX
generation evaluation results.

This script does not call the LLM.
"""

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(BACKEND_ROOT),
)


from evaluation.metrics.generation_metrics import (
    evaluate_generation_case,
)

INPUT_PATH = BACKEND_ROOT / "data" / "evaluation" / "generation_results.json"

OUTPUT_PATH = (
    BACKEND_ROOT / "data" / "evaluation" / "generation_results_with_metrics.json"
)


def main():
    """Calculate and persist generation metrics."""

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    successful_results = []

    for result in results:

        if "error" in result:
            continue

        metrics = evaluate_generation_case(
            ticket=result["ticket"],
            generated_answer=result["generated_answer"],
            expected_answer=result["expected_answer"],
            retrieved_documents=result["retrieved_documents"],
            required_evidence=result["required_evidence"],
        )

        # IMPORTANT:
        # Store the newly calculated metrics
        # back into the result object.
        result["metrics"] = metrics

        successful_results.append(result)

    # Save the UPDATED results.
    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 100)
    print("RESOLVEX GENERATION METRICS")
    print("=" * 100)

    print(f"Evaluation cases: " f"{len(successful_results)}")

    metric_names = [
        "faithfulness",
        "answer_relevance",
        "context_relevance",
        "answer_correctness",
        "evidence_support",
    ]

    print()

    for metric in metric_names:

        values = [result["metrics"][metric] for result in successful_results]

        average = sum(values) / len(values) if values else 0.0

        print(f"{metric:<25} " f"{average:.4f}")

    print()
    print("Detailed results saved to:")
    print(OUTPUT_PATH)

    print("=" * 100)


if __name__ == "__main__":
    main()
