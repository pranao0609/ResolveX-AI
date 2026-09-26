import json
from pathlib import Path

RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evaluation"
    / "generation_results.json"
)


with open(
    RESULTS_PATH,
    "r",
    encoding="utf-8",
) as file:
    results = json.load(file)


results = [result for result in results if "metrics" in result]


print("=" * 120)
print("RESOLVEX GENERATION METRICS")
print("=" * 120)

print(
    f"{'Case':<10}"
    f"{'Faithfulness':>15}"
    f"{'Answer Rel.':>15}"
    f"{'Context Rel.':>15}"
    f"{'Correctness':>15}"
    f"{'Evidence':>15}"
)

print("-" * 120)

for result in results:
    metrics = result["metrics"]

    print(
        f"{result['case_id']:<10}"
        f"{metrics['faithfulness']:>15.4f}"
        f"{metrics['answer_relevance']:>15.4f}"
        f"{metrics['context_relevance']:>15.4f}"
        f"{metrics['answer_correctness']:>15.4f}"
        f"{metrics['evidence_support']:>15.4f}"
    )


print("-" * 120)

metric_names = [
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "answer_correctness",
    "evidence_support",
]

print()
print("AVERAGE SCORES")
print("-" * 40)

for metric in metric_names:
    average = sum(result["metrics"][metric] for result in results) / len(results)

    print(f"{metric:<25}: {average:.4f}")

print("=" * 120)
