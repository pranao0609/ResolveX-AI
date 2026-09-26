import asyncio
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from evaluation.generation.evaluator import GenerationEvaluator

OUTPUT_DIR = BACKEND_ROOT / "data" / "evaluation" / "prompt_experiments"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


async def main() -> None:
    evaluator = GenerationEvaluator()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for prompt_version in (1, 2, 3):
        print()
        print("=" * 80)
        print(f"PROMPT VERSION {prompt_version} EVALUATION")
        print("=" * 80)

        results = await evaluator.evaluate_all(prompt_version=prompt_version)

        output_path = OUTPUT_DIR / f"prompt_v{prompt_version}_results.json"

        print(f"Saving results to: {output_path}")

        evaluator.save_results(
            results=results,
            output_path=output_path,
        )

        print(f"Completed prompt version {prompt_version}: " f"{len(results)} cases")


if __name__ == "__main__":
    asyncio.run(main())
