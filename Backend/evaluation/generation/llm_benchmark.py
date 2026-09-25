from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ai.pipeline.ticket_pipeline import run_pipeline
from evaluation.metrics.llm_metrics import evaluate_llm_output


BACKEND_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BACKEND_ROOT
    / "evaluation"
    / "datasets"
    / "evaluation_dataset.jsonl"
)


class LLMBenchmark:
    """Fixed regression benchmark for ResolveX LLM outputs."""

    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
    ) -> None:
        self.dataset_path = dataset_path

    def load_dataset(self) -> list[dict]:
        """Load the fixed JSONL evaluation dataset."""

        cases = []

        with self.dataset_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    case = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}"
                    ) from exc

                cases.append(case)

        return cases

    @staticmethod
    def _build_ticket(
        case: dict,
        ticket_id: int,
    ) -> SimpleNamespace:
        """Build a lightweight ticket object for the pipeline."""

        return SimpleNamespace(
            id=ticket_id,
            title=case["ticket"][:255],
            description=case["ticket"],
            category="",
            submitted_by="llm_benchmark",
            image=None,
            attachment_paths=[],
        )

    @staticmethod
    def _build_context(
        context_docs: list[dict],
    ) -> str:
        """Convert retrieved documents into evaluation context."""

        parts = []

        for document in context_docs:
            title = document.get("title", "")
            category = document.get("category", "")
            content = document.get("content", "")

            parts.append(
                f"Title: {title}\n"
                f"Category: {category}\n"
                f"Content: {content}"
            )

        return "\n\n".join(parts)

    async def evaluate_case(
        self,
        case: dict,
        ticket_id: int,
        prompt_version: int = 2,
    ) -> dict:
        """Run one fixed benchmark case."""

        ticket = self._build_ticket(
            case,
            ticket_id,
        )

        pipeline_result = await run_pipeline(
            ticket,
            include_evaluation_details=True,
            prompt_version=prompt_version,
        )

        evaluation = pipeline_result.get(
            "evaluation",
            {},
        )

        context_docs = evaluation.get(
            "context_docs",
            [],
        )

        retrieved_context = self._build_context(
            context_docs
        )

        generated_answer = pipeline_result.get(
            "solution",
            "",
        )

        structured_output = pipeline_result.get(
            "structured_output"
        )

        metrics = evaluate_llm_output(
            ticket=case["ticket"],
            generated_answer=generated_answer,
            expected_resolution=case["expected_resolution"],
            retrieved_context=retrieved_context,
            structured_output=structured_output,
        )

        expected_context = set(
            case.get(
                "expected_context",
                [],
            )
        )

        retrieved_ids = {
            document.get("document_id")
            for document in context_docs
        }

        expected_context_recall = (
            len(
                expected_context
                & retrieved_ids
            )
            / len(expected_context)
            if expected_context
            else 0.0
        )

        return {
            "case_id": case["case_id"],
            "prompt_version": prompt_version,
            "ticket": case["ticket"],
            "expected_category": case[
                "expected_category"
            ],
            "actual_category": pipeline_result.get(
                "category"
            ),
            "expected_context": case[
                "expected_context"
            ],
            "retrieved_documents": context_docs,
            "context_recall": expected_context_recall,
            "expected_resolution": case[
                "expected_resolution"
            ],
            "generated_answer": generated_answer,
            "expected_escalation": case[
                "expected_escalation"
            ],
            "actual_escalation": pipeline_result.get(
                "requires_human"
            ),
            "structured_output": structured_output,
            "metrics": metrics,
            "pipeline_confidence": pipeline_result.get(
                "confidence",
                0.0,
            ),
            "llm_confidence": pipeline_result.get(
                "llm_confidence",
                0.0,
            ),
            "fallback_used": pipeline_result.get(
                "fallback_used",
                False,
            ),
            "retrieval_strategy": pipeline_result.get(
                "retrieval_strategy",
                evaluation.get(
                    "retrieval_strategy"
                ),
            ),
        }

    async def evaluate_all(
        self,
        prompt_version: int = 2,
    ) -> list[dict]:
        """Run the complete fixed benchmark."""

        cases = self.load_dataset()

        results = []

        for index, case in enumerate(cases):
            ticket_id = 910000 + index

            print(
                f"[LLM Benchmark] "
                f"{case['case_id']} "
                f"prompt_version={prompt_version}"
            )

            try:
                result = await self.evaluate_case(
                    case,
                    ticket_id,
                    prompt_version,
                )

                results.append(result)

                print(
                    f"    correctness="
                    f"{result['metrics']['correctness']:.4f} "
                    f"faithfulness="
                    f"{result['metrics']['faithfulness']:.4f} "
                    f"relevance="
                    f"{result['metrics']['relevance']:.4f} "
                    f"hallucination="
                    f"{result['metrics']['hallucination']:.4f} "
                    f"instruction="
                    f"{result['metrics']['instruction_adherence']:.4f} "
                    f"structured="
                    f"{result['metrics']['structured_output_validity']:.4f}"
                )

            except Exception as exc:
                results.append(
                    {
                        "case_id": case["case_id"],
                        "prompt_version": prompt_version,
                        "error": str(exc),
                    }
                )

                print(
                    f"    ERROR: {exc}"
                )

        return results

    @staticmethod
    def save_results(
        results: list[dict],
        output_path: Path,
    ) -> None:
        """Persist benchmark results."""

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Saved benchmark results to: "
            f"{output_path}"
        )