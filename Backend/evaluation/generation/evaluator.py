"""
ResolveX generation evaluation.

Runs the actual ResolveX ticket pipeline against manually curated
generation evaluation cases.

This evaluator does not modify or duplicate the production RAG/LLM
logic. It calls the same run_pipeline() used by the application.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from ai.pipeline.ticket_pipeline import run_pipeline
from evaluation.metrics.generation_metrics import (
    evaluate_generation_case,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]

CASES_PATH = (
    BACKEND_ROOT
    / "data"
    / "evaluation"
    / "generation_cases.json"
)

RESULTS_PATH = (
    BACKEND_ROOT
    / "data"
    / "evaluation"
    / "generation_results.json"
)


class GenerationEvaluator:
    """Evaluate ResolveX generation using the production pipeline."""

    def __init__(
        self,
        cases_path: Path = CASES_PATH,
    ):
        self.cases_path = Path(cases_path)

    def load_cases(self) -> list[dict]:
        """Load manually curated generation evaluation cases."""

        with open(
            self.cases_path,
            "r",
            encoding="utf-8",
        ) as file:
            cases = json.load(file)

        if not isinstance(cases, list):
            raise ValueError(
                "Generation evaluation dataset must be a JSON list."
            )

        return cases

    async def evaluate_case(
        self,
        case: dict,
        ticket_id: int,
        prompt_version: int = 1,
    ) -> dict:
        """
        Run one generation evaluation case through the
        actual ResolveX pipeline.
        """

        ticket = SimpleNamespace(
            id=ticket_id,
            description=case["ticket"],
            attachment_paths=None,
        )

        result = await run_pipeline(
            ticket,
            include_evaluation_details=True,
            prompt_version=prompt_version,
        )

        evaluation_data = result.get(
            "evaluation",
            {},
        )

        context_docs = evaluation_data.get(
            "context_docs",
            [],
        )

        retrieved_documents = [
            {
                "document_id": f"kb:{doc['index_id'] + 1}",
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "content": doc.get("content", ""),
                "score": doc.get("score", 0.0),
                "retriever": doc.get("retriever", ""),
            }
            for doc in context_docs
        ]

        metrics = evaluate_generation_case(
            ticket=case["ticket"],
            generated_answer=result["solution"],
            expected_answer=case["expected_answer"],
            retrieved_documents=retrieved_documents,
            required_evidence=case["required_evidence"],
        )

        return {
            "case_id": case["case_id"],
            "prompt_version": prompt_version,
            "ticket": case["ticket"],
            "relevant_documents": case[
                "relevant_documents"
            ],
            "expected_answer": case[
                "expected_answer"
            ],
            "required_evidence": case[
                "required_evidence"
            ],
            "generated_answer": result[
                "solution"
            ],
            "retrieved_documents": retrieved_documents,
            "category": result[
                "category"
            ],
            "confidence": result[
                "confidence"
            ],
            "fallback_used": result[
                "fallback_used"
            ],
            "retrieval_strategy": evaluation_data.get(
                "retrieval_strategy"
            ),
            "metrics": metrics,
        }

    async def evaluate_all(
        self,
        prompt_version: int = 1,
    ) -> list[dict]:
        """Run the complete generation evaluation dataset."""
    
        cases = self.load_cases()
        results = []
    
        for index, case in enumerate(
            cases,
            start=1,
        ):
            print(
                f"[{index}/{len(cases)}] "
                f"Evaluating {case['case_id']} "
                f"with prompt V{prompt_version}..."
            )
    
            try:
                result = await self.evaluate_case(
                    case=case,
                    ticket_id=900000 + index,
                    prompt_version=prompt_version,
                )
    
                results.append(result)
    
                print(
                    f"    fallback={result['fallback_used']} "
                    f"retrieved={len(result['retrieved_documents'])}"
                )
    
            except Exception as exc:
                print(
                    f"    ERROR: "
                    f"{type(exc).__name__}: {exc}"
                )
    
                results.append(
                    {
                        "case_id": case["case_id"],
                        "prompt_version": prompt_version,
                        "ticket": case["ticket"],
                        "relevant_documents": case[
                            "relevant_documents"
                        ],
                        "expected_answer": case[
                            "expected_answer"
                        ],
                        "required_evidence": case[
                            "required_evidence"
                        ],
                        "error": str(exc),
                    }
                )
    
        return results

    def save_results(
        self,
        results: list[dict],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved results to: {output_path}")