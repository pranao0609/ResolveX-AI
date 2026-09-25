"""
Generate the consolidated ResolveX Phase 9 evaluation report.

Combines:
    - Retrieval evaluation results
    - Generation evaluation results
    - Evaluation dataset metadata
    - Methodology
    - Limitations

No LLM API calls are performed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = (
    BACKEND_ROOT
    / "data"
    / "evaluation"
)

REPORT_DIR = (
    BACKEND_ROOT
    / "evaluation"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RETRIEVAL_RESULTS_PATH = (
    EVALUATION_DIR
    / "retrieval_results.json"
)

GENERATION_RESULTS_PATH = (
    EVALUATION_DIR
    / "generation_results_with_metrics.json"
)

KB_DATASET_PATH = (
    EVALUATION_DIR
    / "kb_evaluation_dataset.json"
)

GENERATION_CASES_PATH = (
    EVALUATION_DIR
    / "generation_cases.json"
)

OUTPUT_PATH = (
    REPORT_DIR
    / "phase9_evaluation_report.json"
)


def load_json(path: Path):
    """Load JSON data."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_retrieval_summary(
    retrieval_results: dict,
) -> dict:
    """Build retrieval benchmark summary."""

    summary = {}

    for retriever_name in [
        "bm25",
        "dense",
        "hybrid",
        "hybrid_reranker",
    ]:

        result = retrieval_results.get(
            retriever_name
        )

        if result is None:
            continue

        summary[retriever_name] = {
            "queries": result.get(
                "queries",
                0,
            ),
            "recall@5": result.get(
                "recall@5",
                0.0,
            ),
            "recall@10": result.get(
                "recall@10",
                0.0,
            ),
            "precision@5": result.get(
                "precision@5",
                0.0,
            ),
            "precision@10": result.get(
                "precision@10",
                0.0,
            ),
            "mrr": result.get(
                "mrr",
                0.0,
            ),
            "ndcg@5": result.get(
                "ndcg@5",
                0.0,
            ),
            "ndcg@10": result.get(
                "ndcg@10",
                0.0,
            ),
        }

    return summary


def build_generation_summary(
    generation_results: list[dict],
) -> dict:
    """Build generation benchmark summary."""

    successful = [
        result
        for result in generation_results
        if "metrics" in result
    ]

    if not successful:
        return {
            "evaluation_cases": 0,
        }

    metric_names = [
        "faithfulness",
        "answer_relevance",
        "context_relevance",
        "answer_correctness",
        "evidence_support",
    ]

    summary = {
        "evaluation_cases": len(
            successful
        )
    }

    for metric in metric_names:

        values = [
            result["metrics"][metric]
            for result in successful
            if metric in result["metrics"]
        ]

        summary[metric] = (
            sum(values) / len(values)
            if values
            else 0.0
        )

    return summary


def main():

    print("=" * 100)
    print("RESOLVEX PHASE 9 EVALUATION REPORT")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Load evaluation artifacts
    # ------------------------------------------------------------------

    retrieval_results = load_json(
        RETRIEVAL_RESULTS_PATH
    )

    generation_results = load_json(
        GENERATION_RESULTS_PATH
    )

    kb_dataset = load_json(
        KB_DATASET_PATH
    )

    generation_cases = load_json(
        GENERATION_CASES_PATH
    )

    # ------------------------------------------------------------------
    # Build summaries
    # ------------------------------------------------------------------

    retrieval_summary = (
        build_retrieval_summary(
            retrieval_results
        )
    )

    generation_summary = (
        build_generation_summary(
            generation_results
        )
    )

    # BM25, dense, hybrid and reranker are
    # all evaluated over the same query set.
    retrieval_query_count = 0

    if retrieval_summary:

        first_retriever = next(
            iter(
                retrieval_summary.values()
            )
        )

        retrieval_query_count = first_retriever.get(
            "queries",
            0,
        )

    # ------------------------------------------------------------------
    # Build final report
    # ------------------------------------------------------------------

    report = {
        "report": {
            "name": (
                "ResolveX Phase 9 "
                "Evaluation & Benchmarking"
            ),
            "phase": 9,
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },

        "datasets": {
            "knowledge_base_documents": len(
                kb_dataset
            ),
            "retrieval_evaluation_queries": (
                retrieval_query_count
            ),
            "generation_evaluation_cases": len(
                generation_cases
            ),
        },

        "retrieval_evaluation": {
            "metrics": [
                "Recall@5",
                "Recall@10",
                "Precision@5",
                "Precision@10",
                "MRR",
                "nDCG@5",
                "nDCG@10",
            ],
            "results": retrieval_summary,
        },

        "generation_evaluation": {
            "metrics": [
                "Faithfulness",
                "Answer Relevance",
                "Context Relevance",
                "Answer Correctness",
                "Evidence Support",
            ],
            "results": generation_summary,
        },

        "methodology": {
            "retrieval": (
                "Retrieval was evaluated against "
                "a manually constructed 30-query "
                "benchmark using Recall, Precision, "
                "MRR and nDCG."
            ),
            "generation": (
                "Generation was evaluated using "
                "10 manually curated ticket/reference "
                "answer cases."
            ),
            "semantic_model": (
                "all-MiniLM-L6-v2 was used for "
                "semantic generation metrics."
            ),
            "evidence_support": (
                "Evidence support evaluates whether "
                "required evaluation evidence is "
                "represented by retrieved documents "
                "or the generated answer."
            ),
        },

        "limitations": [
            (
                "The retrieval benchmark currently "
                "contains 30 manually constructed queries."
            ),
            (
                "The generation benchmark currently "
                "contains 10 manually curated cases."
            ),
            (
                "Generation metrics are internal "
                "semantic baselines and are not "
                "human evaluation scores."
            ),
            (
                "Faithfulness and context relevance "
                "are estimated using embedding "
                "similarity and do not constitute "
                "formal factual verification."
            ),
            (
                "The benchmark is based on the current "
                "ResolveX knowledge base."
            ),
        ],
    }

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------

    print()
    print(
        f"Knowledge-base documents: "
        f"{len(kb_dataset)}"
    )

    print(
        f"Retrieval queries: "
        f"{retrieval_query_count}"
    )

    print(
        f"Generation cases: "
        f"{len(generation_cases)}"
    )

    print()
    print("Retrieval evaluation:")

    for name, metrics in retrieval_summary.items():

        print(
            f"\n{name}:"
        )

        print(
            f"  Recall@5:     "
            f"{metrics['recall@5']:.4f}"
        )

        print(
            f"  Recall@10:    "
            f"{metrics['recall@10']:.4f}"
        )

        print(
            f"  MRR:          "
            f"{metrics['mrr']:.4f}"
        )

        print(
            f"  Precision@5:  "
            f"{metrics['precision@5']:.4f}"
        )

        print(
            f"  Precision@10: "
            f"{metrics['precision@10']:.4f}"
        )

        print(
            f"  nDCG@5:       "
            f"{metrics['ndcg@5']:.4f}"
        )

        print(
            f"  nDCG@10:      "
            f"{metrics['ndcg@10']:.4f}"
        )

    print()
    print("Generation evaluation:")

    for metric, value in (
        generation_summary.items()
    ):

        if metric == "evaluation_cases":
            continue

        print(
            f"  {metric:<25} "
            f"{value:.4f}"
        )

    print()
    print(
        "Report saved to:"
    )

    print(OUTPUT_PATH)

    print("=" * 100)


if __name__ == "__main__":
    main()