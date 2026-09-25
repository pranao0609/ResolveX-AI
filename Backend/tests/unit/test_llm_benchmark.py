from pathlib import Path

from evaluation.generation.llm_benchmark import LLMBenchmark


def test_benchmark_loads_fixed_dataset():
    benchmark = LLMBenchmark()

    cases = benchmark.load_dataset()

    assert len(cases) == 10


def test_benchmark_dataset_has_required_fields():
    benchmark = LLMBenchmark()

    cases = benchmark.load_dataset()

    required_fields = {
        "case_id",
        "ticket",
        "expected_category",
        "expected_context",
        "expected_resolution",
        "expected_escalation",
    }

    for case in cases:
        assert required_fields.issubset(case.keys())


def test_benchmark_case_ids_are_unique():
    benchmark = LLMBenchmark()

    cases = benchmark.load_dataset()

    case_ids = [
        case["case_id"]
        for case in cases
    ]

    assert len(case_ids) == len(set(case_ids))


def test_benchmark_save_results(tmp_path: Path):
    benchmark = LLMBenchmark()

    output_path = (
        tmp_path
        / "llm_results.json"
    )

    results = [
        {
            "case_id": "gen_001",
            "metrics": {
                "correctness": 1.0,
            },
        }
    ]

    benchmark.save_results(
        results,
        output_path,
    )

    assert output_path.exists()