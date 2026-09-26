"""
test_final_benchmark.py — Comprehensive unit & integration tests for Phase 24 Final Evaluation & Benchmarking.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from ai.evaluation.final_benchmark_engine import (
    BenchmarkTicket,
    FinalBenchmarkEngine,
    calculate_percentiles,
)
from evaluation.retrieval.metrics import recall_at_k, reciprocal_rank


def test_retrieval_metric_calculations():
    """Verify Recall@K and MRR calculations."""
    retrieved = [10, 20, 30, 40, 50]
    relevant = {30, 60}

    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 0.5  # 30 is in top 3
    assert recall_at_k(retrieved, relevant, 5) == 0.5
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3, 0.01)


def test_latency_aggregation_percentiles():
    """Verify percentile calculation for component latencies."""
    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    res = calculate_percentiles(vals)

    assert res["mean"] == pytest.approx(55.0, 0.1)
    assert res["p50"] == pytest.approx(55.0, 1.0)
    assert res["p95"] >= 90.0
    assert res["p99"] >= 95.0

    # Empty case
    empty_res = calculate_percentiles([])
    assert empty_res["p50"] == 0.0
    assert empty_res["mean"] == 0.0


def test_compute_summary_metrics_safety_and_routing():
    """Verify calculation of routing, safety, and verification summaries."""
    engine = FinalBenchmarkEngine()
    mock_case_results = [
        {
            "case_id": "bench_001",
            "decision": "auto_resolve",
            "verification_passed": True,
            "evidence": ["Doc A"],
            "is_unsafe_auto_resolve": False,
            "routing_correct": True,
            "fallback_used": False,
        },
        {
            "case_id": "bench_002",
            "decision": "ask_clarification",
            "verification_passed": False,
            "evidence": [],
            "is_unsafe_auto_resolve": False,
            "routing_correct": True,
            "fallback_used": False,
        },
        {
            "case_id": "bench_003",
            "decision": "auto_resolve",
            "verification_passed": True,
            "evidence": ["Doc B"],
            "is_unsafe_auto_resolve": True,  # Unsafe!
            "routing_correct": False,
            "fallback_used": True,
        },
    ]

    summaries = engine.compute_summary_metrics(mock_case_results)

    ver = summaries["verification"]
    pol = summaries["policy"]
    safe = summaries["safety"]

    assert ver["total_cases"] == 3
    assert ver["verification_pass_rate"] == pytest.approx(2 / 3, 0.01)
    assert ver["evidence_supported_rate"] == pytest.approx(2 / 3, 0.01)

    assert pol["auto_resolution_rate"] == pytest.approx(2 / 3, 0.01)
    assert pol["clarification_rate"] == pytest.approx(1 / 3, 0.01)
    assert pol["fallback_rate"] == pytest.approx(1 / 3, 0.01)

    assert safe["unsafe_auto_resolution_count"] == 1
    assert safe["safety_routing_accuracy"] == pytest.approx(2 / 3, 0.01)
    assert safe["fail_closed_rate"] == pytest.approx(1 / 3, 0.01)


def test_empty_case_results():
    """Verify compute_summary_metrics handles empty lists without crashing."""
    engine = FinalBenchmarkEngine()
    res = engine.compute_summary_metrics([])
    assert res == {}


def test_report_generation(tmp_path):
    """Verify JSON and Markdown report generation."""
    dataset_path = tmp_path / "mock_dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    engine = FinalBenchmarkEngine(dataset_path=dataset_path)

    sample_report = {
        "metadata": {"timestamp": "2026-09-26T12:00:00Z", "git_commit_sha": "abc1234"},
        "dataset": {
            "total_cases": 1,
            "ground_truth_cases": 1,
            "synthetic_cases": 0,
            "source_file": str(dataset_path),
        },
        "retrieval": {"hybrid": {"recall_at_5": 0.8}},
        "reranking": {"recall_at_5": {"retrieval": 0.8, "reranked": 0.9, "diff": 0.1}},
        "verification": {"verification_pass_rate": 0.85},
        "policy": {"auto_resolution_rate": 0.4},
        "safety": {"unsafe_auto_resolution_count": 0, "safety_routing_accuracy": 1.0},
        "end_to_end": {"e2e_success_rate": 1.0},
        "latency": {"total_workflow": {"p50": 120.0}},
        "errors": {"fallback_count": 0},
    }

    tmp_json = tmp_path / "resolveX_final_benchmark.json"
    tmp_md = tmp_path / "resolveX_final_benchmark.md"

    with (
        patch("ai.evaluation.final_benchmark_engine.JSON_REPORT_PATH", tmp_json),
        patch("ai.evaluation.final_benchmark_engine.MD_REPORT_PATH", tmp_md),
    ):

        json_path = Path(engine.generate_json_report(sample_report))
        md_path = Path(engine.generate_markdown_report(sample_report))

        assert json_path == tmp_json
        assert md_path == tmp_md
        assert json_path.exists()
        assert md_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
            assert loaded_json["metadata"]["git_commit_sha"] == "abc1234"

        md_content = md_path.read_text(encoding="utf-8")
        assert "ResolveX 2.0" in md_content
        assert "Unsafe Auto-Resolutions" in md_content


def test_mlflow_disabled_execution():
    """Verify benchmark runs without error when MLflow is disabled."""
    with patch(
        "ai.experiments.mlflow_tracker.MLflowTracker.is_enabled", return_value=False
    ):
        engine = FinalBenchmarkEngine()
        # Verify engine initializes and dataset loads without MLflow active
        dataset = engine.load_dataset()
        assert isinstance(dataset, list)
        assert len(dataset) > 0


def test_langsmith_disabled_execution():
    """Verify evaluation runs normally when LangSmith tracing is disabled."""
    with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}):
        engine = FinalBenchmarkEngine()
        ret_metrics = engine.evaluate_retrieval(engine.load_dataset())
        assert isinstance(ret_metrics, dict)
