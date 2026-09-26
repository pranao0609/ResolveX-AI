"""
test_performance_profiler.py — Unit tests for Phase 25 Performance Profiling Engine.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai.evaluation.performance_profiler import (
    PerformanceProfilerEngine,
    compute_percentiles,
)


def test_timing_aggregation_percentiles():
    """Verify compute_percentiles calculation."""
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    res = compute_percentiles(vals)
    assert res["mean"] == 30.0
    assert res["p50"] == 30.0
    assert res["p95"] >= 40.0


def test_missing_timing_values():
    """Verify handling of missing/empty timing values."""
    res = compute_percentiles([])
    assert res["mean"] == 0.0
    assert res["p50"] == 0.0
    assert res["p95"] == 0.0
    assert res["p99"] == 0.0


def test_provider_rate_limit_handling():
    """Verify rate limit handling does not crash profiling engine."""
    engine = PerformanceProfilerEngine()
    dataset = [
        {
            "case_id": "bench_001",
            "title": "Test Title",
            "description": "Test Description",
            "expected_category": "software",
        }
    ]

    with patch(
        "ai.evaluation.performance_profiler.execute_resolvex_graph"
    ) as mock_exec:
        mock_exec.return_value = {
            "errors": ["LLMRateLimitError: Error code 429 rate limit reached"],
            "fallback_used": True,
            "decision": "ask_clarification",
            "graph_metadata": {},
        }
        res = engine.profile_e2e_workflow(dataset)
        assert res["provider_rate_limited"] is True
        assert len(res["component_profiles"]["total_workflow"]) > 0


def test_no_secret_leakage_in_reports(tmp_path):
    """Verify generated JSON and Markdown reports contain zero credentials or secrets."""
    dataset_path = tmp_path / "mock_dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    engine = PerformanceProfilerEngine(dataset_path=dataset_path)

    tmp_json = tmp_path / "resolveX_performance_profile.json"
    tmp_md = tmp_path / "resolveX_performance_profile.md"

    sample_sub = {
        "bm25": {"mean": 2.5, "p50": 2.0, "p95": 4.0, "p99": 5.0},
        "faiss": {"mean": 12.1, "p50": 10.0, "p95": 20.0, "p99": 25.0},
        "hybrid_merge": {"mean": 1.2, "p50": 1.0, "p95": 2.0, "p99": 2.5},
        "reranker": {"mean": 45.0, "p50": 40.0, "p95": 80.0, "p99": 90.0},
    }

    sample_wf = {
        "component_profiles": {
            "total_workflow": {
                "mean": 1200.0,
                "p50": 1100.0,
                "p95": 2000.0,
                "p99": 2200.0,
            },
            "ticket_analyzer": {"mean": 5.0, "p50": 4.0, "p95": 8.0, "p99": 10.0},
        },
        "provider_rate_limited": False,
        "llm_calls_recorded": 3,
    }

    baseline = {"p50": 8076.85, "p95": 17951.79, "p99": 19762.41, "mean": 9688.08}

    with (
        patch("ai.evaluation.performance_profiler.PERF_JSON_PATH", tmp_json),
        patch("ai.evaluation.performance_profiler.PERF_MD_PATH", tmp_md),
    ):

        json_file, md_file = engine.generate_performance_reports(
            retrieval_sub_profiles=sample_sub,
            workflow_profiles=sample_wf,
            baseline=baseline,
        )

        assert Path(json_file).exists()
        assert Path(md_file).exists()

        json_text = Path(json_file).read_text(encoding="utf-8")
        md_text = Path(md_file).read_text(encoding="utf-8")

        # Verify no sensitive keywords present
        for forbidden in [
            "GROQ_API_KEY",
            "LANGSMITH_API_KEY",
            "AWS_SECRET",
            "PASSWORD",
            "BEARER",
        ]:
            assert forbidden not in json_text
            assert forbidden not in md_text


def test_performance_report_generation(tmp_path):
    """Verify report generation structure."""
    dataset_path = tmp_path / "mock_dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    engine = PerformanceProfilerEngine(dataset_path=dataset_path)
    tmp_json = tmp_path / "perf.json"
    tmp_md = tmp_path / "perf.md"

    sample_sub = {
        "bm25": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0},
        "faiss": {"mean": 5.0, "p50": 5.0, "p95": 5.0, "p99": 5.0},
        "hybrid_merge": {"mean": 0.5, "p50": 0.5, "p95": 0.5, "p99": 0.5},
        "reranker": {"mean": 20.0, "p50": 20.0, "p95": 20.0, "p99": 20.0},
    }

    sample_wf = {
        "component_profiles": {
            "total_workflow": {"mean": 500.0, "p50": 500.0, "p95": 500.0, "p99": 500.0},
        },
        "provider_rate_limited": False,
        "llm_calls_recorded": 0,
    }

    with (
        patch("ai.evaluation.performance_profiler.PERF_JSON_PATH", tmp_json),
        patch("ai.evaluation.performance_profiler.PERF_MD_PATH", tmp_md),
    ):

        j_path, m_path = engine.generate_performance_reports(
            retrieval_sub_profiles=sample_sub,
            workflow_profiles=sample_wf,
            baseline={"p50": 1000.0},
        )

        with open(j_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "bottleneck" in data
            assert "resource_initialization_audit" in data
            assert "quality_baseline_preserved" in data
