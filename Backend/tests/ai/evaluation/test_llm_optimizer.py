"""
test_llm_optimizer.py — Unit tests for Phase 26 LLM Efficiency Optimizer.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ai.evaluation.llm_optimizer import LLMOptimizerEngine
from ai.llm.prompt_loader import load_prompt, render_prompt


def test_v1_and_v2_prompts_exist_and_load():
    """Verify both v1 and v2 prompt files load cleanly for diagnosis, resolution, and verification."""
    for agent in ["diagnosis", "resolution", "verification"]:
        p_v1 = load_prompt(agent, 1)
        p_v2 = load_prompt(agent, 2)

        assert p_v1["version"] == 1
        assert p_v2["version"] == 2
        assert "system_prompt" in p_v2
        assert "user_prompt" in p_v2


def test_prompt_rendering_v2_reduces_size():
    """Verify v2 prompt rendering produces smaller prompt text while retaining required schema."""
    ticket_text = "VPN error 800 tunnel failure"
    context = "Standard VPN procedure document"

    p_v1 = load_prompt("diagnosis", 1)
    p_v2 = load_prompt("diagnosis", 2)

    sys1, user1 = render_prompt(p_v1, ticket_text, context)
    sys2, user2 = render_prompt(p_v2, ticket_text, context)

    assert len(sys2) + len(user2) < len(sys1) + len(user1)
    assert "problem" in user2
    assert "possible_root_cause" in user2
    assert "evidence" in user2


def test_llm_optimizer_report_generation(tmp_path):
    """Verify JSON and Markdown report generation for Phase 26."""
    dataset_path = tmp_path / "mock_dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")

    engine = LLMOptimizerEngine(dataset_path=dataset_path)

    tmp_json = tmp_path / "resolveX_llm_optimization.json"
    tmp_md = tmp_path / "resolveX_llm_optimization.md"

    sample_analysis = {
        "diagnosis": {
            "v1_est_tokens": 500,
            "v2_est_tokens": 350,
            "saved_est_tokens": 150,
            "pct_prompt_reduction": 30.0,
        },
        "resolution": {
            "v1_est_tokens": 400,
            "v2_est_tokens": 300,
            "saved_est_tokens": 100,
            "pct_prompt_reduction": 25.0,
        },
        "verification": {
            "v1_est_tokens": 600,
            "v2_est_tokens": 400,
            "saved_est_tokens": 200,
            "pct_prompt_reduction": 33.33,
        },
    }

    baseline = {"p50": 8076.85, "p95": 17951.79, "p99": 19762.41, "mean": 9688.08}

    with (
        patch("ai.evaluation.llm_optimizer.OPT_JSON_PATH", tmp_json),
        patch("ai.evaluation.llm_optimizer.OPT_MD_PATH", tmp_md),
    ):

        json_file, md_file = engine.generate_optimization_report(
            sample_analysis, baseline
        )

        assert Path(json_file).exists()
        assert Path(md_file).exists()

        json_text = Path(json_file).read_text(encoding="utf-8")
        md_text = Path(md_file).read_text(encoding="utf-8")

        assert "ResolveX 2.0" in md_text
        assert "Unsafe Auto-Resolutions" in md_text
