import pytest

from ai.llm.schemas import ResolutionResult
from evaluation.metrics.llm_metrics import (
    correctness,
    evaluate_llm_output,
    faithfulness,
    hallucination,
    instruction_adherence,
    relevance,
    structured_output_validity,
)


def test_correctness_identical_text():
    text = "Restart the application and try again."

    assert correctness(text, text) == 1.0


def test_correctness_unrelated_text():
    score = correctness(
        "Restart the application.",
        "The laptop battery is not charging.",
    )

    assert score == 0.0


def test_relevance():
    score = relevance(
        "The application crashes on startup.",
        "The application crashes on startup. Restart the system.",
    )

    assert score > 0.0


def test_faithfulness_supported_context():
    score = faithfulness(
        "Restart the application.",
        "Restart the application and check for updates.",
    )

    assert score > 0.0


def test_hallucination_is_inverse_of_faithfulness():
    generated = "Restart the application."
    context = "Restart the application and check for updates."

    assert hallucination(
        generated,
        context,
    ) == pytest.approx(
        1.0
        - faithfulness(
            generated,
            context,
        )
    )


def test_structured_output_validity():
    result = ResolutionResult(
        diagnosis="Application crash",
        root_cause="Startup failure",
        resolution_steps=[
            "Restart the application.",
        ],
        evidence=[
            "Crash on Startup",
        ],
        confidence=0.9,
        requires_human=False,
    )

    assert structured_output_validity(result) == 1.0


def test_structured_output_invalid():
    invalid_output = {
        "diagnosis": "",
        "root_cause": "",
        "resolution_steps": [],
        "confidence": 2.0,
    }

    assert structured_output_validity(invalid_output) == 0.0


def test_instruction_adherence():
    result = ResolutionResult(
        diagnosis="VPN connection failure",
        root_cause="VPN configuration issue",
        resolution_steps=[
            "Verify VPN configuration.",
            "Reconnect to VPN.",
        ],
        evidence=[
            "VPN Disconnecting",
        ],
        confidence=0.85,
        requires_human=True,
    )

    score = instruction_adherence(result)

    assert score == 1.0


def test_instruction_adherence_accepts_dict():
    output = {
        "diagnosis": "User cannot access email",
        "root_cause": "Invalid credentials",
        "resolution_steps": [
            "Verify the username",
            "Reset the password",
        ],
        "evidence": [
            "Password reset procedure",
        ],
        "confidence": 0.9,
        "requires_human": False,
    }

    score = instruction_adherence(output)

    assert score == 1.0


def test_evaluate_llm_output():
    result = ResolutionResult(
        diagnosis="Application crash",
        root_cause="Startup failure",
        resolution_steps=[
            "Restart the application.",
        ],
        evidence=[
            "Crash on Startup",
        ],
        confidence=0.9,
        requires_human=False,
    )

    metrics = evaluate_llm_output(
        ticket="The application crashes on startup.",
        generated_answer=(
            "The application crashes on startup. " "Restart the application."
        ),
        expected_resolution=(
            "The application crashes on startup. " "Restart the application."
        ),
        retrieved_context=("Crash on Startup: " "Restart the application."),
        structured_output=result,
    )

    assert set(metrics) == {
        "correctness",
        "faithfulness",
        "relevance",
        "hallucination",
        "instruction_adherence",
        "structured_output_validity",
    }

    assert metrics["correctness"] > 0.0
    assert metrics["faithfulness"] > 0.0
    assert metrics["relevance"] > 0.0
    assert metrics["instruction_adherence"] == 1.0
    assert metrics["structured_output_validity"] == 1.0
