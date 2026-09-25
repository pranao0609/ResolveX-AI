import pytest
from pydantic import ValidationError

from ai.llm.schemas import DiagnosisResult


def test_diagnosis_result_valid():
    result = DiagnosisResult(
        problem="User cannot receive password reset email",
        possible_root_cause="Email delivery is being blocked or filtered",
        evidence=["Password Reset Email Delivery"],
        missing_information=["Whether spam was checked"],
        confidence=0.87,
    )

    assert result.problem == "User cannot receive password reset email"
    assert result.possible_root_cause == "Email delivery is being blocked or filtered"
    assert result.evidence == ["Password Reset Email Delivery"]
    assert result.missing_information == ["Whether spam was checked"]
    assert result.confidence == pytest.approx(0.87)


def test_diagnosis_result_defaults_lists():
    result = DiagnosisResult(
        problem="Email is not syncing",
        possible_root_cause="Client synchronization failure",
        confidence=0.75,
    )

    assert result.evidence == []
    assert result.missing_information == []


def test_diagnosis_result_rejects_empty_problem():
    with pytest.raises(ValidationError):
        DiagnosisResult(
            problem="",
            possible_root_cause="Email delivery failure",
            confidence=0.8,
        )


def test_diagnosis_result_rejects_empty_root_cause():
    with pytest.raises(ValidationError):
        DiagnosisResult(
            problem="Cannot reset password",
            possible_root_cause="",
            confidence=0.8,
        )


def test_diagnosis_result_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        DiagnosisResult(
            problem="Cannot log in",
            possible_root_cause="Authentication failure",
            confidence=1.5,
        )


def test_diagnosis_result_rejects_negative_confidence():
    with pytest.raises(ValidationError):
        DiagnosisResult(
            problem="Cannot log in",
            possible_root_cause="Authentication failure",
            confidence=-0.1,
        )