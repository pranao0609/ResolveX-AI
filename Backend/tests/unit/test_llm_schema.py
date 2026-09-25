import pytest
from pydantic import ValidationError

from ai.llm.schemas import ResolutionResult


def test_valid_resolution_result():
    result = ResolutionResult(
        diagnosis="Password reset email is not being delivered.",
        root_cause="The reset email service is delayed.",
        resolution_steps=[
            "Check the user's email address.",
            "Check the spam or junk folder.",
            "Retry the password reset request.",
        ],
        evidence=[
            "Password Reset Email Delivery",
        ],
        confidence=0.92,
        requires_human=False,
    )

    assert result.diagnosis
    assert result.root_cause
    assert len(result.resolution_steps) == 3
    assert result.confidence == 0.92
    assert result.requires_human is False


def test_confidence_cannot_exceed_one():
    with pytest.raises(ValidationError):
        ResolutionResult(
            diagnosis="Test diagnosis",
            root_cause="Test cause",
            resolution_steps=["Test step"],
            confidence=1.1,
            requires_human=False,
        )


def test_confidence_cannot_be_negative():
    with pytest.raises(ValidationError):
        ResolutionResult(
            diagnosis="Test diagnosis",
            root_cause="Test cause",
            resolution_steps=["Test step"],
            confidence=-0.1,
            requires_human=False,
        )


def test_resolution_steps_cannot_be_empty():
    with pytest.raises(ValidationError):
        ResolutionResult(
            diagnosis="Test diagnosis",
            root_cause="Test cause",
            resolution_steps=[],
            confidence=0.8,
            requires_human=False,
        )


def test_diagnosis_cannot_be_empty():
    with pytest.raises(ValidationError):
        ResolutionResult(
            diagnosis="",
            root_cause="Test cause",
            resolution_steps=["Test step"],
            confidence=0.8,
            requires_human=False,
        )