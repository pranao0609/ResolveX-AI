from unittest.mock import patch

from ai.graph.nodes import diagnosis_agent
from ai.llm.schemas import DiagnosisResult


def test_diagnosis_agent_returns_structured_diagnosis():
    mock_result = DiagnosisResult(
        problem="Email authentication failure",
        possible_root_cause="Invalid credentials",
        evidence=[
            "Users must use valid credentials."
        ],
        missing_information=[],
        confidence=0.88,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 101,
                "cleaned_ticket": "Unable to login to email",
                "retrieved_context": (
                    "Users must use valid credentials."
                ),
            }
        )

    assert result["diagnosis_problem"] == (
        "Email authentication failure"
    )

    assert result["diagnosis_root_cause"] == (
        "Invalid credentials"
    )

    assert result["diagnosis_evidence"] == [
        "Users must use valid credentials."
    ]

    assert result["diagnosis_missing_information"] == []

    assert result["diagnosis_confidence"] == 0.88

    assert result["fallback_used"] is False


def test_diagnosis_separates_problem_from_root_cause():
    mock_result = DiagnosisResult(
        problem="User cannot access the email account",
        possible_root_cause=(
            "Authentication credentials are invalid"
        ),
        evidence=[
            "Authentication failed with the supplied credentials."
        ],
        missing_information=[],
        confidence=0.84,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 102,
                "cleaned_ticket": (
                    "I cannot access my email account."
                ),
                "retrieved_context": (
                    "Authentication failed with "
                    "the supplied credentials."
                ),
            }
        )

    assert result["diagnosis_problem"] != (
        result["diagnosis_root_cause"]
    )

    assert result["diagnosis_problem"] == (
        "User cannot access the email account"
    )

    assert result["diagnosis_root_cause"] == (
        "Authentication credentials are invalid"
    )


def test_diagnosis_preserves_supporting_evidence():
    evidence = [
        "Users must use valid credentials.",
        "Authentication failures occur when credentials are invalid.",
    ]

    mock_result = DiagnosisResult(
        problem="Email authentication failure",
        possible_root_cause="Invalid credentials",
        evidence=evidence,
        missing_information=[],
        confidence=0.91,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 103,
                "cleaned_ticket": "Email login fails",
                "retrieved_context": "\n".join(evidence),
            }
        )

    assert result["diagnosis_evidence"] == evidence
    assert len(result["diagnosis_evidence"]) == 2


def test_diagnosis_preserves_missing_information():
    missing_information = [
        "Authentication error logs are unavailable.",
        "The user has not confirmed whether the password was recently changed.",
    ]

    mock_result = DiagnosisResult(
        problem="Email authentication failure",
        possible_root_cause=(
            "Possibly invalid or outdated credentials"
        ),
        evidence=[
            "The user cannot authenticate to the email service."
        ],
        missing_information=missing_information,
        confidence=0.61,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 104,
                "cleaned_ticket": (
                    "Email authentication is failing."
                ),
                "retrieved_context": (
                    "The user cannot authenticate "
                    "to the email service."
                ),
            }
        )

    assert result["diagnosis_missing_information"] == (
        missing_information
    )

    assert len(
        result["diagnosis_missing_information"]
    ) == 2


def test_diagnosis_confidence_is_preserved():
    mock_result = DiagnosisResult(
        problem="Email authentication failure",
        possible_root_cause="Invalid credentials",
        evidence=[
            "Authentication failed."
        ],
        missing_information=[],
        confidence=0.73,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 105,
                "cleaned_ticket": "Cannot authenticate",
                "retrieved_context": (
                    "Authentication failed."
                ),
            }
        )

    assert result["diagnosis_confidence"] == 0.73
    assert 0.0 <= result["diagnosis_confidence"] <= 1.0


def test_diagnosis_fallback_is_marked():
    mock_result = DiagnosisResult(
        problem=(
            "The reported support issue requires "
            "further investigation."
        ),
        possible_root_cause=(
            "No validated root cause could be "
            "established from the available evidence."
        ),
        evidence=[],
        missing_information=[
            "Additional diagnostic information is required."
        ],
        confidence=0.0,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, True),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 106,
                "cleaned_ticket": "System issue",
                "retrieved_context": "",
            }
        )

    assert result["diagnosis_confidence"] == 0.0
    assert result["diagnosis_evidence"] == []
    assert result["fallback_used"] is True


def test_diagnosis_preserves_previous_fallback():
    mock_result = DiagnosisResult(
        problem="Email authentication failure",
        possible_root_cause="Invalid credentials",
        evidence=[
            "Authentication failed."
        ],
        missing_information=[],
        confidence=0.82,
    )

    with patch(
        "ai.graph.nodes.diagnosis.generate_diagnosis",
        return_value=(mock_result, False),
    ):
        result = diagnosis_agent(
            {
                "ticket_id": 107,
                "cleaned_ticket": "Cannot login",
                "retrieved_context": (
                    "Authentication failed."
                ),
                "fallback_used": True,
            }
        )

    assert result["fallback_used"] is True