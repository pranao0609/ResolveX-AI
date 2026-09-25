from unittest.mock import patch

from ai.graph.nodes import resolution_agent


def test_resolution_agent_returns_structured_resolution():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "diagnosis": "Email authentication failure",
            "root_cause": "Invalid credentials",
            "resolution_steps": [
                "Verify the account credentials.",
                "Retry authentication.",
            ],
            "evidence": [
                "Users must use valid credentials."
            ],
            "confidence": 0.91,
            "requires_human": False,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, False),
    ) as mock_generate:
        result = resolution_agent(
            {
                "ticket_id": 201,
                "cleaned_ticket": "Unable to login to email",
                "retrieved_context": (
                    "Users must use valid credentials."
                ),
                "diagnosis": "Email authentication failure",
                "root_cause": "Invalid credentials",
                "category": "authentication",
                "conversation_history": "",
                "fallback_used": False,
            }
        )

    mock_generate.assert_called_once_with(
        ticket_text="Unable to login to email",
        context="Users must use valid credentials.",
        diagnosis="Email authentication failure",
        root_cause="Invalid credentials",
        classification="authentication",
        conversation_history="",
    )

    assert result["resolution_steps"] == [
        "Verify the account credentials.",
        "Retry authentication.",
    ]

    assert result["evidence"] == [
        "Users must use valid credentials."
    ]

    assert result["resolution_confidence"] == 0.91
    assert result["requires_human"] is False
    assert result["fallback_used"] is False


def test_resolution_agent_consumes_diagnosis_and_root_cause():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "diagnosis": "Email authentication failure",
            "root_cause": "Expired credentials",
            "resolution_steps": [
                "Update the expired credentials.",
                "Retry authentication.",
            ],
            "evidence": [
                "Credentials must remain valid."
            ],
            "confidence": 0.87,
            "requires_human": False,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, False),
    ) as mock_generate:
        resolution_agent(
            {
                "ticket_id": 202,
                "cleaned_ticket": "Cannot access email",
                "retrieved_context": (
                    "Credentials must remain valid."
                ),
                "diagnosis": "Email authentication failure",
                "root_cause": "Expired credentials",
                "category": "authentication",
                "conversation_history": (
                    "User reported that the issue started today."
                ),
                "fallback_used": False,
            }
        )

    mock_generate.assert_called_once_with(
        ticket_text="Cannot access email",
        context="Credentials must remain valid.",
        diagnosis="Email authentication failure",
        root_cause="Expired credentials",
        classification="authentication",
        conversation_history=(
            "User reported that the issue started today."
        ),
    )


def test_resolution_agent_preserves_resolution_evidence():
    evidence = [
        "Users must use valid credentials.",
        "Authentication failures require credential verification.",
    ]

    mock_result = type(
        "ResolutionResult",
        (),
        {
            "diagnosis": "Authentication failure",
            "root_cause": "Invalid credentials",
            "resolution_steps": [
                "Verify credentials.",
                "Retry authentication.",
            ],
            "evidence": evidence,
            "confidence": 0.89,
            "requires_human": False,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, False),
    ):
        result = resolution_agent(
            {
                "ticket_id": 203,
                "cleaned_ticket": "Login failure",
                "retrieved_context": "\n".join(evidence),
                "diagnosis": "Authentication failure",
                "root_cause": "Invalid credentials",
                "category": "authentication",
                "fallback_used": False,
            }
        )

    assert result["evidence"] == evidence
    assert len(result["evidence"]) == 2


def test_resolution_agent_preserves_confidence():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "diagnosis": "Email authentication failure",
            "root_cause": "Invalid credentials",
            "resolution_steps": [
                "Verify credentials."
            ],
            "evidence": [
                "Authentication failed."
            ],
            "confidence": 0.74,
            "requires_human": False,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, False),
    ):
        result = resolution_agent(
            {
                "ticket_id": 204,
                "cleaned_ticket": "Cannot authenticate",
                "retrieved_context": "Authentication failed.",
                "diagnosis": "Email authentication failure",
                "root_cause": "Invalid credentials",
                "category": "authentication",
                "fallback_used": False,
            }
        )

    assert result["resolution_confidence"] == 0.74
    assert 0.0 <= result["resolution_confidence"] <= 1.0


def test_resolution_agent_requires_human_when_requested():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "diagnosis": "System failure",
            "root_cause": "Unknown",
            "resolution_steps": [
                "Collect system logs.",
                "Escalate to human support.",
            ],
            "evidence": [],
            "confidence": 0.35,
            "requires_human": True,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, False),
    ):
        result = resolution_agent(
            {
                "ticket_id": 205,
                "cleaned_ticket": "System failure",
                "retrieved_context": "",
                "diagnosis": "System failure",
                "root_cause": "Unknown",
                "category": "system",
                "fallback_used": False,
            }
        )

    assert result["requires_human"] is True
    assert result["resolution_confidence"] == 0.35


def test_resolution_agent_handles_fallback():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "diagnosis": (
                "The reported support issue requires "
                "further investigation."
            ),
            "root_cause": (
                "No validated root cause could be established."
            ),
            "resolution_steps": [
                "Review the ticket details.",
                "Collect relevant logs.",
                "Escalate the ticket.",
            ],
            "evidence": [],
            "confidence": 0.0,
            "requires_human": True,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, True),
    ):
        result = resolution_agent(
            {
                "ticket_id": 206,
                "cleaned_ticket": "System issue",
                "retrieved_context": "",
                "diagnosis": "",
                "root_cause": "",
                "category": "system",
                "fallback_used": False,
            }
        )

    assert result["resolution_confidence"] == 0.0
    assert result["requires_human"] is True
    assert result["fallback_used"] is True


def test_resolution_agent_preserves_previous_fallback():
    mock_result = type(
        "ResolutionResult",
        (),
        {
            "diagnosis": "Email issue",
            "root_cause": "Unknown",
            "resolution_steps": [
                "Review the ticket."
            ],
            "evidence": [],
            "confidence": 0.75,
            "requires_human": False,
        },
    )()

    with patch(
        "ai.graph.nodes.resolution.generate_solution",
        return_value=(mock_result, False),
    ):
        result = resolution_agent(
            {
                "ticket_id": 207,
                "cleaned_ticket": "Email issue",
                "retrieved_context": "Email KB",
                "diagnosis": "Email issue",
                "root_cause": "Unknown",
                "category": "email",
                "fallback_used": True,
            }
        )

    assert result["fallback_used"] is True