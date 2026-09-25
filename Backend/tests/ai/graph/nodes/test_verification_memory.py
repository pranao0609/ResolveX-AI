from __future__ import annotations

from unittest.mock import patch

from ai.graph.nodes.verification import verification_agent


def _make_mock_verification(
    passed: bool = True,
    confidence: float = 0.92,
):
    return type(
        "VerificationResult",
        (),
        {
            "verification_passed": passed,
            "verification_reason": (
                "Resolution passed all verification checks."
                if passed
                else "Resolution failed verification checks."
            ),
            "confidence": confidence,
            "supported_by_evidence": passed,
            "hallucination_detected": not passed,
            "complete": passed,
            "policy_compliant": passed,
            "resolution_correct": passed,
            "model_dump": lambda self: {
                "supported_by_evidence": passed,
                "hallucination_detected": not passed,
                "complete": passed,
                "policy_compliant": passed,
                "resolution_correct": passed,
                "confidence": confidence,
                "verification_passed": passed,
                "verification_reason": (
                    "Resolution passed all verification checks."
                    if passed
                    else "Resolution failed verification checks."
                ),
            },
        },
    )()


def test_verification_uses_conversation_memory():
    mock_result = _make_mock_verification(passed=True, confidence=0.95)

    state = {
        "ticket_id": 601,
        "cleaned_ticket": "Unable to access email",
        "diagnosis": "Email authentication failure",
        "root_cause": "Invalid credentials",
        "resolution_steps": ["Reset email password."],
        "retrieved_context": "Users must use valid credentials.",
        "conversation_history": [
            {
                "role": "user",
                "content": "Unable to access email",
            },
            {
                "role": "assistant",
                "content": "Email authentication failure",
            },
        ],
        "previous_tickets": [],
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ) as mock_verify:
        result = verification_agent(state)

    mock_verify.assert_called_once()
    kwargs = mock_verify.call_args.kwargs

    assert "Conversation Context" in kwargs["context"]
    assert "Unable to access email" in kwargs["context"]
    assert result["verification_passed"] is True
    assert result["metadata"]["memory"]["conversation_memory_count"] == 2


def test_verification_uses_historical_memory():
    mock_result = _make_mock_verification(passed=True, confidence=0.91)

    state = {
        "ticket_id": 602,
        "cleaned_ticket": "VPN connection failed",
        "diagnosis": "VPN authentication error",
        "root_cause": "Expired credentials",
        "resolution_steps": ["Reset VPN credentials."],
        "retrieved_context": "VPN requires valid credentials.",
        "conversation_history": [],
        "previous_tickets": [
            {
                "ticket_id": 401,
                "title": "VPN login failure",
                "description": "VPN failed to connect.",
                "solution": "Reset VPN credentials.",
                "category": "network",
                "confidence": 0.95,
            }
        ],
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ) as mock_verify:
        result = verification_agent(state)

    kwargs = mock_verify.call_args.kwargs

    assert "Historical Troubleshooting Evidence" in kwargs["context"]
    assert "Reset VPN credentials." in kwargs["context"]
    assert result["verification_passed"] is True
    assert result["metadata"]["memory"]["historical_ticket_count"] == 1
    assert result["metadata"]["memory"]["historical_memory_used"] is True


def test_verification_combines_rag_and_historical_evidence():
    mock_result = _make_mock_verification(passed=True, confidence=0.88)

    state = {
        "ticket_id": 603,
        "cleaned_ticket": "Printer offline",
        "diagnosis": "Printer connectivity issue",
        "root_cause": "Disconnected network cable",
        "resolution_steps": ["Reconnect network cable."],
        "retrieved_context": "RAG evidence: Check network cable.",
        "conversation_history": [],
        "previous_tickets": [
            {
                "ticket_id": 302,
                "title": "Printer issue",
                "solution": "Plug in network cable.",
            }
        ],
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ) as mock_verify:
        verification_agent(state)

    kwargs = mock_verify.call_args.kwargs

    assert "Current Retrieved Evidence" in kwargs["context"]
    assert "RAG evidence: Check network cable." in kwargs["context"]
    assert "Historical Troubleshooting Evidence" in kwargs["context"]
    assert "Plug in network cable." in kwargs["context"]


def test_verification_without_memory_preserves_existing_behavior():
    mock_result = _make_mock_verification(passed=True, confidence=0.90)

    state = {
        "ticket_id": 604,
        "cleaned_ticket": "Software crash",
        "diagnosis": "Runtime exception",
        "root_cause": "Memory leak",
        "resolution_steps": ["Apply software patch."],
        "retrieved_context": "Apply patch v1.2.",
        "conversation_history": [],
        "previous_tickets": [],
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ) as mock_verify:
        result = verification_agent(state)

    kwargs = mock_verify.call_args.kwargs

    assert kwargs["context"] == "Apply patch v1.2."
    assert result["verification_passed"] is True
    assert result["metadata"]["memory"]["historical_ticket_count"] == 0
    assert result["metadata"]["memory"]["historical_memory_used"] is False
    assert result["metadata"]["memory"]["conversation_memory_count"] == 0


def test_verification_memory_metadata_exists():
    mock_result = _make_mock_verification(passed=True, confidence=0.85)

    state = {
        "ticket_id": 605,
        "cleaned_ticket": "App error",
        "diagnosis": "App issue",
        "root_cause": "Config error",
        "resolution_steps": ["Update config."],
        "retrieved_context": "Config settings.",
        "conversation_history": [{"role": "user", "content": "Help"}],
        "previous_tickets": [{"ticket_id": 1, "title": "Config issue"}],
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ):
        result = verification_agent(state)

    memory_metadata = result["metadata"]["memory"]

    assert memory_metadata["historical_ticket_count"] == 1
    assert memory_metadata["historical_memory_used"] is True
    assert memory_metadata["conversation_memory_count"] == 1


def test_historical_memory_does_not_automatically_pass_verification():
    # Even if historical tickets exist, if verify_resolution determines verification fails, it MUST fail.
    mock_failed_result = _make_mock_verification(passed=False, confidence=0.40)

    state = {
        "ticket_id": 606,
        "cleaned_ticket": "Severe security issue",
        "diagnosis": "Compromised credential",
        "root_cause": "Leaked password",
        "resolution_steps": ["Do nothing."],
        "retrieved_context": "Security policy requires credential revocation.",
        "conversation_history": [],
        "previous_tickets": [
            {
                "ticket_id": 99,
                "title": "Previous security issue",
                "solution": "Revoke credentials.",
            }
        ],
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_failed_result, False),
    ):
        result = verification_agent(state)

    assert result["verification_passed"] is False
    assert result["verification_confidence"] == 0.40


def test_verification_output_contract_remains_intact():
    mock_result = _make_mock_verification(passed=True, confidence=0.92)

    state = {
        "ticket_id": 607,
        "cleaned_ticket": "Issue",
        "diagnosis": "Diag",
        "root_cause": "Cause",
        "resolution_steps": ["Step 1"],
        "retrieved_context": "Context",
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        return_value=(mock_result, False),
    ):
        result = verification_agent(state)

    assert "verification_passed" in result
    assert "verification_reason" in result
    assert "verification_confidence" in result
    assert "verification_supported_by_evidence" in result
    assert "verification_hallucination_detected" in result
    assert "verification_complete" in result
    assert "verification_policy_compliant" in result
    assert "verification_resolution_correct" in result
    assert "verification_result" in result
    assert "fallback_used" in result
    assert "metadata" in result


def test_verification_fallback_remains_intact():
    state = {
        "ticket_id": 608,
        "cleaned_ticket": "Issue",
        "diagnosis": "Diag",
        "root_cause": "Cause",
        "resolution_steps": ["Step 1"],
        "retrieved_context": "Context",
        "fallback_used": False,
    }

    with patch(
        "ai.graph.nodes.verification.verify_resolution",
        side_effect=RuntimeError("LLM service unavailable"),
    ):
        result = verification_agent(state)

    assert result["verification_passed"] is False
    assert result["verification_confidence"] == 0.0
    assert result["verification_supported_by_evidence"] is False
    assert result["verification_hallucination_detected"] is True
    assert result["fallback_used"] is True
    assert "errors" in result
