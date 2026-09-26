from types import SimpleNamespace
from unittest.mock import patch


def _make_ticket():
    return SimpleNamespace(
        id=101,
        title="VPN connection failure",
        description=("The VPN disconnects immediately after login."),
        attachment_paths=None,
    )


def _mock_graph_state(
    decision="auto_resolve",
    requires_human=False,
):
    return {
        "ticket_id": 101,
        "cleaned_ticket": (
            "VPN connection failure. " "The VPN disconnects immediately after login."
        ),
        "category": "network",
        "category_confidence": 0.95,
        "retrieved_context": "VPN troubleshooting documentation",
        "retrieved_documents": [
            {
                "title": "VPN Troubleshooting",
                "content": "Restart the VPN client.",
                "source": "knowledge_base",
                "score": 0.91,
                "index_id": 1,
                "retriever": "dense",
            }
        ],
        "retrieval_metadata": {
            "strategy": "hybrid",
            "result_count": 1,
        },
        "diagnosis": "VPN client authentication/session failure.",
        "root_cause": "Expired or invalid VPN session state.",
        "diagnosis_confidence": 0.90,
        "resolution_steps": [
            "Restart the VPN client.",
            "Clear the existing VPN session.",
            "Authenticate again.",
        ],
        "evidence": ["VPN troubleshooting documentation."],
        "resolution_confidence": 0.90,
        "verification_passed": True,
        "verification_reason": "Resolution is supported by evidence.",
        "verification_confidence": 0.92,
        "decision": decision,
        "requires_human": requires_human,
        "escalation_reason": "",
        "fallback_used": False,
        "errors": [],
        "warnings": [],
        "request_id": "request-test",
        "graph_run_id": "graph-test",
        "metadata": {
            "last_stage": decision,
        },
    }


def test_graph_executor_invokes_langgraph():
    from ai.graph.executor import execute_resolvex_graph

    ticket = _make_ticket()

    with patch("ai.graph.executor.resolvex_graph.invoke") as mock_invoke:

        mock_invoke.return_value = _mock_graph_state()

        result = execute_resolvex_graph(ticket)

    mock_invoke.assert_called_once()

    assert result["decision"] == "auto_resolve"
    assert result["auto_resolved"] is True
    assert result["escalated_to_human"] is False
    assert result["category"] == "network"
    assert result["confidence"] > 0.0
    assert result["graph_run_id"] == "graph-test"


def test_graph_executor_maps_human_review():
    from ai.graph.executor import execute_resolvex_graph

    ticket = _make_ticket()

    with patch("ai.graph.executor.resolvex_graph.invoke") as mock_invoke:

        mock_invoke.return_value = _mock_graph_state(
            decision="human_review",
            requires_human=True,
        )

        result = execute_resolvex_graph(ticket)

    assert result["decision"] == "human_review"
    assert result["auto_resolved"] is False
    assert result["escalated_to_human"] is True


def test_graph_executor_maps_escalation():
    from ai.graph.executor import execute_resolvex_graph

    ticket = _make_ticket()

    with patch("ai.graph.executor.resolvex_graph.invoke") as mock_invoke:

        mock_invoke.return_value = _mock_graph_state(
            decision="escalate",
            requires_human=True,
        )

        result = execute_resolvex_graph(ticket)

    assert result["decision"] == "escalate"
    assert result["auto_resolved"] is False
    assert result["escalated_to_human"] is True


def test_graph_executor_preserves_fallback():
    from ai.graph.executor import execute_resolvex_graph

    ticket = _make_ticket()

    state = _mock_graph_state()
    state["fallback_used"] = True

    with patch("ai.graph.executor.resolvex_graph.invoke") as mock_invoke:

        mock_invoke.return_value = state

        result = execute_resolvex_graph(ticket)

    assert result["fallback_used"] is True
    assert result["auto_resolved"] is False
    assert result["escalated_to_human"] is True


def test_graph_executor_preserves_errors():
    from ai.graph.executor import execute_resolvex_graph

    ticket = _make_ticket()

    state = _mock_graph_state(
        decision="escalate",
        requires_human=True,
    )

    state["errors"] = ["retrieval_agent: RuntimeError: retrieval failed"]

    with patch("ai.graph.executor.resolvex_graph.invoke") as mock_invoke:

        mock_invoke.return_value = state

        result = execute_resolvex_graph(ticket)

    assert result["errors"]
    assert "retrieval_agent" in result["errors"][0]
    assert result["auto_resolved"] is False


def test_graph_executor_includes_evaluation_details():
    from ai.graph.executor import execute_resolvex_graph

    ticket = _make_ticket()

    with patch("ai.graph.executor.resolvex_graph.invoke") as mock_invoke:

        mock_invoke.return_value = _mock_graph_state()

        result = execute_resolvex_graph(
            ticket,
            include_evaluation_details=True,
        )

    assert "evaluation" in result

    evaluation = result["evaluation"]

    assert "cleaned_text" in evaluation
    assert "context_docs" in evaluation
    assert "retrieval_metadata" in evaluation
    assert "graph_run_id" in evaluation
