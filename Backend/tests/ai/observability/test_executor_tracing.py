"""
test_executor_tracing.py — Integration test verifying GraphExecutor uses observability config.
"""

from unittest.mock import MagicMock, patch
import pytest

from ai.graph.executor import execute_resolvex_graph


class DummyTicket:
    def __init__(
        self, ticket_id=101, title="Test Ticket", description="Description of issue"
    ):
        self.id = ticket_id
        self.title = title
        self.description = description


def test_execute_resolvex_graph_passes_tracing_config():
    ticket = DummyTicket(ticket_id=404)

    with patch("ai.graph.executor.resolvex_graph.invoke") as mock_invoke:
        mock_invoke.return_value = {
            "category": "technical",
            "category_confidence": 0.95,
            "diagnosis": "Server memory exhaustion",
            "root_cause": "Memory leak",
            "resolution_steps": ["Restart service"],
            "evidence": ["Log entry 404"],
            "diagnosis_confidence": 0.9,
            "resolution_confidence": 0.92,
            "verification_confidence": 0.95,
            "verification_passed": True,
            "requires_human": False,
            "fallback_used": False,
            "decision": "auto_resolve",
            "errors": [],
            "warnings": [],
            "retrieved_documents": [],
            "graph_run_id": "run-404",
        }

        result = execute_resolvex_graph(ticket)

        assert mock_invoke.called
        call_args, call_kwargs = mock_invoke.call_args
        assert "config" in call_kwargs
        config = call_kwargs["config"]
        assert config["run_name"] == "ResolveX Ticket 404"
        assert config["configurable"]["thread_id"] == "404"
        assert result["decision"] == "auto_resolve"
        assert result["auto_resolved"] is True
