from ai.graph.graph import resolvex_graph
from ai.graph.visualize import get_graph_mermaid


def test_graph_contains_all_phase_14_nodes():
    graph = resolvex_graph.get_graph()

    expected_nodes = {
        "initialize_state",
        "ticket_analyzer",
        "retrieval_agent",
        "diagnosis_agent",
        "resolution_agent",
        "verification_agent",
        "decision_agent",
        "auto_resolve",
        "human_review",
        "escalate",
    }

    actual_nodes = set(graph.nodes.keys())

    assert expected_nodes.issubset(
        actual_nodes
    )


def test_graph_contains_terminal_nodes():
    graph = resolvex_graph.get_graph()

    assert "auto_resolve" in graph.nodes
    assert "human_review" in graph.nodes
    assert "escalate" in graph.nodes


def test_graph_mermaid_generation():
    mermaid = get_graph_mermaid()

    assert isinstance(
        mermaid,
        str,
    )

    assert len(mermaid) > 0


def test_graph_mermaid_contains_core_agents():
    mermaid = get_graph_mermaid()

    expected_nodes = [
        "initialize_state",
        "ticket_analyzer",
        "retrieval_agent",
        "diagnosis_agent",
        "resolution_agent",
        "verification_agent",
        "decision_agent",
    ]

    for node in expected_nodes:
        assert node in mermaid


def test_graph_mermaid_contains_decision_paths():
    mermaid = get_graph_mermaid()

    assert "auto_resolve" in mermaid
    assert "human_review" in mermaid
    assert "escalate" in mermaid