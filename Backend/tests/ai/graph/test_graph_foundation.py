from ai.graph.graph import (
    build_resolvex_graph,
    route_decision,
)
from ai.graph.state import ResolveXState


def test_resolvex_state_accepts_ticket_input():
    state: ResolveXState = {
        "ticket_id": 1,
        "ticket_text": "unable to access email",
    }

    assert state["ticket_id"] == 1
    assert state["ticket_text"] == (
        "unable to access email"
    )


def test_graph_builds_successfully():
    graph = build_resolvex_graph()

    assert graph is not None


def test_decision_routes_to_auto_resolve():
    state: ResolveXState = {
        "decision": "auto_resolve",
    }

    assert route_decision(state) == "auto_resolve"


def test_decision_routes_to_escalation():
    state: ResolveXState = {
        "decision": "escalate",
    }

    assert route_decision(state) == "escalate"


def test_decision_defaults_to_human_review():
    state: ResolveXState = {}

    assert route_decision(state) == "human_review"


def test_decision_explicit_human_review():
    state: ResolveXState = {
        "decision": "human_review",
    }

    assert route_decision(state) == "human_review"


def test_graph_executes_basic_workflow():
    graph = build_resolvex_graph()

    state: ResolveXState = {
        "ticket_id": 1,
        "ticket_text": "unable to access email",
    }

    result = graph.invoke(state)

    assert result["cleaned_ticket"] == (
        "unable to access email"
    )

    assert result["decision"] in {
    "auto_resolve",
    "ask_clarification",
    "human_review",
    "escalate",
}
    if result["decision"] == "ask_clarification":
        assert result["requires_human"] is False
    else:
        assert result["requires_human"] is True

def test_ticket_analyzer_classifies_ticket():
    from ai.graph.graph import ticket_analyzer

    state = {
        "ticket_id": 1,
        "ticket_text": "My Outlook email is not working.",
    }

    result = ticket_analyzer(state)

    assert result["cleaned_ticket"]
    assert result["category"] == "software"
    assert result["category_confidence"] > 0

def test_ticket_analyzer_cleans_ticket_text():
    from ai.graph.graph import ticket_analyzer

    state = {
        "ticket_id": 2,
        "ticket_text": "<p>   VPN   is   NOT   working   </p>",
    }

    result = ticket_analyzer(state)

    assert result["cleaned_ticket"] == "vpn is not working"
    assert result["category"] == "network"

def test_ticket_analyzer_processes_attachments(monkeypatch):
    from ai.graph import graph

    monkeypatch.setattr(
        graph,
        "parse_attachments",
        lambda paths: "error screenshot: connection timeout",
    )

    state = {
        "ticket_id": 3,
        "ticket_text": "Internet is not working.",
        "attachment_paths": ["error.txt"],
    }

    result = graph.ticket_analyzer(state)

    assert "[Attachments]" in result["cleaned_ticket"]
    assert "connection timeout" in result["cleaned_ticket"]
    assert result["category"] == "network"

def test_graph_ticket_analyzer_populates_state():
    from ai.graph.graph import resolvex_graph

    state = {
        "ticket_id": 4,
        "ticket_text": "I cannot connect to the VPN.",
    }

    result = resolvex_graph.invoke(state)

    assert result["cleaned_ticket"]
    assert result["category"] == "network"
    assert result["category_confidence"] > 0

def test_retrieval_agent_handles_empty_query():
    from ai.graph.graph import retrieval_agent

    state = {
        "ticket_id": 1,
        "cleaned_ticket": "",
    }

    result = retrieval_agent(state)

    assert result["retrieved_context"] == ""
    assert result["retrieved_documents"] == []
    assert result["retrieval_metadata"]["result_count"] == 0

def test_retrieval_agent_returns_documents():
    from ai.graph.graph import retrieval_agent

    state = {
        "ticket_id": 2,
        "cleaned_ticket": "unable to access email account",
    }

    result = retrieval_agent(state)

    assert "retrieved_context" in result
    assert "retrieved_documents" in result
    assert "retrieval_metadata" in result

    assert isinstance(
        result["retrieved_documents"],
        list,
    )

    assert result["retrieval_metadata"]["strategy"]

def test_retrieval_agent_document_structure():
    from ai.graph.graph import retrieval_agent

    state = {
        "ticket_id": 3,
        "cleaned_ticket": "vpn connection is not working",
    }

    result = retrieval_agent(state)

    for document in result["retrieved_documents"]:
        assert "index_id" in document
        assert "score" in document
        assert "retriever" in document
        assert "content" in document

def test_retrieval_agent_metadata_matches_documents():
    from ai.graph.graph import retrieval_agent

    state = {
        "ticket_id": 4,
        "cleaned_ticket": "database connection failed",
    }

    result = retrieval_agent(state)

    documents = result["retrieved_documents"]
    metadata = result["retrieval_metadata"]

    assert metadata["result_count"] == len(documents)

    assert metadata["index_ids"] == [
        document["index_id"]
        for document in documents
    ]

def test_graph_retrieval_agent_populates_state():
    from ai.graph.graph import resolvex_graph

    state = {
        "ticket_id": 5,
        "ticket_text": "My email account is not working.",
    }

    result = resolvex_graph.invoke(state)

    assert result["cleaned_ticket"]
    assert result["category"] == "software"

    assert "retrieved_context" in result
    assert "retrieved_documents" in result
    assert "retrieval_metadata" in result