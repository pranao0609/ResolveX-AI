from unittest.mock import patch

from ai.graph.graph import (
    retrieval_decision_agent,
    route_retrieval_decision,
)


def make_state(
    documents=None,
    retrieval_metadata=None,
    metadata=None,
):
    return {
        "ticket_id": "TEST-16-2",
        "request_id": "request-16-2",
        "graph_run_id": "graph-16-2",
        "cleaned_ticket": (
            "VPN connection fails after password reset"
        ),
        "retrieved_documents": (
            documents or []
        ),
        "retrieval_metadata": (
            retrieval_metadata or {}
        ),
        "metadata": (
            metadata or {}
        ),
        "errors": [],
        "warnings": [],
    }


def test_retrieval_decision_continues_with_sufficient_evidence():
    state = make_state(
        documents=[
            {"score": 0.90},
            {"score": 0.80},
            {"score": 0.70},
        ],
        retrieval_metadata={
            "status": "success",
            "top_score": 0.90,
            "score_gap": 0.10,
            "retrieval_query": "vpn password reset",
            "normalized_query": "vpn password reset",
            "original_query": "VPN connection fails after password reset",
        },
    )

    result = retrieval_decision_agent(
        state
    )

    metadata = result[
        "retrieval_metadata"
    ]

    assert metadata[
        "enough_evidence"
    ] is True

    assert metadata[
        "retrieval_decision"
    ] == "continue"

    assert metadata[
        "retry_allowed"
    ] is False


def test_retrieval_decision_retries_when_evidence_is_insufficient():
    state = make_state(
        documents=[
            {"score": 0.08},
        ],
        retrieval_metadata={
            "status": "success",
            "top_score": 0.08,
            "score_gap": 0.01,
            "retrieval_query": "vpn issue",
            "normalized_query": "vpn issue",
            "original_query": "VPN connection issue",
            "rewrite_needed": True,
        },
    )

    result = retrieval_decision_agent(
        state
    )

    metadata = result[
        "retrieval_metadata"
    ]

    assert metadata[
        "enough_evidence"
    ] is False

    assert metadata[
        "retrieval_decision"
    ] == "retry"

    assert metadata[
        "retry_allowed"
    ] is True

    assert metadata[
        "retry_strategy"
    ] in {
        "broaden",
        "narrow",
        "rewrite",
    }

    assert metadata[
        "next_query"
    ]


def test_retrieval_decision_stops_after_max_attempts():
    state = make_state(
        documents=[
            {"score": 0.05},
        ],
        retrieval_metadata={
            "status": "success",
            "top_score": 0.05,
            "score_gap": 0.01,
            "retrieval_query": "vpn issue",
            "normalized_query": "vpn issue",
            "original_query": "VPN connection issue",
        },
        metadata={
            "retrieval_attempt": 1,
        },
    )

    result = retrieval_decision_agent(
        state
    )

    metadata = result[
        "retrieval_metadata"
    ]

    assert metadata[
        "retrieval_attempt"
    ] == 2

    assert metadata[
        "retry_allowed"
    ] is False

    assert metadata[
        "retrieval_decision"
    ] == "continue"


def test_retrieval_decision_handles_empty_context():
    state = make_state(
        documents=[],
        retrieval_metadata={
            "status": "empty",
            "top_score": None,
            "score_gap": None,
            "retrieval_query": "vpn issue",
            "normalized_query": "vpn issue",
            "original_query": "VPN connection issue",
        },
    )

    result = retrieval_decision_agent(
        state
    )

    metadata = result[
        "retrieval_metadata"
    ]

    assert metadata[
        "enough_evidence"
    ] is False

    assert metadata[
        "retrieval_decision"
    ] == "retry"

    assert metadata[
        "retry_allowed"
    ] is True


def test_retrieval_decision_handles_failure():
    state = make_state(
        documents=[],
        retrieval_metadata={
            "status": "failure",
            "top_score": None,
            "score_gap": None,
            "retrieval_query": "vpn issue",
            "normalized_query": "vpn issue",
            "original_query": "VPN connection issue",
        },
    )

    result = retrieval_decision_agent(
        state
    )

    metadata = result[
        "retrieval_metadata"
    ]

    assert metadata[
        "enough_evidence"
    ] is False

    assert metadata[
        "evidence_reason"
    ] == "Retrieval failed."


def test_route_retrieval_decision_retry():
    state = {
        "retrieval_metadata": {
            "retrieval_decision": "retry"
        }
    }

    assert (
        route_retrieval_decision(state)
        == "retry"
    )


def test_route_retrieval_decision_continue():
    state = {
        "retrieval_metadata": {
            "retrieval_decision": "continue"
        }
    }

    assert (
        route_retrieval_decision(state)
        == "continue"
    )


def test_route_defaults_to_continue():
    state = {
        "retrieval_metadata": {}
    }

    assert (
        route_retrieval_decision(state)
        == "continue"
    )