from typing import get_type_hints, TypedDict

from ai.graph.state import ResolveXState


def test_resolvex_state_is_typeddict():
    assert issubclass(ResolveXState, dict)
    assert hasattr(ResolveXState, "__annotations__")


def test_phase19_memory_fields_exist():
    annotations = get_type_hints(ResolveXState)

    assert "conversation_history" in annotations
    assert "previous_tickets" in annotations


def test_phase19_tool_state_exists():
    annotations = get_type_hints(ResolveXState)

    assert "tool_calls" in annotations


def test_phase19_observability_fields_exist():
    annotations = get_type_hints(ResolveXState)

    assert "model_versions" in annotations
    assert "prompt_versions" in annotations
    assert "latency" in annotations


def test_phase19_policy_fields_exist():
    annotations = get_type_hints(ResolveXState)

    assert "policy_features" in annotations
    assert "policy_action" in annotations
    assert "policy_metadata" in annotations


def test_phase19_state_accepts_new_fields():
    state: ResolveXState = {
        "ticket_id": 1,
        "ticket_text": "Unable to access email",
        "conversation_history": [
            {
                "role": "user",
                "content": "Unable to access email",
            }
        ],
        "previous_tickets": [
            {
                "ticket_id": 42,
                "resolution": "Reset Outlook credentials",
            }
        ],
        "tool_calls": [
            {
                "agent": "retrieval_agent",
                "tool": "search_knowledge_base",
                "result_count": 5,
                "success": True,
                "latency_ms": 84.0,
            }
        ],
        "model_versions": {
            "diagnosis": "openai/gpt-oss-120b",
        },
        "prompt_versions": {
            "diagnosis": "v1",
        },
        "latency": {
            "retrieval": 125.4,
            "diagnosis": 2100.7,
        },
        "policy_features": {
            "classification_confidence": 0.96,
            "retrieval_score": 0.82,
            "diagnosis_confidence": 0.88,
        },
        "policy_action": "human_review",
        "policy_metadata": {
            "policy_version": "baseline-v1",
        },
    }

    assert state["conversation_history"]
    assert state["previous_tickets"]
    assert state["tool_calls"]
    assert state["model_versions"]
    assert state["prompt_versions"]
    assert state["latency"]
    assert state["policy_features"]
    assert state["policy_action"] == "human_review"
    assert state["policy_metadata"]


def test_existing_phase14_18_fields_remain_available():
    state: ResolveXState = {
        "ticket_id": 1,
        "ticket_text": "Unable to access email",
        "cleaned_ticket": "unable to access email",
        "category": "software",
        "category_confidence": 0.96,
        "retrieved_context": "",
        "retrieved_documents": [],
        "retrieval_metadata": {},
        "diagnosis": "",
        "root_cause": "",
        "diagnosis_confidence": 0.0,
        "diagnosis_result": {},
        "resolution_steps": [],
        "evidence": [],
        "resolution_confidence": 0.0,
        "verification_passed": False,
        "verification_reason": "",
        "verification_confidence": 0.0,
        "verification_result": {},
        "decision": "human_review",
        "requires_human": True,
        "escalation_reason": "",
        "fallback_used": False,
        "errors": [],
        "warnings": [],
    }

    assert state["category"] == "software"
    assert state["decision"] == "human_review"
    assert state["requires_human"] is True
