from __future__ import annotations

import pytest

from ai.memory.conversation_memory import ConversationMemory


def test_conversation_memory_starts_empty() -> None:
    memory = ConversationMemory()

    assert memory.count == 0
    assert memory.history == []


def test_append_event() -> None:
    memory = ConversationMemory()

    event = memory.append(
        role="user",
        content="VPN is not connecting.",
        ticket_id=10,
    )

    assert event["role"] == "user"
    assert event["content"] == "VPN is not connecting."
    assert event["metadata"]["ticket_id"] == 10

    assert memory.count == 1


def test_recent_returns_latest_events() -> None:
    memory = ConversationMemory()

    for index in range(5):
        memory.append(
            role="assistant",
            content=f"event {index}",
        )

    recent = memory.recent(limit=2)

    assert len(recent) == 2
    assert recent[0]["content"] == "event 3"
    assert recent[1]["content"] == "event 4"


def test_memory_is_bounded() -> None:
    memory = ConversationMemory(max_entries=3)

    for index in range(5):
        memory.append(
            role="user",
            content=f"event {index}",
        )

    assert memory.count == 3

    history = memory.history

    assert history[0]["content"] == "event 2"
    assert history[1]["content"] == "event 3"
    assert history[2]["content"] == "event 4"


def test_history_is_defensive_copy() -> None:
    memory = ConversationMemory()

    memory.append(
        role="user",
        content="Original",
    )

    history = memory.history

    history[0]["content"] = "Modified"

    assert memory.history[0]["content"] == "Original"


def test_from_state() -> None:
    state = {
        "conversation_history": [
            {
                "role": "user",
                "content": "VPN failed",
            }
        ]
    }

    memory = ConversationMemory.from_state(state)

    assert memory.count == 1
    assert memory.history[0]["content"] == "VPN failed"


def test_update_state_does_not_mutate_original() -> None:
    state = {
        "ticket_id": 10,
        "conversation_history": [],
    }

    memory = ConversationMemory.from_state(state)

    memory.append(
        role="user",
        content="VPN failed",
    )

    updated = memory.update_state(state)

    assert state["conversation_history"] == []

    assert updated["conversation_history"][0]["content"] == ("VPN failed")


def test_empty_role_is_rejected() -> None:
    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.append(
            role="",
            content="test",
        )


def test_empty_content_is_rejected() -> None:
    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.append(
            role="user",
            content="",
        )


def test_invalid_memory_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        ConversationMemory(max_entries=0)

    with pytest.raises(ValueError):
        ConversationMemory(max_entries=101)
