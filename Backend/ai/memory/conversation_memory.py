"""
Short-term conversational memory for ResolveX.

This memory is intentionally in-process/state-oriented.
Persistent historical memory belongs to TicketMemoryStore.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class ConversationMemory:
    """
    Bounded short-term memory for the current ResolveX execution.

    The class does not own LangGraph state directly. It operates on
    conversation-history lists so it can be used safely by nodes,
    tools, and tests.
    """

    DEFAULT_MAX_ENTRIES = 20
    MAX_ALLOWED_ENTRIES = 100

    def __init__(
        self,
        history: list[dict[str, Any]] | None = None,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")

        if max_entries > self.MAX_ALLOWED_ENTRIES:
            raise ValueError(f"max_entries cannot exceed {self.MAX_ALLOWED_ENTRIES}.")

        self.max_entries = max_entries
        self._history: list[dict[str, Any]] = []

        if history:
            self._history = [
                deepcopy(item) for item in history if isinstance(item, dict)
            ][-self.max_entries :]

    @property
    def history(self) -> list[dict[str, Any]]:
        """
        Return a defensive copy of the current history.

        Callers cannot accidentally mutate internal memory.
        """

        return deepcopy(self._history)

    @property
    def count(self) -> int:
        """Return the number of stored events."""

        return len(self._history)

    def append(
        self,
        *,
        role: str,
        content: str,
        **metadata: Any,
    ) -> dict[str, Any]:
        """
        Append one conversational event.

        Args:
            role: Event source, e.g. user, assistant, system, tool.
            content: Human-readable event content.
            metadata: Additional structured event metadata.
        """

        clean_role = str(role).strip().lower()
        clean_content = str(content).strip()

        if not clean_role:
            raise ValueError("Memory event role cannot be empty.")

        if not clean_content:
            raise ValueError("Memory event content cannot be empty.")

        event: dict[str, Any] = {
            "role": clean_role,
            "content": clean_content,
        }

        if metadata:
            event["metadata"] = deepcopy(metadata)

        self._history.append(event)

        if len(self._history) > self.max_entries:
            self._history = self._history[-self.max_entries :]

        return deepcopy(event)

    def add_event(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        """Append an already structured memory event."""

        if not isinstance(event, dict):
            raise TypeError("Memory event must be a dictionary.")

        role = event.get("role")
        content = event.get("content")

        if not role:
            raise ValueError("Memory event requires a role.")

        if content is None:
            raise ValueError("Memory event requires content.")

        metadata = {
            key: value for key, value in event.items() if key not in {"role", "content"}
        }

        return self.append(
            role=str(role),
            content=str(content),
            **metadata,
        )

    def recent(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent events.

        If limit is omitted, return the configured memory window.
        """

        if limit is None:
            limit = self.max_entries

        if limit < 1:
            raise ValueError("limit must be at least 1.")

        return deepcopy(self._history[-limit:])

    def clear(self) -> None:
        """Clear all short-term memory."""

        self._history.clear()

    def to_context(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return memory in an agent-safe context format.

        This intentionally returns a plain list of dictionaries.
        """

        return self.recent(limit=limit)

    def update_state(
        self,
        state: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Return a state update containing the current conversation history.

        The original state object is not mutated.
        """

        updated_state = dict(state)
        updated_state["conversation_history"] = self.to_context(
            limit=limit,
        )

        return updated_state

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> "ConversationMemory":
        """Create short-term memory from ResolveX state."""

        history = state.get("conversation_history", [])

        if not isinstance(history, list):
            history = []

        return cls(
            history=history,
            max_entries=max_entries,
        )
