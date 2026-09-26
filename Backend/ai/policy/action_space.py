"""
action_space.py — Canonical Action Space for ResolveX Contextual Bandit Decision Policy.
"""

from enum import Enum
from typing import List, Dict, Any, Union


class DecisionAction(str, Enum):
    AUTO_RESOLVE = "auto_resolve"
    ASK_CLARIFICATION = "ask_clarification"
    HUMAN_REVIEW = "human_review"
    ESCALATE = "escalate"


class ActionSpace:
    """
    Canonical action space mapping between string action names and integer action indices.
    """

    ACTIONS: List[DecisionAction] = [
        DecisionAction.AUTO_RESOLVE,
        DecisionAction.ASK_CLARIFICATION,
        DecisionAction.HUMAN_REVIEW,
        DecisionAction.ESCALATE,
    ]

    _ACTION_TO_INDEX: Dict[DecisionAction, int] = {
        action: idx for idx, action in enumerate(ACTIONS)
    }

    _INDEX_TO_ACTION: Dict[int, DecisionAction] = {
        idx: action for idx, action in enumerate(ACTIONS)
    }

    @classmethod
    def num_actions(cls) -> int:
        return len(cls.ACTIONS)

    @classmethod
    def to_index(cls, action: Union[str, DecisionAction]) -> int:
        if isinstance(action, str):
            try:
                action = DecisionAction(action.lower())
            except ValueError:
                raise ValueError(f"Unknown action string: {action}")
        if action not in cls._ACTION_TO_INDEX:
            raise ValueError(f"Action {action} not in action space")
        return cls._ACTION_TO_INDEX[action]

    @classmethod
    def to_action(cls, index: int) -> DecisionAction:
        if index not in cls._INDEX_TO_ACTION:
            raise ValueError(
                f"Action index {index} out of bounds (0..{len(cls.ACTIONS)-1})"
            )
        return cls._INDEX_TO_ACTION[index]

    @classmethod
    def all_actions(cls) -> List[DecisionAction]:
        return list(cls.ACTIONS)

    @classmethod
    def all_action_names(cls) -> List[str]:
        return [action.value for action in cls.ACTIONS]
