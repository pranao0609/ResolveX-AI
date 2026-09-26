"""
test_action_space.py — Unit tests for ActionSpace and DecisionAction enum.
"""

import pytest
from ai.policy.action_space import ActionSpace, DecisionAction


def test_action_space_values():
    assert DecisionAction.AUTO_RESOLVE == "auto_resolve"
    assert DecisionAction.ASK_CLARIFICATION == "ask_clarification"
    assert DecisionAction.HUMAN_REVIEW == "human_review"
    assert DecisionAction.ESCALATE == "escalate"


def test_action_space_mapping():
    assert ActionSpace.num_actions() == 4
    assert ActionSpace.to_index("auto_resolve") == 0
    assert ActionSpace.to_index(DecisionAction.ASK_CLARIFICATION) == 1
    assert ActionSpace.to_index("human_review") == 2
    assert ActionSpace.to_index("escalate") == 3

    assert ActionSpace.to_action(0) == DecisionAction.AUTO_RESOLVE
    assert ActionSpace.to_action(1) == DecisionAction.ASK_CLARIFICATION
    assert ActionSpace.to_action(2) == DecisionAction.HUMAN_REVIEW
    assert ActionSpace.to_action(3) == DecisionAction.ESCALATE


def test_invalid_action():
    with pytest.raises(ValueError):
        ActionSpace.to_index("invalid_action")

    with pytest.raises(ValueError):
        ActionSpace.to_action(99)
