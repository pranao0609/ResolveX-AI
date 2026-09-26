"""
environment.py — Offline Contextual Bandit Environment for ResolveX.

Provides an environment interface for evaluating contextual bandit algorithms (LinUCB, Thompson Sampling)
against a dataset of contexts, actions, rewards, and action masks.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ai.policy.action_space import ActionSpace, DecisionAction
from ai.policy.dataset import DecisionPolicyDataset, DecisionPolicySample


@dataclass
class StepResult:
    context: np.ndarray
    sample_id: str
    action_mask: np.ndarray
    selected_action: str
    selected_action_index: int
    reward: float
    is_correct_action: bool


class ContextualBanditEnvironment:
    """
    Offline simulation environment for contextual bandit evaluation.
    Evaluates policy decisions step-by-step or in batch over a DecisionPolicyDataset.
    """

    def __init__(self, dataset: DecisionPolicyDataset, random_seed: int = 42):
        self.dataset = dataset
        self.dataset.sort_deterministically()
        self.random_seed = random_seed
        self.current_idx = 0
        self.rng = np.random.RandomState(random_seed)

    def reset(self) -> None:
        self.current_idx = 0
        self.rng = np.random.RandomState(self.random_seed)

    def has_next(self) -> bool:
        return self.current_idx < len(self.dataset)

    def get_current_sample(self) -> Optional[DecisionPolicySample]:
        if not self.has_next():
            return None
        return self.dataset.samples[self.current_idx]

    def get_action_mask(self, sample: DecisionPolicySample) -> np.ndarray:
        """
        Build binary action mask (True for allowed, False for forbidden).
        By default in environment all 4 actions are available, unless constrained.
        """
        mask = np.ones(ActionSpace.num_actions(), dtype=bool)
        return mask

    def step(self, chosen_action_index: int) -> StepResult:
        """
        Execute one step given chosen action index (0..3).
        Returns StepResult and advances iterator.
        """
        if not self.has_next():
            raise IndexError("Environment has reached end of dataset")

        sample = self.dataset.samples[self.current_idx]
        chosen_action = ActionSpace.to_action(chosen_action_index).value
        logged_action = sample.action
        mask = self.get_action_mask(sample)

        # In offline bandit OPE, if chosen action matches logged action, reward is sample reward.
        # Otherwise reward is 0.0 or estimated via proxy reward model.
        if chosen_action == logged_action:
            reward = sample.get_reward()
            is_correct = True
        else:
            reward = 0.0
            is_correct = False

        res = StepResult(
            context=sample.context.to_dict(),
            sample_id=sample.sample_id,
            action_mask=mask,
            selected_action=chosen_action,
            selected_action_index=chosen_action_index,
            reward=reward,
            is_correct_action=is_correct,
        )

        self.current_idx += 1
        return res
