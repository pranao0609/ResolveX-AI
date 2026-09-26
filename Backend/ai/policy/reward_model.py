"""
reward_model.py — Explicit Reward Model for ResolveX Contextual Bandit Evaluation & Learning.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from ai.policy.action_space import DecisionAction


@dataclass
class RewardWeights:
    """
    Configurable reward weights for proxy and observed reward calculation.
    """

    successful_resolution: float = 1.0
    human_acceptance: float = 0.8
    customer_satisfaction: float = 1.0

    unnecessary_escalation: float = -0.5
    unsafe_autoresolve: float = -2.0
    verification_failure: float = -1.5
    policy_violation: float = -2.0
    excessive_latency: float = -0.2

    clarification_penalty: float = -0.1
    human_review_cost: float = -0.3


class RewardModel:
    """
    Computes rewards for policy evaluation and bandit updating.
    Distinguishes observed ground truth reward from heuristic proxy reward.
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()

    def calculate_reward(
        self,
        action: str,
        state: Dict[str, Any],
        observed_outcome: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculates total reward for given action and state/outcome.
        If observed_outcome is available, computes observed reward.
        Otherwise computes proxy reward based on state verification & safety indicators.
        """
        action_clean = action.lower() if isinstance(action, str) else action.value

        # Check for explicit observed ground truth outcome first
        if observed_outcome:
            return self._calculate_observed_reward(
                action_clean, state, observed_outcome
            )

        # Fallback to proxy reward
        return self._calculate_proxy_reward(action_clean, state)

    def _calculate_observed_reward(
        self, action: str, state: Dict[str, Any], outcome: Dict[str, Any]
    ) -> float:
        reward = 0.0

        # Direct ground truth feedback
        if outcome.get("successful_resolution"):
            reward += self.weights.successful_resolution
        if outcome.get("human_accepted"):
            reward += self.weights.human_acceptance
        if outcome.get("customer_satisfied"):
            reward += self.weights.customer_satisfaction

        if outcome.get("policy_violated"):
            reward += self.weights.policy_violation
        if outcome.get("unsafe_autoresolve"):
            reward += self.weights.unsafe_autoresolve
        if outcome.get("unnecessary_escalation"):
            reward += self.weights.unnecessary_escalation

        return reward

    def _calculate_proxy_reward(self, action: str, state: Dict[str, Any]) -> float:
        reward = 0.0

        vr = state.get("verification_result") or {}
        if isinstance(vr, dict):
            ver_passed = bool(vr.get("verification_passed", True))
            ver_score = float(vr.get("verification_score", 0.0))
        else:
            ver_passed = not bool(state.get("verification_failed", False))
            ver_score = 0.0

        requires_human = bool(state.get("requires_human")) or bool(
            state.get("explicit_requires_human")
        )
        missing_info = bool(state.get("missing_information"))
        has_errors = bool(state.get("errors")) or bool(state.get("error"))

        if action == DecisionAction.AUTO_RESOLVE.value:
            if not ver_passed or has_errors or requires_human:
                reward += self.weights.unsafe_autoresolve
            else:
                reward += self.weights.successful_resolution * (0.5 + 0.5 * ver_score)

        elif action == DecisionAction.ASK_CLARIFICATION.value:
            if missing_info:
                reward += 0.5  # Appropriate clarification
            else:
                reward += self.weights.clarification_penalty

        elif action == DecisionAction.HUMAN_REVIEW.value:
            if requires_human or not ver_passed:
                reward += 0.5  # Safe triage
            else:
                reward += self.weights.human_review_cost

        elif action == DecisionAction.ESCALATE.value:
            if has_errors:
                reward += 0.5  # Necessary escalation
            else:
                reward += self.weights.unnecessary_escalation

        return reward
