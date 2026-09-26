"""
hitl_feedback.py — Human-in-the-Loop (HITL) Feedback Loop & Reward Integration (Phase 20.6).

Captures reviewer feedback, maps feedback to ground truth outcomes, integrates with RewardModel,
and persists feedback records for offline policy evaluation and future retraining.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from ai.policy.reward_model import RewardModel, RewardWeights
from ai.policy.action_space import DecisionAction


@dataclass
class HumanFeedback:
    """
    Structured human-in-the-loop reviewer feedback item.
    """

    feedback_id: str
    ticket_id: str
    reviewer_id: Optional[str] = "reviewer_01"
    reviewed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    accepted: bool = True
    original_action: str = "human_review"
    corrected_action: Optional[str] = None
    correction_reason: Optional[str] = None
    feedback_notes: Optional[str] = None
    resolution_quality: Optional[float] = None
    escalation_correct: Optional[bool] = None
    unsafe_resolution: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanFeedback":
        return cls(**data)


class HumanFeedbackStore:
    """
    Persistence store for HITL feedback records.
    Does NOT automatically retrain production online model state to prevent uncontrolled drift.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self._store: Dict[str, HumanFeedback] = {}
        if (
            self.filepath
            and os.path.exists(self.filepath)
            and os.path.getsize(self.filepath) > 0
        ):
            self.load_from_json(self.filepath)

    def save_feedback(self, feedback: HumanFeedback) -> None:
        self._store[feedback.ticket_id] = feedback
        if self.filepath:
            self.save_to_json(self.filepath)

    def get_feedback(self, ticket_id: str) -> Optional[HumanFeedback]:
        return self._store.get(ticket_id)

    def list_all(self) -> List[HumanFeedback]:
        return list(self._store.values())

    def save_to_json(self, filepath: str) -> None:
        data = [fb.to_dict() for fb in self._store.values()]
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            fb = HumanFeedback.from_dict(item)
            self._store[fb.ticket_id] = fb


class HITLRewardIntegrator:
    """
    Integrates HITL feedback with existing Phase 19 RewardModel.
    Calculates observed ground truth rewards when HITL feedback exists.
    """

    def __init__(self, reward_model: Optional[RewardModel] = None):
        self.reward_model = reward_model or RewardModel()

    def compute_reward_from_feedback(
        self,
        action: str,
        state: Dict[str, Any],
        feedback: Optional[HumanFeedback] = None,
    ) -> Tuple[float, str]:
        """
        Compute reward for action and state given optional HumanFeedback.
        Returns tuple of (reward_value, reward_source).
        """
        if feedback is None:
            r = self.reward_model.calculate_reward(action, state)
            return float(r), "proxy"

        act_clean = action.lower()
        observed_outcome: Dict[str, Any] = {}

        if feedback.accepted:
            observed_outcome["human_accepted"] = True
            if act_clean == DecisionAction.AUTO_RESOLVE.value:
                observed_outcome["successful_resolution"] = True
            elif act_clean == DecisionAction.HUMAN_REVIEW.value:
                observed_outcome["human_accepted"] = True
        else:
            observed_outcome["human_accepted"] = False
            if feedback.unsafe_resolution:
                observed_outcome["unsafe_autoresolve"] = True
                observed_outcome["policy_violated"] = True

        if (
            feedback.escalation_correct is False
            and act_clean == DecisionAction.ESCALATE.value
        ):
            observed_outcome["unnecessary_escalation"] = True

        reward = self.reward_model.calculate_reward(
            action, state, observed_outcome=observed_outcome
        )
        return float(reward), "hitl_observed"
