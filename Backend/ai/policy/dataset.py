"""
dataset.py — Decision Policy Dataset Abstraction for ResolveX.

Provides reusable dataset representation for historical and synthetic decision contexts.
Supports serialization, loading, validation, feature extraction, and matrix generation for OPE/Bandits.
"""

from dataclasses import dataclass, field, asdict
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from ai.policy.action_space import ActionSpace, DecisionAction
from ai.policy.feature_extractor import (
    DecisionPolicyFeatures,
    DecisionPolicyFeatureExtractor,
)


@dataclass
class DecisionPolicySample:
    """
    A single historical or synthetic decision sample.
    Distinguishes context, action, reward/outcome, and metadata.
    """

    sample_id: str
    context: DecisionPolicyFeatures
    action: str
    observed_reward: Optional[float] = None
    proxy_reward: Optional[float] = None
    propensity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.action not in ActionSpace.all_action_names():
            raise ValueError(
                f"Invalid action {self.action}. Must be one of {ActionSpace.all_action_names()}"
            )
        if self.propensity is not None and not (0.0 <= self.propensity <= 1.0):
            raise ValueError(
                f"Propensity must be between 0.0 and 1.0, got {self.propensity}"
            )

    def get_reward(self) -> float:
        if self.observed_reward is not None:
            return self.observed_reward
        if self.proxy_reward is not None:
            return self.proxy_reward
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "context": self.context.to_dict(),
            "action": self.action,
            "observed_reward": self.observed_reward,
            "proxy_reward": self.proxy_reward,
            "propensity": self.propensity,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionPolicySample":
        ctx_dict = data.get("context") or {}
        ctx = DecisionPolicyFeatures(**ctx_dict)
        sample = cls(
            sample_id=str(data["sample_id"]),
            context=ctx,
            action=str(data["action"]),
            observed_reward=data.get("observed_reward"),
            proxy_reward=data.get("proxy_reward"),
            propensity=data.get("propensity"),
            metadata=data.get("metadata") or {},
        )
        sample.validate()
        return sample


class DecisionPolicyDataset:
    """
    Deterministic container for decision policy samples.
    """

    def __init__(self, samples: Optional[List[DecisionPolicySample]] = None):
        self.samples: List[DecisionPolicySample] = samples or []

    def add_sample(self, sample: DecisionPolicySample) -> None:
        sample.validate()
        self.samples.append(sample)

    def sort_deterministically(self) -> None:
        """Sort samples deterministically by sample_id."""
        self.samples.sort(key=lambda s: s.sample_id)

    def __len__(self) -> int:
        return len(self.samples)

    def get_feature_matrix(self) -> np.ndarray:
        """Extract feature matrix X of shape (N, D)."""
        if not self.samples:
            return np.empty(
                (0, DecisionPolicyFeatureExtractor.feature_dimension()),
                dtype=np.float64,
            )
        vectors = [
            DecisionPolicyFeatureExtractor.to_vector(s.context) for s in self.samples
        ]
        return np.vstack(vectors)

    def get_action_indices(self) -> np.ndarray:
        """Extract 1D array of action indices of shape (N,)."""
        return np.array(
            [ActionSpace.to_index(s.action) for s in self.samples], dtype=np.int64
        )

    def get_rewards(self) -> np.ndarray:
        """Extract 1D array of rewards of shape (N,)."""
        return np.array([s.get_reward() for s in self.samples], dtype=np.float64)

    def get_propensities(self) -> Optional[np.ndarray]:
        """Extract 1D array of propensities if present for all samples, else None."""
        if any(s.propensity is None for s in self.samples):
            return None
        return np.array([s.propensity for s in self.samples], dtype=np.float64)

    def to_json(self, filepath: str) -> None:
        self.sort_deterministically()
        data = [s.to_dict() for s in self.samples]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "DecisionPolicyDataset":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = [DecisionPolicySample.from_dict(d) for d in data]
        ds = cls(samples)
        ds.sort_deterministically()
        return ds

    @classmethod
    def create_synthetic_fixture(
        cls, num_samples: int = 100, seed: int = 42
    ) -> "DecisionPolicyDataset":
        """
        Generate deterministic synthetic dataset for offline OPE and algorithm testing.
        Clearly marked as evaluation fixture.
        """
        rng = np.random.RandomState(seed)
        samples = []

        actions = ActionSpace.all_action_names()

        for i in range(num_samples):
            class_conf = float(rng.uniform(0.3, 0.99))
            ret_score = float(rng.uniform(0.2, 0.95))
            ret_gap = float(rng.uniform(0.0, 0.4))
            num_docs = float(rng.randint(0, 10))
            diag_conf = float(rng.uniform(0.4, 0.99))
            res_conf = float(rng.uniform(0.4, 0.99))
            ver_score = float(rng.uniform(0.3, 0.99))

            p_val = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
            c_val = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
            prev_att = float(rng.randint(0, 4))
            hist_res = float(rng.uniform(0.4, 0.9))

            mem_hist_count = float(rng.randint(0, 5))
            mem_hist_used = float(rng.choice([0.0, 1.0]))
            mem_conv_count = float(rng.randint(0, 10))
            mem_has_sol = float(rng.choice([0.0, 1.0]))

            requires_human = float(rng.choice([0.0, 1.0], p=[0.8, 0.2]))
            verification_failed = float(rng.choice([0.0, 1.0], p=[0.85, 0.15]))
            missing_info = float(rng.choice([0.0, 1.0], p=[0.8, 0.2]))
            fallback_used = float(rng.choice([0.0, 1.0], p=[0.9, 0.1]))
            errors_present = float(rng.choice([0.0, 1.0], p=[0.9, 0.1]))
            safety_violation = float(rng.choice([0.0, 1.0], p=[0.95, 0.05]))

            ctx = DecisionPolicyFeatures(
                classification_confidence=class_conf,
                retrieval_score=ret_score,
                retrieval_gap=ret_gap,
                number_of_documents=num_docs,
                diagnosis_confidence=diag_conf,
                resolution_confidence=res_conf,
                verification_score=ver_score,
                ticket_priority=p_val,
                ticket_complexity=c_val,
                previous_attempts=prev_att,
                historical_resolution_rate=hist_res,
                memory_historical_ticket_count=mem_hist_count,
                memory_historical_used=mem_hist_used,
                memory_conversation_count=mem_conv_count,
                memory_has_historical_solution=mem_has_sol,
                requires_human=requires_human,
                verification_failed=verification_failed,
                missing_information=missing_info,
                fallback_used=fallback_used,
                errors_present=errors_present,
                safety_violation=safety_violation,
            )

            # Assign synthetic action based on rules or random
            if errors_present > 0:
                act = DecisionAction.ESCALATE.value
            elif requires_human > 0 or verification_failed > 0:
                act = DecisionAction.HUMAN_REVIEW.value
            elif missing_info > 0:
                act = DecisionAction.ASK_CLARIFICATION.value
            elif class_conf > 0.75 and ver_score > 0.7:
                act = DecisionAction.AUTO_RESOLVE.value
            else:
                act = rng.choice(actions)

            # Assign synthetic rewards
            if act == DecisionAction.AUTO_RESOLVE.value and (
                verification_failed > 0 or errors_present > 0
            ):
                reward = -2.0
            elif act == DecisionAction.AUTO_RESOLVE.value:
                reward = 1.0
            elif act == DecisionAction.HUMAN_REVIEW.value and requires_human > 0:
                reward = 0.8
            else:
                reward = float(rng.uniform(-0.5, 0.5))

            sample = DecisionPolicySample(
                sample_id=f"sample_{i:04d}",
                context=ctx,
                action=act,
                observed_reward=reward,
                proxy_reward=reward,
                propensity=0.25,
                metadata={"synthetic": True},
            )
            samples.append(sample)

        ds = cls(samples)
        ds.sort_deterministically()
        return ds
