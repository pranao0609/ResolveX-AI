"""
linucb.py — Disjoint LinUCB Contextual Bandit Policy for ResolveX.

Mathematical Assumptions:
-------------------------
1. Context Vector x in R^d: Context features are extracted via DecisionPolicyFeatureExtractor.
2. Independent Parameter Vector theta_a in R^d per action a in {0..K-1}:
   Expected reward under action a is linear: E[r | x, a] = x^T * theta_a.
3. Ridge Regularization: Matrix A_a = X_a^T * X_a + lambda * I_d (default lambda = 1.0).
4. Exploration Bonus: p_{a} = x^T * theta_a + alpha * sqrt(x^T * A_a^{-1} * x).
   Ref: Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW 2010.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import numpy as np
from ai.policy.action_space import ActionSpace, DecisionAction
from ai.policy.feature_extractor import (
    DecisionPolicyFeatureExtractor,
    DecisionPolicyFeatures,
)


class LinUCBPolicy:
    """
    LinUCB Policy with disjoint linear models for each action.
    """

    MODEL_TYPE: str = "linucb"
    MODEL_VERSION: str = "v1"

    def __init__(
        self,
        dimension: int = DecisionPolicyFeatureExtractor.feature_dimension(),
        alpha: float = 0.5,
        l2_reg: float = 1.0,
        random_seed: int = 42,
    ):
        self.dimension = dimension
        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)
        self.random_seed = random_seed
        self.num_actions = ActionSpace.num_actions()

        self.A = [
            self.l2_reg * np.eye(self.dimension, dtype=np.float64)
            for _ in range(self.num_actions)
        ]
        self.b = [
            np.zeros((self.dimension, 1), dtype=np.float64)
            for _ in range(self.num_actions)
        ]
        self.A_inv = [
            np.eye(self.dimension, dtype=np.float64) / self.l2_reg
            for _ in range(self.num_actions)
        ]

    def _update_inv(self, a_idx: int) -> None:
        try:
            self.A_inv[a_idx] = np.linalg.inv(self.A[a_idx])
        except np.linalg.LinAlgError:
            self.A_inv[a_idx] = np.linalg.pinv(self.A[a_idx])

    def predict(
        self,
        context: Union[np.ndarray, DecisionPolicyFeatures, Dict[str, Any]],
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[DecisionAction, Dict[str, Any]]:
        """
        Predict best action given context vector or state dict.
        Returns (selected_action, telemetry_dict).
        """
        if isinstance(context, np.ndarray):
            x = context.astype(np.float64).flatten()
        elif isinstance(context, DecisionPolicyFeatures):
            x = DecisionPolicyFeatureExtractor.to_vector(context)
        else:
            x = DecisionPolicyFeatureExtractor.state_to_vector(context)

        if len(x) != self.dimension:
            raise ValueError(
                f"Context vector dimension mismatch: expected {self.dimension}, got {len(x)}"
            )

        x_col = x.reshape(-1, 1)

        if action_mask is None:
            action_mask = np.ones(self.num_actions, dtype=bool)

        scores = np.full(self.num_actions, -np.inf, dtype=np.float64)
        mean_estimates = np.zeros(self.num_actions, dtype=np.float64)
        exploration_bonuses = np.zeros(self.num_actions, dtype=np.float64)

        for a in range(self.num_actions):
            if not action_mask[a]:
                continue
            A_inv_a = self.A_inv[a]
            theta_a = A_inv_a @ self.b[a]

            mean = float((theta_a.T @ x_col).item())
            var = float((x_col.T @ A_inv_a @ x_col).item())
            bonus = self.alpha * np.sqrt(max(0.0, var))

            mean_estimates[a] = mean
            exploration_bonuses[a] = bonus
            scores[a] = mean + bonus

        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            best_action_idx = ActionSpace.to_index(DecisionAction.HUMAN_REVIEW)
        else:
            best_action_idx = int(valid_indices[np.argmax(scores[valid_indices])])

        best_action = ActionSpace.to_action(best_action_idx)

        telemetry = {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "dimension": self.dimension,
            "selected_action": best_action.value,
            "selected_action_index": best_action_idx,
            "scores": {
                ActionSpace.to_action(i).value: float(scores[i])
                for i in range(self.num_actions)
            },
            "mean_estimates": {
                ActionSpace.to_action(i).value: float(mean_estimates[i])
                for i in range(self.num_actions)
            },
            "exploration_bonuses": {
                ActionSpace.to_action(i).value: float(exploration_bonuses[i])
                for i in range(self.num_actions)
            },
            "selected_score": float(scores[best_action_idx]),
        }

        return best_action, telemetry

    def update(
        self,
        context: Union[np.ndarray, DecisionPolicyFeatures, Dict[str, Any]],
        action: Union[str, DecisionAction, int],
        reward: float,
    ) -> None:
        if isinstance(context, np.ndarray):
            x = context.astype(np.float64).flatten()
        elif isinstance(context, DecisionPolicyFeatures):
            x = DecisionPolicyFeatureExtractor.to_vector(context)
        else:
            x = DecisionPolicyFeatureExtractor.state_to_vector(context)

        a_idx = action if isinstance(action, int) else ActionSpace.to_index(action)
        x_col = x.reshape(-1, 1)

        self.A[a_idx] += x_col @ x_col.T
        self.b[a_idx] += float(reward) * x_col
        self._update_inv(a_idx)

    def fit(self, dataset: Any) -> None:
        from ai.policy.dataset import DecisionPolicyDataset

        if not isinstance(dataset, DecisionPolicyDataset):
            raise TypeError("dataset must be an instance of DecisionPolicyDataset")

        for sample in dataset.samples:
            x = DecisionPolicyFeatureExtractor.to_vector(sample.context)
            a_idx = ActionSpace.to_index(sample.action)
            r = sample.get_reward()
            self.update(x, a_idx, r)

    def save(self, filepath: str) -> None:
        data = {
            "dimension": self.dimension,
            "alpha": self.alpha,
            "l2_reg": self.l2_reg,
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "A": [self.A[i].tolist() for i in range(self.num_actions)],
            "b": [self.b[i].tolist() for i in range(self.num_actions)],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LinUCBPolicy":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        policy = cls(
            dimension=data["dimension"],
            alpha=data["alpha"],
            l2_reg=data["l2_reg"],
        )
        policy.A = [np.array(arr, dtype=np.float64) for arr in data["A"]]
        policy.b = [np.array(arr, dtype=np.float64) for arr in data["b"]]
        for a in range(policy.num_actions):
            policy._update_inv(a)
        return policy
