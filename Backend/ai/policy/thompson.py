"""
thompson.py — Linear Contextual Thompson Sampling Policy for ResolveX.

Mathematical Assumptions:
-------------------------
1. Linear Gaussian Regression model per action a in {0..K-1}:
   r_a | x, theta_a ~ N(x^T * theta_a, sigma^2) where sigma^2 is noise variance (default 0.25).
2. Prior Distribution over parameter vector theta_a:
   theta_a ~ N(0, v^2 * I_d) where v^2 is prior variance (default 1.0).
3. Posterior Distribution:
   Given observed data D_a = {(x_i, r_i)}, posterior is Gaussian N(mu_a, Sigma_a):
   Sigma_a = ( (1/v^2) * I_d + (1/sigma^2) * X_a^T * X_a )^{-1}
   mu_a = (1/sigma^2) * Sigma_a * X_a^T * r_a
4. Decision Rule:
   Sample tilde_theta_a ~ N(mu_a, v^2 * Sigma_a) for each candidate action a.
   Choose a_star = argmax_{a} (x^T * tilde_theta_a).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import numpy as np
from ai.policy.action_space import ActionSpace, DecisionAction
from ai.policy.feature_extractor import (
    DecisionPolicyFeatureExtractor,
    DecisionPolicyFeatures,
)


class ThompsonSamplingPolicy:
    """
    Linear Contextual Thompson Sampling with Bayesian parameter posterior per action.
    """

    MODEL_TYPE: str = "thompson_sampling"
    MODEL_VERSION: str = "v1"

    def __init__(
        self,
        dimension: int = DecisionPolicyFeatureExtractor.feature_dimension(),
        v2: float = 1.0,
        sigma2: float = 0.25,
        random_seed: int = 42,
    ):
        self.dimension = dimension
        self.v2 = float(v2)
        self.sigma2 = float(sigma2)
        self.random_seed = random_seed
        self.num_actions = ActionSpace.num_actions()
        self.rng = np.random.RandomState(random_seed)

        self.B = [
            (1.0 / self.v2) * np.eye(self.dimension, dtype=np.float64)
            for _ in range(self.num_actions)
        ]
        self.f = [
            np.zeros((self.dimension, 1), dtype=np.float64)
            for _ in range(self.num_actions)
        ]
        self.B_inv = [
            self.v2 * np.eye(self.dimension, dtype=np.float64)
            for _ in range(self.num_actions)
        ]
        self.mu = [
            np.zeros((self.dimension, 1), dtype=np.float64)
            for _ in range(self.num_actions)
        ]

    def _update_posterior(self, a_idx: int) -> None:
        try:
            self.B_inv[a_idx] = np.linalg.inv(self.B[a_idx])
        except np.linalg.LinAlgError:
            self.B_inv[a_idx] = np.linalg.pinv(self.B[a_idx])
        self.mu[a_idx] = self.B_inv[a_idx] @ self.f[a_idx]

    def set_seed(self, seed: int) -> None:
        self.random_seed = seed
        self.rng = np.random.RandomState(seed)

    def predict(
        self,
        context: Union[np.ndarray, DecisionPolicyFeatures, Dict[str, Any]],
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[DecisionAction, Dict[str, Any]]:
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
        sampled_scores = np.zeros(self.num_actions, dtype=np.float64)

        for a in range(self.num_actions):
            if not action_mask[a]:
                continue
            mu_a = self.mu[a]
            cov_a = self.v2 * self.B_inv[a]

            cov_a = (cov_a + cov_a.T) / 2.0
            try:
                theta_sampled = self.rng.multivariate_normal(
                    mu_a.flatten(), cov_a
                ).reshape(-1, 1)
            except np.linalg.LinAlgError:
                theta_sampled = mu_a

            mean = float((mu_a.T @ x_col).item())
            sampled_val = float((theta_sampled.T @ x_col).item())

            mean_estimates[a] = mean
            sampled_scores[a] = sampled_val
            scores[a] = sampled_val

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
            "sampled_scores": {
                ActionSpace.to_action(i).value: float(sampled_scores[i])
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

        self.B[a_idx] += (1.0 / self.sigma2) * (x_col @ x_col.T)
        self.f[a_idx] += (float(reward) / self.sigma2) * x_col
        self._update_posterior(a_idx)

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
            "v2": self.v2,
            "sigma2": self.sigma2,
            "random_seed": self.random_seed,
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "B": [self.B[i].tolist() for i in range(self.num_actions)],
            "f": [self.f[i].tolist() for i in range(self.num_actions)],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ThompsonSamplingPolicy":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        policy = cls(
            dimension=data["dimension"],
            v2=data["v2"],
            sigma2=data["sigma2"],
            random_seed=data["random_seed"],
        )
        policy.B = [np.array(arr, dtype=np.float64) for arr in data["B"]]
        policy.f = [np.array(arr, dtype=np.float64) for arr in data["f"]]
        for a in range(policy.num_actions):
            policy._update_posterior(a)
        return policy
