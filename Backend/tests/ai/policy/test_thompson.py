"""
test_thompson.py — Unit tests for Thompson Sampling contextual bandit algorithm.
"""

import os
import tempfile
import numpy as np
import pytest
from ai.policy.thompson import ThompsonSamplingPolicy
from ai.policy.dataset import DecisionPolicyDataset
from ai.policy.action_space import DecisionAction


def test_thompson_init_and_predict():
    policy = ThompsonSamplingPolicy(v2=1.0, sigma2=0.25, random_seed=42)
    state = {"diagnosis_confidence": 0.8}
    action, telemetry = policy.predict(state)

    assert isinstance(action, DecisionAction)
    assert telemetry["model_type"] == "thompson_sampling"
    assert "sampled_scores" in telemetry


def test_thompson_reproducibility():
    p1 = ThompsonSamplingPolicy(random_seed=123)
    p2 = ThompsonSamplingPolicy(random_seed=123)

    state = {"diagnosis_confidence": 0.85, "resolution_confidence": 0.90}
    act1, _ = p1.predict(state)
    act2, _ = p2.predict(state)
    assert act1 == act2


def test_thompson_fit_and_save_load():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=10, seed=42)
    policy = ThompsonSamplingPolicy(random_seed=42)
    policy.fit(ds)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        policy.save(tmp_path)
        loaded = ThompsonSamplingPolicy.load(tmp_path)
        assert loaded.random_seed == 42
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
