"""
test_linucb.py — Unit tests for LinUCB policy algorithm.
"""

import os
import tempfile
import numpy as np
import pytest
from ai.policy.linucb import LinUCBPolicy
from ai.policy.dataset import DecisionPolicyDataset
from ai.policy.action_space import DecisionAction


def test_linucb_init_and_predict():
    policy = LinUCBPolicy(alpha=0.5, random_seed=42)
    state = {"diagnosis_confidence": 0.8, "resolution_confidence": 0.9}
    action, telemetry = policy.predict(state)

    assert isinstance(action, DecisionAction)
    assert telemetry["model_type"] == "linucb"
    assert "scores" in telemetry


def test_linucb_action_masking():
    policy = LinUCBPolicy(alpha=0.5)
    state = {"diagnosis_confidence": 0.8}
    mask = np.array([False, False, True, False])  # only human_review allowed
    action, telemetry = policy.predict(state, action_mask=mask)
    assert action == DecisionAction.HUMAN_REVIEW


def test_linucb_fit_and_update():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=10, seed=42)
    policy = LinUCBPolicy(alpha=0.5)

    policy.fit(ds)
    action, telemetry = policy.predict(ds.samples[0].context)
    assert isinstance(action, DecisionAction)


def test_linucb_save_load():
    policy = LinUCBPolicy(alpha=0.3)
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=5, seed=42)
    policy.fit(ds)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        policy.save(tmp_path)
        loaded = LinUCBPolicy.load(tmp_path)
        assert loaded.alpha == 0.3
        act1, _ = policy.predict(ds.samples[0].context)
        act2, _ = loaded.predict(ds.samples[0].context)
        assert act1 == act2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
