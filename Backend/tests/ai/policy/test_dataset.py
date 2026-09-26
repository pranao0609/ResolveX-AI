"""
test_dataset.py — Unit tests for DecisionPolicyDataset and DecisionPolicySample.
"""

import os
import tempfile
import pytest
from ai.policy.dataset import DecisionPolicyDataset, DecisionPolicySample
from ai.policy.feature_extractor import DecisionPolicyFeatures


def test_dataset_sample_validation():
    ctx = DecisionPolicyFeatures()
    sample = DecisionPolicySample(
        sample_id="s1",
        context=ctx,
        action="auto_resolve",
        observed_reward=1.0,
        propensity=0.5,
    )
    sample.validate()
    assert sample.get_reward() == 1.0


def test_invalid_sample_action():
    ctx = DecisionPolicyFeatures()
    sample = DecisionPolicySample(
        sample_id="s1",
        context=ctx,
        action="invalid_act",
    )
    with pytest.raises(ValueError):
        sample.validate()


def test_synthetic_fixture_generation():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=20, seed=42)
    assert len(ds) == 20
    X = ds.get_feature_matrix()
    y = ds.get_action_indices()
    r = ds.get_rewards()

    assert X.shape[0] == 20
    assert y.shape[0] == 20
    assert r.shape[0] == 20


def test_dataset_serialization():
    ds = DecisionPolicyDataset.create_synthetic_fixture(num_samples=10, seed=123)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ds.to_json(tmp_path)
        loaded_ds = DecisionPolicyDataset.from_json(tmp_path)
        assert len(loaded_ds) == 10
        assert loaded_ds.samples[0].sample_id == ds.samples[0].sample_id
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
