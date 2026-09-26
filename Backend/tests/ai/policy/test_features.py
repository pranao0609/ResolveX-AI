"""
test_features.py — Unit tests for feature extraction layer.
"""

import numpy as np
import pytest
from ai.policy.feature_extractor import (
    DecisionPolicyFeatureExtractor,
    DecisionPolicyFeatures,
)


def test_feature_extractor_default():
    state = {}
    features = DecisionPolicyFeatureExtractor.extract_features(state)
    assert features.classification_confidence == 0.0
    assert features.ticket_priority == 0.5
    assert features.ticket_complexity == 0.5

    vec = DecisionPolicyFeatureExtractor.to_vector(features)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float64
    assert len(vec) == DecisionPolicyFeatureExtractor.feature_dimension()


def test_feature_extractor_from_dict():
    state = {
        "diagnosis_confidence": 0.85,
        "resolution_confidence": 0.90,
        "verification_result": {
            "verification_score": 0.88,
            "verification_passed": True,
        },
        "priority": "high",
        "complexity": "urgent",
        "previous_tickets": [{"id": 1, "solution": "Fixed config"}],
        "conversation_history": ["msg1", "msg2"],
        "requires_human": False,
    }
    features = DecisionPolicyFeatureExtractor.extract_features(state)
    assert features.diagnosis_confidence == 0.85
    assert features.resolution_confidence == 0.90
    assert features.verification_score == 0.88
    assert features.ticket_priority == 0.75
    assert features.ticket_complexity == 1.0
    assert features.memory_historical_ticket_count == 1.0
    assert features.memory_historical_used == 1.0
    assert features.memory_conversation_count == 2.0
    assert features.memory_has_historical_solution == 1.0
    assert features.requires_human == 0.0
    assert features.verification_failed == 0.0


def test_feature_vector_ordering():
    features = DecisionPolicyFeatures(classification_confidence=0.7)
    vec = DecisionPolicyFeatureExtractor.to_vector(features)
    assert vec[0] == 0.7
