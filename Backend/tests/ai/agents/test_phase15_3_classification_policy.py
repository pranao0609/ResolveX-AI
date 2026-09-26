from ai.agents.classification_policy import (
    classify_confidence_level,
    classification_requires_review,
    classification_requires_reclassification,
)


def test_high_confidence():
    assert classify_confidence_level(0.75) == "high"
    assert classify_confidence_level(0.95) == "high"
    assert classify_confidence_level(1.0) == "high"


def test_medium_confidence():
    assert classify_confidence_level(0.50) == "medium"
    assert classify_confidence_level(0.60) == "medium"
    assert classify_confidence_level(0.7499) == "medium"


def test_low_confidence():
    assert classify_confidence_level(0.49) == "low"
    assert classify_confidence_level(0.20) == "low"
    assert classify_confidence_level(0.0) == "low"


def test_high_confidence_does_not_require_review():
    assert classification_requires_review(0.75) is False
    assert classification_requires_review(0.95) is False


def test_medium_confidence_requires_review():
    assert classification_requires_review(0.74) is True
    assert classification_requires_review(0.60) is True


def test_low_confidence_requires_reclassification():
    assert classification_requires_reclassification(0.49) is True
    assert classification_requires_reclassification(0.20) is True


def test_medium_confidence_does_not_require_reclassification():
    assert classification_requires_reclassification(0.50) is False
    assert classification_requires_reclassification(0.60) is False


def test_confidence_is_clamped():
    assert classify_confidence_level(1.5) == "high"
    assert classify_confidence_level(-0.5) == "low"
