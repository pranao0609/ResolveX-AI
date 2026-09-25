from unittest.mock import patch

import pytest

from ai.classification.classifier import (
    reclassify_with_zero_shot,
)


# ---------------------------------------------------------------------------
# Valid zero-shot reclassification
# ---------------------------------------------------------------------------


def test_reclassify_with_zero_shot_returns_mapped_calibrated_result():
    mock_result = {
        "labels": ["network_error"],
        "scores": [0.80],
    }

    with patch(
        "ai.classification.classifier._get_zero_shot_classifier",
        return_value=lambda text, categories: mock_result,
    ):

        category, confidence = reclassify_with_zero_shot(
            "The network connection keeps dropping."
        )

    assert category == "network"

    # 0.80 * 1.2 + 0.05 = 1.01 -> capped at 0.95
    assert confidence == 0.95


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_reclassify_with_zero_shot_rejects_short_text():
    with pytest.raises(ValueError):
        reclassify_with_zero_shot("abc")


# ---------------------------------------------------------------------------
# Model unavailable
# ---------------------------------------------------------------------------


def test_reclassify_with_zero_shot_raises_when_model_unavailable():
    with patch(
        "ai.classification.classifier._get_zero_shot_classifier",
        return_value=None,
    ):

        with pytest.raises(RuntimeError):
            reclassify_with_zero_shot(
                "The network connection is failing."
            )


# ---------------------------------------------------------------------------
# Invalid model result
# ---------------------------------------------------------------------------


def test_reclassify_with_zero_shot_rejects_invalid_result():
    mock_result = {
        "labels": [],
        "scores": [],
    }

    with patch(
        "ai.classification.classifier._get_zero_shot_classifier",
        return_value=lambda text, categories: mock_result,
    ):

        with pytest.raises(RuntimeError):
            reclassify_with_zero_shot(
                "The network connection is failing."
            )