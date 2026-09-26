"""
test_hitl.py — Unit tests for HITL Feedback Loop and Reward Integration.
"""

import os
import tempfile
from ai.evaluation.hitl_feedback import (
    HumanFeedback,
    HumanFeedbackStore,
    HITLRewardIntegrator,
)


def test_human_feedback_model():
    fb = HumanFeedback(
        feedback_id="fb_01",
        ticket_id="ticket_201",
        accepted=True,
        original_action="auto_resolve",
    )
    assert fb.accepted is True
    assert fb.ticket_id == "ticket_201"


def test_human_feedback_store():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        store = HumanFeedbackStore(tmp_path)
        fb = HumanFeedback(
            feedback_id="fb_02",
            ticket_id="ticket_202",
            accepted=False,
            corrected_action="human_review",
        )
        store.save_feedback(fb)

        loaded_store = HumanFeedbackStore(tmp_path)
        fetched = loaded_store.get_feedback("ticket_202")
        assert fetched is not None
        assert fetched.accepted is False
        assert fetched.corrected_action == "human_review"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_hitl_reward_integrator_observed():
    integrator = HITLRewardIntegrator()
    state = {
        "verification_result": {"verification_passed": True, "verification_score": 0.9}
    }
    fb = HumanFeedback(
        feedback_id="fb_03",
        ticket_id="t3",
        accepted=True,
        original_action="auto_resolve",
    )

    reward, source = integrator.compute_reward_from_feedback(
        "auto_resolve", state, feedback=fb
    )
    assert source == "hitl_observed"
    assert reward > 0.0


def test_hitl_reward_integrator_proxy():
    integrator = HITLRewardIntegrator()
    state = {
        "verification_result": {"verification_passed": True, "verification_score": 0.9}
    }

    reward, source = integrator.compute_reward_from_feedback(
        "auto_resolve", state, feedback=None
    )
    assert source == "proxy"
