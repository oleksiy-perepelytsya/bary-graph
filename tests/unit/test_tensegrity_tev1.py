from __future__ import annotations

from tensegrity_pipeline.tev1 import (
    FakeDecision,
    classify_role,
    role_questions,
)
from tensegrity_pipeline.tg_vec import COMPLEMENT, CONTRADICT


def test_fake_decision_is_deterministic_and_bounded():
    m = FakeDecision()
    r1 = m.decide("some state", role_questions())
    r2 = m.decide("some state", role_questions())
    assert r1 == r2
    ans = r1["answers"]["role"]
    assert ans["choice"] in (COMPLEMENT, CONTRADICT)
    probs = ans["probabilities"]
    assert 0.0 <= probs[COMPLEMENT] <= 1.0
    assert 0.0 <= probs[CONTRADICT] <= 1.0
    assert abs(probs[COMPLEMENT] + probs[CONTRADICT] - 1.0) < 1e-9


def test_classify_role_reports_winner_and_probability():
    cls = classify_role(FakeDecision(), "these two sentences agree")
    assert cls["role"] in (COMPLEMENT, CONTRADICT)
    assert 0.0 <= cls["role_probability"] <= 1.0
    assert cls["role_source"] == "fake"


def test_classify_role_never_raises_on_failure():
    class Broken:
        source = "tev1"

        def decide(self, state, questions):
            raise RuntimeError("no model")

    cls = classify_role(Broken(), "whatever")
    assert cls["role"] is None
    assert cls["role_probability"] == 0.0
    assert cls["role_source"] == "unclassified"


def test_classify_role_falls_back_to_argmax_when_choice_unknown():
    class Weird:
        source = "tev1"

        def decide(self, state, questions):
            return {
                "answers": {
                    "role": {
                        "choice": "maybe",
                        "probabilities": {COMPLEMENT: 0.2, CONTRADICT: 0.8},
                    }
                }
            }

    cls = classify_role(Weird(), "state")
    assert cls["role"] == CONTRADICT
    assert abs(cls["role_probability"] - 0.8) < 1e-9


def test_role_questions_offer_both_roles():
    q = role_questions()
    assert q["role"]["type"] == "choice"
    assert set(q["role"]["criteria"]) == {COMPLEMENT, CONTRADICT}
