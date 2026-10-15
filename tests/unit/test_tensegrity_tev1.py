from __future__ import annotations

from tensegrity_pipeline.tev1 import (
    ROLE_CRITERIA,
    ROLE_INSTRUCTION,
    FakeDecision,
    classify_role,
    role_questions,
)
from tensegrity_pipeline.tg_vec import (
    APPLIES_TO,
    CONTRADICTS,
    EXTENDS,
    IS_INSTANCE_OF,
    ROLES,
    SAME_PHENOMENON,
)


def test_fake_decision_is_deterministic_and_bounded():
    m = FakeDecision()
    r1 = m.decide("some state", role_questions())
    r2 = m.decide("some state", role_questions())
    assert r1 == r2
    ans = r1["answers"]["role"]
    assert ans["choice"] in ROLES
    probs = ans["probabilities"]
    assert set(probs) == set(ROLES)
    for role in ROLES:
        assert 0.0 <= probs[role] <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_fake_decision_varies_across_states():
    m = FakeDecision()
    choices = {
        m.decide(f"state {i}", role_questions())["answers"]["role"]["choice"]
        for i in range(50)
    }
    assert len(choices) > 1  # not pinned to a single relation


def test_classify_role_reports_winner_and_probability():
    cls = classify_role(FakeDecision(), "these two sentences agree")
    assert cls["role"] in ROLES
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
                        "probabilities": {
                            CONTRADICTS: 0.2,
                            APPLIES_TO: 0.1,
                            IS_INSTANCE_OF: 0.1,
                            EXTENDS: 0.1,
                            SAME_PHENOMENON: 0.8,
                        },
                    }
                }
            }

    cls = classify_role(Weird(), "state")
    assert cls["role"] == SAME_PHENOMENON
    assert abs(cls["role_probability"] - 0.8) < 1e-9


def test_role_questions_offer_every_relation():
    q = role_questions()
    assert q["role"]["type"] == "choice"
    assert q["role"]["instructions"] == ROLE_INSTRUCTION
    assert set(q["role"]["criteria"]) == set(ROLES)
    assert q["role"]["criteria"] == ROLE_CRITERIA
