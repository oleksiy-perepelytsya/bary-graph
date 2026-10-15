from __future__ import annotations

import json

from tensegrity_pipeline.tev1 import (
    ROLE_CRITERIA,
    ROLE_INSTRUCTION,
    FakeDecision,
    Tev1Client,
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


class _CapturingOpener:
    """Stand-in urlopen that records the Request and returns a canned body."""

    def __init__(self) -> None:
        self.requests: list = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        return _FakeResponse(json.dumps({"answers": {}}).encode())


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_tev1_client_sends_no_auth_header_when_no_token(monkeypatch):
    for var in ("OLLAMA_API_KEY", "OLLAMA_AUTH_TOKEN", "TEV1_API_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    opener = _CapturingOpener()
    Tev1Client("http://localhost:11435", "tev1:0.8b", opener=opener).decide("s", {})
    assert "Authorization" not in opener.requests[0].headers
    assert opener.requests[0].full_url.endswith("/v1/systemone")


def test_tev1_client_auth_header_from_env_token(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TEV1_API_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "tok123")
    opener = _CapturingOpener()
    Tev1Client("https://tunnel.example", "tev1:0.8b", opener=opener).decide("s", {})
    # urllib capitalizes header keys on the Request object.
    assert opener.requests[0].headers["Authorization"] == "Bearer tok123"


def test_tev1_client_explicit_bearer_wins(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "envtok")
    opener = _CapturingOpener()
    Tev1Client("https://tunnel.example", "m", bearer="argtok", opener=opener).decide("s", {})
    assert opener.requests[0].headers["Authorization"] == "Bearer argtok"
