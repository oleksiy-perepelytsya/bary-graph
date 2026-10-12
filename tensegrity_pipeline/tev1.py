"""tev1 decision model client — the complement/contradict role classifier.

The model is served by ollama on the Jev TypeSafe endpoint ``POST
/v1/systemone`` (the same wire format :mod:`cognitive.not_sure` uses). It reads
a STATE once and scores the allowed answer tokens directly.

Role contract used throughout the pipeline::

    {
      "role": "complement" | "contradict" | None,   # argmax of probabilities
      "role_probability": 0.0 .. 1.0,                # winning probability only
      "confidence": ... | None,                      # model-reported confidence
      "role_source": "tev1" | "fake" | "unclassified",
    }

An unreachable model yields ``role=None`` with ``role_source="unclassified"`` —
the pipeline records the gap rather than inventing a stance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

from lib.config import Settings
from tensegrity_pipeline.tg_vec import COMPLEMENT, CONTRADICT, ROLES

_log = logging.getLogger(__name__)

ROLE_FIELD = "role"
ROLE_INSTRUCTION = (
    "Decide the logical relationship between the statements in the state: do "
    "they complement each other (consistent, mutually supporting) or "
    "contradict each other (mutually exclusive, one negates the other)?"
)
ROLE_CRITERIA: dict[str, str] = {
    COMPLEMENT: "the statements are consistent and reinforce the same meaning",
    CONTRADICT: "the statements are mutually exclusive and cannot both be true",
}


class DecisionModel(Protocol):
    source: str

    def decide(self, state: str, questions: dict) -> dict: ...


class Tev1Error(RuntimeError):
    """Transport/HTTP failure talking to the tev1 endpoint."""


class Tev1Client:
    """Minimal ``/v1/systemone`` client with transport retries."""

    source = "tev1"

    def __init__(
        self,
        url: str,
        model: str,
        *,
        timeout: float = 300.0,
        retries: int = 2,
        opener=urllib.request.urlopen,
    ) -> None:
        self._url = url.rstrip("/") + "/v1/systemone"
        self._model = model
        self._timeout = timeout
        self._retries = retries
        self._opener = opener

    def decide(self, state: str, questions: dict) -> dict:
        payload = json.dumps(
            {"model": self._model, "state": state, "questions": questions}
        ).encode()
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            req = urllib.request.Request(
                self._url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with self._opener(req, timeout=self._timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace").strip()
                raise Tev1Error(
                    f"tev1 HTTP {exc.code} (model {self._model!r}): {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
                if attempt < self._retries:
                    wait = 2 * (2**attempt)
                    _log.warning(
                        "tev1 transport error (attempt %d/%d): %s — retry in %ds",
                        attempt + 1, self._retries + 1, exc, wait,
                    )
                    time.sleep(wait)
                    continue
                raise Tev1Error(f"cannot reach tev1 at {self._url}: {exc}") from exc
        raise Tev1Error(f"cannot reach tev1 at {self._url}: {last}")


class FakeDecision:
    """Deterministic offline stand-in for CI and unit tests."""

    source = "fake"

    def decide(self, state: str, questions: dict) -> dict:
        h = int.from_bytes(hashlib.sha256(state.encode("utf-8")).digest()[:8], "big")
        p_complement = 0.05 + 0.90 * ((h % 10_000) / 10_000.0)
        answers: dict[str, dict] = {}
        for field in questions:
            answers[field] = {
                "type": "choice",
                "choice": COMPLEMENT if p_complement >= 0.5 else CONTRADICT,
                "probabilities": {
                    COMPLEMENT: p_complement,
                    CONTRADICT: 1.0 - p_complement,
                },
                "confidence": abs(p_complement - 0.5) * 2.0,
            }
        return {"model": "fake", "answers": answers}


def role_questions() -> dict:
    """The single-role question object sent to the decision model."""
    return {
        ROLE_FIELD: {
            "type": "choice",
            "instructions": ROLE_INSTRUCTION,
            "criteria": dict(ROLE_CRITERIA),
        }
    }


def classify_role(model: DecisionModel, state: str) -> dict:
    """Ask ``model`` for the complement/contradict role of ``state``.

    Never raises: transport failures are logged and returned as
    ``role_source="unclassified"`` so the caller can keep going.
    """
    try:
        resp = model.decide(state, role_questions())
    except Exception as exc:  # noqa: BLE001 - degrade, never crash the stage
        _log.warning("tev1 role classification failed: %s", exc)
        return _unclassified()
    answers = resp.get("answers") if isinstance(resp, dict) else None
    ans = answers.get(ROLE_FIELD) if isinstance(answers, dict) else None
    if not isinstance(ans, dict):
        return _unclassified()
    choice = ans.get("choice")
    probs = ans.get("probabilities") or {}
    if choice not in ROLES:
        # Some responses land on a token we didn't offer; fall back to argmax.
        if isinstance(probs, dict) and probs:
            choice = max(ROLES, key=lambda r: float(probs.get(r, 0.0)))
        else:
            return _unclassified()
    prob = float(probs.get(choice, 0.0)) if isinstance(probs, dict) else 0.0
    return {
        "role": choice,
        "role_probability": prob,
        "confidence": ans.get("confidence"),
        "role_source": getattr(model, "source", "tev1"),
    }


def _unclassified() -> dict:
    return {
        "role": None,
        "role_probability": 0.0,
        "confidence": None,
        "role_source": "unclassified",
    }


def get_decision_model(settings: Settings) -> DecisionModel:
    """FakeDecision when ``fake_decision``/``BARY_FAKE_DECISION`` is set, else
    a live :class:`Tev1Client`."""
    fake = getattr(settings, "fake_decision", False)
    env = os.environ.get("BARY_FAKE_DECISION", "")
    if fake or env.strip().lower() in {"1", "true", "yes", "on"}:
        return FakeDecision()
    return Tev1Client(
        settings.tev1_url,  # type: ignore[attr-defined]
        settings.tev1_model,  # type: ignore[attr-defined]
        timeout=getattr(settings, "tev1_timeout_seconds", 300.0),
    )
