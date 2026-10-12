"""Tensegrity pipeline settings.

Reuses :class:`lib.config.Settings` for all the generic plumbing (Mongo,
embedder, batching, state dirs) and adds the tensegrity-specific knobs: the
tev1 decision-model endpoint and the input document path.

The build profile is selected exactly like the PoC: ``--env NAME`` →
``.env.NAME``. The tensegrity runner defaults to ``--env tensegrity`` so
``python -m tensegrity_pipeline 01_parse --doc foo.txt`` works out of the box
without sourcing anything by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from lib.config import Settings


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TensegritySettings(Settings):
    """PoC settings + the tev1 decision model and the input document."""

    # --- tev1 decision model (complement / contradict) ---
    tev1_url: str = "http://localhost:11435"
    tev1_model: str = "tev1:0.8b"
    tev1_timeout_seconds: float = 300.0
    fake_decision: bool = False

    # --- Input document (t01 only). CLI --doc overrides this. ---
    tensegrity_doc: Path | None = None

    @classmethod
    def load(cls, dotenv_path: str | os.PathLike | None = None) -> TensegritySettings:
        base = Settings.load(dotenv_path)
        base_fields = {f.name: getattr(base, f.name) for f in fields(Settings)}
        doc = _env_str("TENSEGRITY_DOC", "")
        return cls(
            **base_fields,
            tev1_url=_env_str("TEV1_URL", cls.tev1_url),
            tev1_model=_env_str("TEV1_MODEL", cls.tev1_model),
            tev1_timeout_seconds=_env_float(
                "TEV1_TIMEOUT_SECONDS", cls.tev1_timeout_seconds
            ),
            fake_decision=_env_bool("BARY_FAKE_DECISION", False),
            tensegrity_doc=Path(doc) if doc else None,
        )
