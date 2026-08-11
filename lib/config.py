from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# Default q_seed per L14 fermion-order tier. Keys match the "q_seed key" column
# of the fermion table in CLAUDE.md (one key per priority tier — NOT per
# edge_type, since tiers 5 and 6 both produce edge_type='same_phenomenon').
# Overridable individually via Q_SEED_<UPPER>.
_DEFAULT_Q_SEEDS: dict[str, float] = {
    "contradicts": 0.85,
    "applies_to": 0.55,
    "is_instance_of": 0.65,
    "extends": 0.60,
    "coordinate_terms": 0.70,
    "synonyms": 0.90,
}


def _load_q_seeds() -> dict[str, float]:
    return {k: _env_float(f"Q_SEED_{k.upper()}", v) for k, v in _DEFAULT_Q_SEEDS.items()}


@dataclass(frozen=True)
class Settings:
    # --- Mongo ---
    mongo_uri: str = "mongodb://localhost:27117/?directConnection=true"
    mongo_db: str = "barygraph_poc"
    mongo_collection: str = "barygraph"
    mongo_doi_bridges_collection: str = "doi_bridges"
    # Safety net for integration-test teardown: only DBs starting with this prefix
    # may be dropped by the test fixture. Changing this requires updating the
    # fixture guard in tests/conftest.py.
    mongo_test_db_prefix: str = "barygraph_test_"

    # --- Models (Ollama) ---
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text:v1.5"
    embed_dim: int = 768
    embed_timeout_seconds: float = 600.0
    fake_embed: bool = False

    # --- Data / state ---
    kaikki_path: Path = field(default_factory=lambda: Path("data/kaikki-en.jsonl"))
    parsed_dir: Path = field(default_factory=lambda: Path("data/parsed"))
    pipeline_state_dir: Path = field(default_factory=lambda: Path("pipeline_state"))

    # --- Batching ---
    batch_size: int = 512          # generic ingest / bulk-write batch
    embed_batch_size: int = 16     # texts-per-request to Ollama (CPU-safe)

    # --- Tunable thresholds (pipeline) ---
    q_min_l15: float = 0.72                    # Stage 3b greedy-match floor
    disambig_threshold: float = 0.72           # lib.disambiguate cosine fallback
    meta_bary_cos_threshold: float = 0.90      # Stage 7: L13 triad formation
    polysemy_q_floor: float = 0.40             # open question; tune post L14
    level_factor_alpha: float = 0.5           # v0.5 R6: accumulated_weight amplification
    batch_dup_threshold: float = 0.95         # academic batch: same-sense-reworded merge floor

    # --- L14 edge q_seeds (fermion order) ---
    q_seeds: dict[str, float] = field(default_factory=_load_q_seeds)

    # --- Vertex AI ---
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-pro-preview-05-06"
    vertex_gcs_bucket: str = ""
    vertex_temperature: float = 0.2
    vertex_frequency_penalty: float = 0.0

    # --- Misc ---
    log_level: str = "INFO"

    @classmethod
    def load(cls, dotenv_path: str | os.PathLike | None = None) -> Settings:
        """Load settings from environment, with optional .env file."""
        load_dotenv(dotenv_path, override=False)
        return cls(
            mongo_uri=_env_str("MONGO_URI", cls.mongo_uri),
            mongo_db=_env_str("MONGO_DB", cls.mongo_db),
            mongo_collection=_env_str("MONGO_COLLECTION", cls.mongo_collection),
            mongo_doi_bridges_collection=_env_str(
                "MONGO_DOI_BRIDGES_COLLECTION", cls.mongo_doi_bridges_collection
            ),
            mongo_test_db_prefix=_env_str("MONGO_TEST_DB_PREFIX", cls.mongo_test_db_prefix),
            ollama_url=_env_str("OLLAMA_URL", cls.ollama_url),
            embed_model=_env_str("EMBED_MODEL", cls.embed_model),
            embed_dim=_env_int("EMBED_DIM", cls.embed_dim),
            embed_timeout_seconds=_env_float("EMBED_TIMEOUT_SECONDS", cls.embed_timeout_seconds),
            fake_embed=_env_bool("BARY_FAKE_EMBED", False),
            kaikki_path=Path(_env_str("KAIKKI_PATH", "data/kaikki-en.jsonl")),
            parsed_dir=Path(_env_str("PARSED_DIR", "data/parsed")),
            pipeline_state_dir=Path(_env_str("PIPELINE_STATE_DIR", "pipeline_state")),
            batch_size=_env_int("BATCH_SIZE", cls.batch_size),
            embed_batch_size=_env_int("EMBED_BATCH_SIZE", cls.embed_batch_size),
            q_min_l15=_env_float("Q_MIN_L15", cls.q_min_l15),
            disambig_threshold=_env_float("DISAMBIG_THRESHOLD", cls.disambig_threshold),
            meta_bary_cos_threshold=_env_float(
                "META_BARY_COS_THRESHOLD", cls.meta_bary_cos_threshold
            ),
            polysemy_q_floor=_env_float("POLYSEMY_Q_FLOOR", cls.polysemy_q_floor),
            level_factor_alpha=_env_float("LEVEL_FACTOR_ALPHA", cls.level_factor_alpha),
            batch_dup_threshold=_env_float("BATCH_DUP_THRESHOLD", cls.batch_dup_threshold),
            q_seeds=_load_q_seeds(),
            vertex_project=_env_str("VERTEX_PROJECT", cls.vertex_project),
            vertex_location=_env_str("VERTEX_LOCATION", cls.vertex_location),
            vertex_model=_env_str("VERTEX_MODEL", cls.vertex_model),
            vertex_gcs_bucket=_env_str("VERTEX_GCS_BUCKET", cls.vertex_gcs_bucket),
            vertex_temperature=_env_float("VERTEX_TEMPERATURE", cls.vertex_temperature),
            vertex_frequency_penalty=_env_float(
                "VERTEX_FREQUENCY_PENALTY", cls.vertex_frequency_penalty
            ),
            log_level=_env_str("LOG_LEVEL", cls.log_level),
        )
