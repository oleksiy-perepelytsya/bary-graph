from __future__ import annotations

from lib.config import Settings


def test_load_defaults(monkeypatch, tmp_path):
    for k in ("MONGO_DB", "BATCH_SIZE", "BARY_FAKE_EMBED"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.load()
    assert s.mongo_db == "barygraph_poc"
    assert s.embed_dim == 768
    assert s.fake_embed is False


def test_load_env_override(monkeypatch):
    monkeypatch.setenv("MONGO_DB", "xyz")
    monkeypatch.setenv("BATCH_SIZE", "16")
    monkeypatch.setenv("BARY_FAKE_EMBED", "1")
    s = Settings.load()
    assert s.mongo_db == "xyz"
    assert s.batch_size == 16
    assert s.fake_embed is True


def test_q_seeds_defaults(monkeypatch):
    for k in (
        "Q_SEED_CONTRADICTS", "Q_SEED_APPLIES_TO", "Q_SEED_IS_INSTANCE_OF",
        "Q_SEED_EXTENDS", "Q_SEED_COORDINATE_TERMS", "Q_SEED_SYNONYMS",
    ):
        monkeypatch.delenv(k, raising=False)
    s = Settings.load()
    assert s.q_seeds == {
        "contradicts": 0.85,
        "applies_to": 0.55,
        "is_instance_of": 0.65,
        "extends": 0.60,
        "coordinate_terms": 0.70,
        "synonyms": 0.90,
    }


def test_q_seed_override(monkeypatch):
    monkeypatch.setenv("Q_SEED_CONTRADICTS", "0.42")
    s = Settings.load()
    assert s.q_seeds["contradicts"] == 0.42
    # others untouched
    assert s.q_seeds["synonyms"] == 0.90


def test_all_tunables_overridable(monkeypatch):
    overrides = {
        "META_BARY_COS_THRESHOLD": "0.95",
        "POLYSEMY_Q_FLOOR": "0.5",
        "EMBED_TIMEOUT_SECONDS": "10",
        "EMBED_BATCH_SIZE": "7",
    }
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    s = Settings.load()
    assert s.meta_bary_cos_threshold == 0.95
    assert s.polysemy_q_floor == 0.5
    assert s.embed_timeout_seconds == 10
    assert s.embed_batch_size == 7


def test_mongo_test_prefix_default():
    s = Settings.load()
    assert s.mongo_test_db_prefix == "barygraph_test_"


def test_doi_bridges_collection_default_and_override(monkeypatch):
    monkeypatch.delenv("MONGO_DOI_BRIDGES_COLLECTION", raising=False)
    assert Settings.load().mongo_doi_bridges_collection == "doi_bridges"
    monkeypatch.setenv("MONGO_DOI_BRIDGES_COLLECTION", "custom_bridges")
    assert Settings.load().mongo_doi_bridges_collection == "custom_bridges"


def test_batch_dup_threshold_default_and_override(monkeypatch):
    monkeypatch.delenv("BATCH_DUP_THRESHOLD", raising=False)
    assert Settings.load().batch_dup_threshold == 0.95
    monkeypatch.setenv("BATCH_DUP_THRESHOLD", "0.9")
    assert Settings.load().batch_dup_threshold == 0.9
