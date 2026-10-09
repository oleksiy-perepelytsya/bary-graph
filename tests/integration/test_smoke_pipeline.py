from __future__ import annotations

import importlib

import pytest

from lib import checkpoint as cp_mod
from lib.config import Settings
from lib.db import STANDARD_INDEXES, ensure_indexes, ping
from scripts._base import STAGE_ORDER, STAGES, StageOrderError

pytestmark = pytest.mark.integration

FIXTURE = "tests/fixtures/kaikki-sample.jsonl"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path, tmp_state_dir, mongo_test_db):
    monkeypatch.setenv("BARY_FAKE_EMBED", "1")
    monkeypatch.setenv("KAIKKI_PATH", FIXTURE)
    monkeypatch.setenv("PARSED_DIR", str(tmp_path / "parsed"))
    # FakeEmbedder produces near-orthogonal vectors → relax cosine thresholds
    # so the fixture forms at least one BE per stage.
    monkeypatch.setenv("Q_MIN_L15", "0.0")
    monkeypatch.setenv("META_BARY_COS_THRESHOLD", "-1.0")
    yield


def test_mongo_reachable():
    assert ping(Settings.load()) is True


def test_test_db_carries_project_prefix(mongo_test_db):
    """Isolation invariant: the integration DB name must be project-owned."""
    s = Settings.load()
    assert s.mongo_db.startswith(s.mongo_test_db_prefix)
    assert mongo_test_db.database.name == s.mongo_db


def test_ensure_indexes(mongo_test_db):
    names = ensure_indexes(mongo_test_db)
    assert len(names) == len(STANDARD_INDEXES)
    info = mongo_test_db.index_information()
    # _id_ + the standard ones
    assert len(info) == len(STANDARD_INDEXES) + 1


def test_stage_order_guard_blocks_skip(tmp_state_dir):
    """Running stage 03 before 01/02 must fail loudly."""
    mod = importlib.import_module(STAGES["03_insert_nodes"])
    with pytest.raises(StageOrderError):
        mod.run([])


def test_full_pipeline_end_to_end(mongo_test_db, tmp_state_dir):
    """Run every stage in order on the 11-word fixture and assert invariants."""
    s = Settings.load()
    coll = mongo_test_db

    for stage in STAGE_ORDER:
        mod = importlib.import_module(STAGES[stage])
        mod.run([])
        assert cp_mod.is_done(stage, s), f"{stage} did not mark done"
        assert (tmp_state_dir / f"{stage}.json").exists()

    # --- Node counts ---
    n_words = coll.count_documents({"doc_type": "node", "node_type": "word"})
    n_senses = coll.count_documents({"doc_type": "node", "node_type": "sense"})
    assert n_words == 11
    assert n_senses == 14  # 11 words; happy/sad/gloomy have 2 senses each

    # --- Every L14 word has a vector after stage 05 ---
    assert coll.count_documents(
        {"doc_type": "node", "node_type": "word", "vector": None}
    ) == 0

    # --- BaryEdges exist at L15 and L14 ---
    assert coll.count_documents({"doc_type": "baryedge", "level": 15}) >= 1
    assert coll.count_documents({"doc_type": "baryedge", "level": 14}) >= 1

    # --- Unique-parent invariant: every parent_edge_id points at a baryedge ---
    for doc in coll.find({"parent_edge_id": {"$ne": None}}, {"parent_edge_id": 1}):
        parent = coll.find_one({"_id": doc["parent_edge_id"]}, {"doc_type": 1})
        assert parent is not None and parent["doc_type"] == "baryedge"

    # --- bary_vec invariant: all stored vectors are subtype-9 binary vectors ---
    sample = coll.find_one({"doc_type": "baryedge"}, {"vector": 1})
    assert sample["vector"].subtype == 9
    assert len(sample["vector"]) == 2 + s.embed_dim * 4  # header + float32 payload

    # --- Fermion order: with antonyms present, tier-1 'contradicts' fires ---
    assert coll.count_documents(
        {"doc_type": "baryedge", "level": 14, "edge_type": "contradicts"}
    ) >= 1

    # --- MetaBary: at least one L13 doc, structural fields only ---
    mb = coll.find_one({"doc_type": "baryedge", "level": 13})
    assert mb is not None
    assert "edge_type" not in mb and "type_vector" not in mb

    # --- Safeguard: re-running a completed stage without --reset is refused ---
    with pytest.raises(StageOrderError, match="already completed"):
        importlib.import_module(STAGES["04_l15_edges"]).run([])

    # --- Safeguard: data-level guard catches existing edges even after the
    #     checkpoint is cleared (e.g. pipeline_state/ wiped but DB intact) ---
    cp_mod.reset("04_l15_edges", s)
    with pytest.raises(RuntimeError, match="already present"):
        importlib.import_module(STAGES["04_l15_edges"]).run([])
