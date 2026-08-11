from __future__ import annotations

import importlib
import json

import pytest

from lib import doi_bridge
from lib.config import Settings
from lib.parse_batch import BATCH_POS
from scripts._base import STAGE_ORDER, STAGES

pytestmark = pytest.mark.integration

FIXTURE = "tests/fixtures/kaikki-sample.jsonl"
BATCH_FIXTURE = "tests/fixtures/batch-for-ingest.jsonl"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path, tmp_state_dir, mongo_test_db):
    monkeypatch.setenv("BARY_FAKE_EMBED", "1")
    monkeypatch.setenv("KAIKKI_PATH", FIXTURE)
    monkeypatch.setenv("PARSED_DIR", str(tmp_path / "parsed"))
    # FakeEmbedder produces near-orthogonal vectors → relax cosine thresholds
    # so the fixture forms at least one BE per stage (mirrors test_smoke_pipeline).
    monkeypatch.setenv("Q_MIN_L15", "0.0")
    monkeypatch.setenv("META_BARY_COS_THRESHOLD", "-1.0")
    yield


def _run_full_pipeline():
    for stage in STAGE_ORDER:
        importlib.import_module(STAGES[stage]).run([])


def test_ingest_batch_merges_and_creates(mongo_test_db, tmp_state_dir):
    coll = mongo_test_db
    s = Settings.load()
    _run_full_pipeline()

    happy_word = coll.find_one(
        {"doc_type": "node", "node_type": "word", "properties.word": "happy"}
    )
    assert happy_word is not None
    n_senses_before = coll.count_documents(
        {"doc_type": "node", "node_type": "sense", "properties.word": "happy"}
    )

    from scripts import ingest_batch

    ingest_batch.run([BATCH_FIXTURE])

    # --- brand-new term: new "term"-pos word + sense ---
    new_word = coll.find_one(
        {"doc_type": "node", "node_type": "word", "properties.word": "graphene oxide"}
    )
    assert new_word is not None
    assert new_word["properties"]["pos"] == BATCH_POS
    assert new_word["vector"] is None  # not yet computed — s05 hasn't run
    new_sense = coll.find_one(
        {"doc_type": "node", "node_type": "sense", "properties.word": "graphene oxide"}
    )
    assert new_sense is not None
    assert new_sense["properties"]["doi"] == ["10.1/paperA"]
    assert new_sense["parent_edge_id"] is None  # orphan until s04 runs

    # --- near-duplicate gloss on "happy": merged into the EXISTING sense,
    #     no new sense created; existing sense gains the new doi ---
    n_senses_after = coll.count_documents(
        {"doc_type": "node", "node_type": "sense", "properties.word": "happy"}
    )
    assert n_senses_after == n_senses_before
    merged_sense = coll.find_one(
        {
            "doc_type": "node",
            "node_type": "sense",
            "properties.word": "happy",
            "properties.gloss": "Feeling or showing pleasure or contentment.",
        }
    )
    assert merged_sense is not None
    assert "10.1/paperA" in merged_sense["properties"]["doi"]

    # --- doi_bridges: reachable at the sense level for both terms ---
    bridge_coll = doi_bridge.get_bridge_collection(s)
    assert "10.1/paperA" in doi_bridge.dois_for_node(bridge_coll, new_sense["_id"])
    assert "10.1/paperA" in doi_bridge.dois_for_node(bridge_coll, merged_sense["_id"])

    # --- doi_bridges: propagated up the chain that already existed above
    #     "happy"'s merged sense (paired into an L15 BE by the earlier
    #     full-pipeline run) ---
    be_id = merged_sense["parent_edge_id"]
    assert be_id is not None
    assert "10.1/paperA" in doi_bridge.dois_for_node(bridge_coll, be_id)

    # --- ingest_batch wrote the touched-word-ids file for s05 scoping ---
    word_ids_file = tmp_state_dir / "batch_word_ids.json"
    assert word_ids_file.exists()
    touched = json.loads(word_ids_file.read_text())
    assert str(new_word["_id"]) in touched

    # --- weave the new orphan (graphene oxide) into the graph ---
    importlib.import_module(STAGES["04_l15_edges"]).run(["--force"])
    importlib.import_module(STAGES["05_word_vectors"]).run(
        ["--force", "--word-ids-file", str(word_ids_file)]
    )
    importlib.import_module(STAGES["06_l14_edges"]).run(["--force"])
    importlib.import_module(STAGES["07_orphan_reentry"]).run(["--force"])
    importlib.import_module(STAGES["08_metabary"]).run(["--force"])

    new_word_after = coll.find_one({"_id": new_word["_id"]})
    assert new_word_after["vector"] is not None
    assert new_word_after["parent_edge_id"] is not None  # absorbed via L14 orphan reentry

    # --- doi now reachable at the new word's L14 BaryEdge too ---
    assert "10.1/paperA" in doi_bridge.dois_for_node(
        bridge_coll, new_word_after["parent_edge_id"]
    )
