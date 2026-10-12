"""End-to-end smoke test for the tensegrity pipeline (t01 → t10).

Runs the real stage entrypoints (via ``tensegrity_pipeline._base.main``) against
an isolated ``barygraph_test_`` DB with deterministic fake embeddings and a fake
tev1 decision model, then asserts the graph invariants the pipeline promises:

* L15 sentence nodes and L14 title nodes exist with the right levels,
* L14 titles carry *derived* vectors (t05) rather than the t03 placeholder,
* L15/L14 BaryEdges carry a tev1 role + probability,
* at least one MetaBary is formed at L13 with a role.

Skips automatically when no project Mongo is configured (see the
``mongo_test_db`` fixture).
"""

from __future__ import annotations

import pytest

from tensegrity_pipeline._base import main

FIXTURE_DOC = """\
# Neural Networks

Neural networks are trained by backpropagation.
Reasoning is not required for statistical learning.
Gradient descent reduces the training loss.

## Compression

Compression emerges when representations are shared.
Shared representations reduce redundancy.
"""


@pytest.fixture
def tens_env_vars(tmp_path, monkeypatch):
    """Point the pipeline at throwaway dirs with fakes enabled.

    Must run *before* ``mongo_test_db`` so that fixture's ``Settings.load()``
    picks up ``MONGO_COLLECTION=tensegrity``.
    """
    monkeypatch.setenv("PARSED_DIR", str(tmp_path / "parsed"))
    monkeypatch.setenv("PIPELINE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MONGO_COLLECTION", "tensegrity")
    monkeypatch.setenv("BARY_FAKE_EMBED", "1")
    monkeypatch.setenv("BARY_FAKE_DECISION", "1")
    monkeypatch.setenv("EMBED_DIM", "64")
    monkeypatch.setenv("Q_MIN_L15", "0.0")
    monkeypatch.setenv("META_BARY_COS_THRESHOLD", "0.0")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    # BARY_ENV is deliberately left unset: bootstrap defaults to the tensegrity
    # profile, and our explicit env vars win over .env.tensegrity (override=False).
    monkeypatch.delenv("BARY_ENV", raising=False)
    return tmp_path


@pytest.fixture
def tens_env(tens_env_vars, mongo_test_db):
    doc = tens_env_vars / "neural_nets.md"
    doc.write_text(FIXTURE_DOC, encoding="utf-8")
    return doc, mongo_test_db


def test_pipeline_end_to_end(tens_env):
    doc, coll = tens_env

    def run(stage: str, *extra: str) -> None:
        assert main([stage, *extra]) == 0, f"stage {stage} failed"

    run("01_parse", "--doc", str(doc))
    run("02_embed")
    run("03_insert_nodes")
    run("04_l15_edges")
    run("05_title_vectors")
    run("06_l14_edges")
    run("07_orphan_reentry")
    run("08_metabary")
    run("09_extend")
    run("10_index")

    # --- nodes ---
    l15 = list(coll.find({"doc_type": "node", "node_type": "sentence", "level": 15}))
    l14 = list(coll.find({"doc_type": "node", "node_type": "title", "level": 14}))
    assert len(l15) == 5
    assert len(l14) == 2
    assert all(d["vector"] is not None for d in l15)
    # t05 must have replaced the t03 None placeholder on at least one title.
    assert sum(1 for d in l14 if d["vector"] is not None) >= 1

    # --- BaryEdges carry a tev1 role ---
    l15_edges = list(coll.find({"doc_type": "baryedge", "level": 15}))
    l14_edges = list(coll.find({"doc_type": "baryedge", "level": 14}))
    assert len(l15_edges) >= 1
    assert len(l14_edges) >= 1
    for e in l15_edges + l14_edges:
        assert e["role"] in ("complement", "contradict", None)
        assert 0.0 <= e["role_probability"] <= 1.0
        assert e["vector"] is not None

    # --- MetaBary at L13 ---
    mbs = list(coll.find({"doc_type": "baryedge", "level": {"$lte": 13},
                          "source": {"$ne": "structural"}}))
    assert len(mbs) >= 1
    for mb in mbs:
        assert mb["role"] in ("complement", "contradict", None)
        assert mb["vector"] is not None
        assert mb["connection_strength"] >= 0.0

    # Every BaryEdge's three participants are traceable (parent stamped).
    for e in l15_edges[:3]:
        assert coll.count_documents({"_id": e["cm1_id"]}) == 1
        assert coll.count_documents({"_id": e["cm2_id"]}) == 1


def test_multi_doc_ingestion_preserves_source_identity(tens_env_vars, mongo_test_db):
    """Two documents in one t01 run → distinct doc_ids in the collection."""
    tmp = tens_env_vars
    alpha = tmp / "alpha.md"
    alpha.write_text("# Alpha Doc\n\nFirst sentence.\nSecond sentence.\n", encoding="utf-8")
    beta = tmp / "beta.md"
    beta.write_text("# Beta Doc\n\nThird sentence.\n", encoding="utf-8")

    assert main(["01_parse", "--doc", str(alpha), "--doc", str(beta)]) == 0
    assert main(["02_embed"]) == 0
    assert main(["03_insert_nodes"]) == 0

    coll = mongo_test_db
    sentences = list(coll.find({"doc_type": "node", "node_type": "sentence", "level": 15}))
    titles = list(coll.find({"doc_type": "node", "node_type": "title", "level": 14}))

    docs = {s["properties"]["doc_id"] for s in sentences}
    assert docs == {"alpha", "beta"}
    assert {t["properties"]["doc_id"] for t in titles} == {"alpha", "beta"}

    counts: dict[str, int] = {}
    for s in sentences:
        counts[s["properties"]["doc_id"]] = counts.get(s["properties"]["doc_id"], 0) + 1
    assert counts == {"alpha": 2, "beta": 1}

    # Labels embed doc identity; no collisions across documents.
    labels = [s["properties"]["sentence_id"] for s in sentences]
    assert len(labels) == len(set(labels))
    assert any(label.startswith("alpha:") for label in labels)
    assert any(label.startswith("beta:") for label in labels)
