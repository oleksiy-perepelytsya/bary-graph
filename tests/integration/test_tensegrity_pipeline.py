"""End-to-end smoke test for the tensegrity pipeline (t01 → t10).

Runs the real stage entrypoints (via ``tensegrity_pipeline._base.main``) against
an isolated ``barygraph_test_`` DB with deterministic fake embeddings and a fake
tev1 decision model, then asserts the graph invariants the pipeline promises:

* L15 sentence nodes and L14 title nodes exist with the right levels,
* L14 titles carry *derived* vectors (t05) rather than the t03 placeholder,
* L15/L14 BaryEdges carry a tev1 role + probability,
* at least one MetaBary is formed at L13 with a role,
* every node/BaryEdge/MetaBary is reachable from its source document through
  the ``doc_bridges`` provenance index.

Skips automatically when no project Mongo is configured (see the
``mongo_test_db`` fixture).
"""

from __future__ import annotations

import pytest

from tensegrity_pipeline import doc_bridge
from tensegrity_pipeline._base import main
from tensegrity_pipeline.config import TensegritySettings

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

    # --- document provenance (doc_bridges) ---
    # One entry per ingested document; t03 registers the nodes and
    # t04/t06/t07/t08/t09 propagate it up to every edge and MetaBary.
    bridge = doc_bridge.get_bridge_collection(TensegritySettings.load())
    entries = list(bridge.find({}))
    assert [e["_id"] for e in entries] == ["neural_nets"]
    registered = set(entries[0]["node_ids"])
    graph_docs = {d["_id"] for d in coll.find({}, {"_id": 1})}
    assert graph_docs, "pipeline produced no documents"
    assert graph_docs <= registered, "every graph doc must trace to its source document"
    assert doc_bridge.docs_for_node(bridge, l15_edges[0]["_id"]) == ["neural_nets"]
    assert doc_bridge.docs_for_node(bridge, mbs[0]["_id"]) == ["neural_nets"]


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


def test_multi_doc_provenance_partitions_by_document(tens_env_vars, mongo_test_db):
    """Two documents in one run → two provenance entries, and every graph doc
    traces to the document(s) it came from."""
    tmp = tens_env_vars
    # Enough sentences that at least one positive-cosine pair is certain with the
    # fake embedder's near-orthogonal random vectors (Q_MIN_L15=0.0 here).
    alpha_lines = "\n".join(f"Alpha sentence number {k} about topic {k}." for k in range(6))
    beta_lines = "\n".join(f"Beta sentence number {k} about subject {k}." for k in range(6))
    alpha = tmp / "alpha.md"
    alpha.write_text(f"# Alpha Doc\n\n{alpha_lines}\n", encoding="utf-8")
    beta = tmp / "beta.md"
    beta.write_text(f"# Beta Doc\n\n{beta_lines}\n", encoding="utf-8")

    assert main(["01_parse", "--doc", str(alpha), "--doc", str(beta)]) == 0
    assert main(["02_embed"]) == 0
    assert main(["03_insert_nodes"]) == 0
    assert main(["04_l15_edges"]) == 0

    coll = mongo_test_db
    bridge = doc_bridge.get_bridge_collection(TensegritySettings.load())
    entries = {d["_id"]: set(d["node_ids"]) for d in bridge.find({})}
    assert set(entries) == {"alpha", "beta"}

    # Nodes live under exactly their own document — no cross-contamination.
    for node in coll.find({"doc_type": "node"}):
        doc_id = node["properties"]["doc_id"]
        assert node["_id"] in entries[doc_id]
        for other in entries:
            if other != doc_id:
                assert node["_id"] not in entries[other]

    # Every edge is covered by at least one document (cross-doc edges by both).
    edges = list(coll.find({"doc_type": "baryedge"}))
    assert edges, "no L15 edges formed"
    for edge in edges:
        assert any(edge["_id"] in ids for ids in entries.values())

