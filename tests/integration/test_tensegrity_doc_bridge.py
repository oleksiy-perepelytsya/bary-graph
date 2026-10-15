"""Integration tests for the tensegrity document-provenance index.

Mirrors ``tests/integration/test_doi_bridge.py`` (the PoC's DOI index) with the
key swapped from a paper DOI to a tensegrity ``doc_id`` — the t01 filename stem.
Everything runs against an isolated ``barygraph_test_*`` DB via the
``mongo_test_db`` fixture (which also owns the teardown).
"""

from __future__ import annotations

import pytest
from bson import ObjectId

from tensegrity_pipeline import doc_bridge

pytestmark = pytest.mark.integration


@pytest.fixture
def bridge_coll(mongo_test_db):
    from tensegrity_pipeline.config import TensegritySettings

    s = TensegritySettings.load()
    yield doc_bridge.get_bridge_collection(s)


def _node(coll, doc_id: str, sid: str):
    return coll.insert_one(
        {
            "doc_type": "node",
            "node_type": "sentence",
            "level": 15,
            "properties": {"doc_id": doc_id, "sentence_id": sid, "text": sid},
        }
    ).inserted_id


def test_register_creates_doc_with_node_id(bridge_coll):
    doc_bridge.register(bridge_coll, ["paper-a"], "sent-1")
    doc = bridge_coll.find_one({"_id": "paper-a"})
    assert doc is not None
    assert doc["node_ids"] == ["sent-1"]


def test_register_is_idempotent_addtoset(bridge_coll):
    doc_bridge.register(bridge_coll, ["paper-a"], "sent-1")
    doc_bridge.register(bridge_coll, ["paper-a"], "sent-1")
    doc = bridge_coll.find_one({"_id": "paper-a"})
    assert doc["node_ids"] == ["sent-1"]


def test_propagate_unions_docs_from_constituents(bridge_coll):
    doc_bridge.register(bridge_coll, ["paper-a"], "sent-1")
    doc_bridge.register(bridge_coll, ["paper-b"], "sent-2")
    doc_bridge.propagate(bridge_coll, "be-1", ["sent-1", "sent-2"])
    assert set(doc_bridge.docs_for_node(bridge_coll, "be-1")) == {"paper-a", "paper-b"}


def test_propagate_no_op_when_no_constituent_is_registered(bridge_coll):
    doc_bridge.propagate(bridge_coll, "be-plain", ["sent-plain-1", "sent-plain-2"])
    assert doc_bridge.docs_for_node(bridge_coll, "be-plain") == []
    assert bridge_coll.count_documents({}) == 0


def test_propagate_chain_visible_at_every_level(bridge_coll, mongo_test_db):
    """Build a tiny 3-level chain by hand (sentence -> L15 BE -> L13 MB) via
    parent_edge_id, propagate a document at each construction step, and confirm
    docs_for_node resolves it at all three levels."""
    coll = mongo_test_db
    sent_id = ObjectId()
    be_id = ObjectId()
    mb_id = ObjectId()

    coll.insert_one({"_id": sent_id, "doc_type": "node", "node_type": "sentence",
                     "parent_edge_id": be_id, "properties": {"doc_id": "paper-x"}})
    doc_bridge.register(bridge_coll, ["paper-x"], sent_id)

    coll.insert_one({"_id": be_id, "doc_type": "baryedge", "level": 15,
                     "cm1_id": sent_id, "cm2_id": ObjectId(), "parent_edge_id": mb_id})
    doc_bridge.propagate(bridge_coll, be_id, [sent_id])

    coll.insert_one({"_id": mb_id, "doc_type": "baryedge", "level": 13,
                     "cm1_id": be_id, "cm2_id": ObjectId(), "parent_edge_id": None})
    doc_bridge.propagate(bridge_coll, mb_id, [be_id])

    assert doc_bridge.docs_for_node(bridge_coll, sent_id) == ["paper-x"]
    assert doc_bridge.docs_for_node(bridge_coll, be_id) == ["paper-x"]
    assert doc_bridge.docs_for_node(bridge_coll, mb_id) == ["paper-x"]


def test_propagate_up_chain_adds_new_doc_to_existing_branch(bridge_coll, mongo_test_db):
    """An existing node gains a NEW document it didn't have before (the merge /
    corrected-re-ingest path) — propagate_up_chain must reach every level above
    it, not just the leaf."""
    coll = mongo_test_db
    sent_id = ObjectId()
    be_id = ObjectId()
    mb_id = ObjectId()
    coll.insert_one({"_id": sent_id, "doc_type": "node", "node_type": "sentence",
                     "parent_edge_id": be_id, "properties": {"doc_id": "paper-old"}})
    coll.insert_one({"_id": be_id, "doc_type": "baryedge", "level": 15,
                     "parent_edge_id": mb_id})
    coll.insert_one({"_id": mb_id, "doc_type": "baryedge", "level": 13,
                     "parent_edge_id": None})
    doc_bridge.register(bridge_coll, ["paper-old"], sent_id)
    doc_bridge.register(bridge_coll, ["paper-old"], be_id)
    doc_bridge.register(bridge_coll, ["paper-old"], mb_id)

    doc_bridge.propagate_up_chain(bridge_coll, coll, sent_id, ["paper-new"])

    for node_id in (sent_id, be_id, mb_id):
        assert set(doc_bridge.docs_for_node(bridge_coll, node_id)) == {
            "paper-old", "paper-new"
        }


def test_register_nodes_for_docs_covers_every_node_of_each_doc(bridge_coll, mongo_test_db):
    """The t03 root path: one batched call registers every node under its own
    document, with no cross-contamination between documents."""
    coll = mongo_test_db
    a1 = _node(coll, "a", "a:p0:s0")
    a2 = _node(coll, "a", "a:p0:s1")
    b1 = _node(coll, "b", "b:p0:s0")

    n = doc_bridge.register_nodes_for_docs(bridge_coll, coll, ["a", "b"])
    assert n == 3

    assert set(doc_bridge.docs_for_node(bridge_coll, a1)) == {"a"}
    assert set(doc_bridge.docs_for_node(bridge_coll, a2)) == {"a"}
    assert set(doc_bridge.docs_for_node(bridge_coll, b1)) == {"b"}

    # Idempotent: a second pass adds nothing.
    assert doc_bridge.register_nodes_for_docs(bridge_coll, coll, ["a", "b"]) == 3
    assert bridge_coll.find_one({"_id": "a"})["node_ids"] == [a1, a2]


def test_docs_for_nodes_batched_lookup(bridge_coll, mongo_test_db):
    """docs_for_nodes fans a shared document back out only to the requested ids,
    and returns an entry (possibly empty) for every id asked about."""
    coll = mongo_test_db
    a1 = _node(coll, "a", "a:p0:s0")
    a2 = _node(coll, "a", "a:p0:s1")
    b1 = _node(coll, "b", "b:p0:s0")
    doc_bridge.register_nodes_for_docs(bridge_coll, coll, ["a", "b"])

    unknown = ObjectId()
    out = doc_bridge.docs_for_nodes(bridge_coll, [a1, b1, unknown])
    assert set(out[a1]) == {"a"}
    assert set(out[b1]) == {"b"}
    assert out[unknown] == []
    assert a2 in bridge_coll.find_one({"_id": "a"})["node_ids"]


def test_ensure_indexes_creates_multikey_index(bridge_coll):
    names = doc_bridge.ensure_indexes(bridge_coll)
    assert names
    info = bridge_coll.index_information()
    assert any("node_ids" in str(v.get("key")) for v in info.values())
