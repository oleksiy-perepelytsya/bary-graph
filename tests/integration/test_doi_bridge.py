from __future__ import annotations

import pytest
from bson import ObjectId

from lib import doi_bridge

pytestmark = pytest.mark.integration


@pytest.fixture
def bridge_coll(mongo_test_db):
    from lib.config import Settings
    from lib.db import get_client

    s = Settings.load()
    coll = get_client(s)[s.mongo_db][s.mongo_doi_bridges_collection]
    yield coll


def test_register_creates_doc_with_node_id(bridge_coll):
    doi_bridge.register(bridge_coll, ["10.1/a"], "sense-1")
    doc = bridge_coll.find_one({"_id": "10.1/a"})
    assert doc is not None
    assert doc["node_ids"] == ["sense-1"]


def test_register_is_idempotent_addtoset(bridge_coll):
    doi_bridge.register(bridge_coll, ["10.1/a"], "sense-1")
    doi_bridge.register(bridge_coll, ["10.1/a"], "sense-1")
    doc = bridge_coll.find_one({"_id": "10.1/a"})
    assert doc["node_ids"] == ["sense-1"]


def test_propagate_unions_dois_from_constituents(bridge_coll):
    doi_bridge.register(bridge_coll, ["10.1/a"], "sense-1")
    doi_bridge.register(bridge_coll, ["10.1/b"], "sense-2")
    doi_bridge.propagate(bridge_coll, "be-1", ["sense-1", "sense-2"])
    assert set(doi_bridge.dois_for_node(bridge_coll, "be-1")) == {"10.1/a", "10.1/b"}


def test_propagate_no_op_when_no_constituent_has_doi(bridge_coll):
    doi_bridge.propagate(bridge_coll, "be-kaikki", ["sense-plain-1", "sense-plain-2"])
    assert doi_bridge.dois_for_node(bridge_coll, "be-kaikki") == []
    assert bridge_coll.count_documents({}) == 0


def test_propagate_chain_visible_at_every_level(bridge_coll, mongo_test_db):
    """Build a tiny 3-level chain by hand (sense -> L15 BE -> L13 MB) via
    parent_edge_id, propagate a doi at each construction step, and confirm
    dois_for_node resolves it at all three levels."""
    coll = mongo_test_db
    sense_id = ObjectId()
    be_id = ObjectId()
    mb_id = ObjectId()

    coll.insert_one({"_id": sense_id, "doc_type": "node", "node_type": "sense",
                      "parent_edge_id": be_id, "properties": {"doi": ["10.1/x"]}})
    doi_bridge.register(bridge_coll, ["10.1/x"], sense_id)

    coll.insert_one({"_id": be_id, "doc_type": "baryedge", "level": 15,
                      "cm1_id": sense_id, "cm2_id": ObjectId(), "parent_edge_id": mb_id})
    doi_bridge.propagate(bridge_coll, be_id, [sense_id])

    coll.insert_one({"_id": mb_id, "doc_type": "baryedge", "level": 13,
                      "cm1_id": be_id, "cm2_id": ObjectId(), "parent_edge_id": None})
    doi_bridge.propagate(bridge_coll, mb_id, [be_id])

    assert doi_bridge.dois_for_node(bridge_coll, sense_id) == ["10.1/x"]
    assert doi_bridge.dois_for_node(bridge_coll, be_id) == ["10.1/x"]
    assert doi_bridge.dois_for_node(bridge_coll, mb_id) == ["10.1/x"]


def test_propagate_up_chain_adds_new_doi_to_existing_branch(bridge_coll, mongo_test_db):
    """A near-duplicate merge attaches a NEW doi to a sense that's already
    parented several levels up — propagate_up_chain must reach every level
    above it, not just the leaf."""
    coll = mongo_test_db
    sense_id = ObjectId()
    be_id = ObjectId()
    mb_id = ObjectId()
    coll.insert_one({"_id": sense_id, "doc_type": "node", "node_type": "sense",
                      "parent_edge_id": be_id, "properties": {"doi": ["10.1/old"]}})
    coll.insert_one({"_id": be_id, "doc_type": "baryedge", "level": 15,
                      "parent_edge_id": mb_id})
    coll.insert_one({"_id": mb_id, "doc_type": "baryedge", "level": 13,
                      "parent_edge_id": None})
    doi_bridge.register(bridge_coll, ["10.1/old"], sense_id)
    doi_bridge.register(bridge_coll, ["10.1/old"], be_id)
    doi_bridge.register(bridge_coll, ["10.1/old"], mb_id)

    doi_bridge.propagate_up_chain(bridge_coll, coll, sense_id, ["10.1/new"])

    for node_id in (sense_id, be_id, mb_id):
        dois = set(doi_bridge.dois_for_node(bridge_coll, node_id))
        assert dois == {"10.1/old", "10.1/new"}


def test_ensure_indexes_creates_multikey_index(bridge_coll):
    names = doi_bridge.ensure_indexes(bridge_coll)
    assert names
    info = bridge_coll.index_information()
    assert any("node_ids" in str(v.get("key")) for v in info.values())
