"""DOI provenance index: a sibling collection to ``barygraph``.

Every academic-batch sense carries its source DOI(s) directly
(``properties.doi`` on the L15 sense node — see lib.docs.sense_node). This
module is the *only* place DOI provenance is tracked above the sense level:
rather than denormalizing a ``dois`` field onto every BaryEdge/MetaBary doc,
each DOI gets one document here listing every node/edge/triad whose ancestry
includes it — ``{_id: <doi>, node_ids: [...]}``.

``register`` is the root case (dois already known — used when a new sense is
inserted). ``propagate`` is the general case used at every BE/MB/word
creation site: it reverse-looks-up which DOIs any of the new doc's
constituents (cm1_id/cm2_id/bridge, or a word's sense ids) already belong to,
and adds the new doc under each. Because every constituent was itself
registered here at its own creation time before it could become someone
else's constituent, the same function works uniformly at every level.

Safe to call unconditionally on pure-kaikki data (no batch content
anywhere in a document's ancestry) — the reverse lookup just finds nothing
and ``propagate`` becomes a no-op query.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, UpdateOne
from pymongo.collection import Collection

from lib.config import Settings
from lib.db import get_client

COLLECTION_NAME = "doi_bridges"


def get_bridge_collection(settings: Settings) -> Collection:
    return get_client(settings)[settings.mongo_db][settings.mongo_doi_bridges_collection]


def ensure_indexes(coll: Collection) -> list[str]:
    """``_id`` uniqueness is automatic; ``node_ids`` needs a multikey index
    for reverse lookups (``dois_for_node`` / ``propagate``)."""
    return [coll.create_index([("node_ids", ASCENDING)])]


def _add_node_to_dois(coll: Collection, dois: list[str], node_id: Any) -> None:
    if not dois:
        return
    now = datetime.now(timezone.utc)
    ops = [
        UpdateOne(
            {"_id": doi},
            {
                "$addToSet": {"node_ids": node_id},
                "$set": {"updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        for doi in dois
    ]
    coll.bulk_write(ops, ordered=False)


def register(coll: Collection, dois: list[str], node_id: Any) -> None:
    """Root registration: dois are already known (new sense/word inserts)."""
    _add_node_to_dois(coll, dois, node_id)


def propagate(coll: Collection, new_id: Any, constituent_ids: list[Any]) -> None:
    """Union the DOIs already attached to any constituent, register new_id
    under each. Called right after every BE/MB/word write."""
    if not constituent_ids:
        return
    dois = [d["_id"] for d in coll.find({"node_ids": {"$in": constituent_ids}}, {"_id": 1})]
    _add_node_to_dois(coll, dois, new_id)


def propagate_up_chain(
    bridge_coll: Collection, main_coll: Collection, leaf_id: Any, new_dois: list[str]
) -> None:
    """An existing node gained DOI(s) it didn't have before (near-duplicate
    sense merge). Register at leaf_id itself, then walk parent_edge_id
    upward through whatever's already been built above it, registering at
    every level — that structure predates this doi's arrival at the leaf.

    Works uniformly for a sense, a word, or a BaryEdge as the starting
    leaf_id, since every doc type carries parent_edge_id.
    """
    if not new_dois:
        return
    _add_node_to_dois(bridge_coll, new_dois, leaf_id)
    current_id = leaf_id
    seen: set[Any] = {leaf_id}
    while True:
        doc = main_coll.find_one({"_id": current_id}, {"parent_edge_id": 1})
        parent_id = (doc or {}).get("parent_edge_id")
        if parent_id is None or parent_id in seen:
            break
        _add_node_to_dois(bridge_coll, new_dois, parent_id)
        seen.add(parent_id)
        current_id = parent_id


def dois_for_node(coll: Collection, node_id: Any) -> list[str]:
    """Reverse lookup for query-time use: which DOIs does this node trace to."""
    return [d["_id"] for d in coll.find({"node_ids": node_id}, {"_id": 1})]


def dois_for_nodes(coll: Collection, node_ids: list[Any]) -> dict[Any, list[str]]:
    """Batched reverse lookup for several nodes at once (e.g. a search result
    page) — one query instead of one per node. A DOI's node_ids array can
    span many unrelated docs, so this fans each matched DOI doc back out only
    to the requested ids it actually contains, not its full node_ids list.
    """
    if not node_ids:
        return {}
    wanted = set(node_ids)
    out: dict[Any, list[str]] = {nid: [] for nid in node_ids}
    for doc in coll.find({"node_ids": {"$in": node_ids}}, {"node_ids": 1}):
        for nid in doc["node_ids"]:
            if nid in wanted:
                out[nid].append(doc["_id"])
    return out
