"""Document provenance index: a sibling collection to ``tensegrity``.

Every ingested sentence/title node already carries its source document id
directly (``properties.doc_id`` — see :mod:`tensegrity_pipeline.docs`). This
module is the *only* place document provenance is tracked above the node
level: rather than denormalizing a provenance field onto every BaryEdge /
MetaBary doc, each document gets one document here listing every
node/edge/triad whose ancestry includes it::

    {_id: <doc_id>, node_ids: [...], created_at, updated_at}

``doc_id`` is the t01 filename stem (``foo.md`` → ``foo``), so the key is the
human-readable document name.

This is a direct adaptation of :mod:`lib. doi_bridge` — the PoC's DOI
provenance index for academic batches — with the DOI swapped for a tensegrity
``doc_id``. The two call shapes carry over unchanged:

* ``register`` is the root case (document id already known — used when t03
  inserts nodes).
* ``propagate`` is the general case used at every BE/MB creation site: it
  reverse-looks-up which documents any of the new doc's constituents
  (``cm1_id``/``cm2_id``/bridge) already belong to, and adds the new doc under
  each. Because every constituent was itself registered here at its own
  creation time before it could become someone else's constituent, the same
  function works uniformly at every level.
* ``propagate_up_chain`` covers the late-arrival case (an existing node gains a
  document it didn't have before): register at the leaf, then walk
  ``parent_edge_id`` upward through whatever's already been built above it.

Safe to call unconditionally — the reverse lookup just finds nothing and
``propagate`` becomes a no-op query.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, UpdateOne
from pymongo.collection import Collection

from lib.db import get_client
from tensegrity_pipeline.config import TensegritySettings


def get_bridge_collection(settings: TensegritySettings) -> Collection:
    return get_client(settings)[settings.mongo_db][settings.doc_bridges_collection]


def ensure_indexes(coll: Collection) -> list[str]:
    """``_id`` uniqueness is automatic; ``node_ids`` needs a multikey index
    for reverse lookups (``docs_for_node`` / ``propagate``)."""
    return [coll.create_index([("node_ids", ASCENDING)])]


def _add_op(doc_id: str, node_id: Any, now: datetime) -> UpdateOne:
    return UpdateOne(
        {"_id": doc_id},
        {
            "$addToSet": {"node_ids": node_id},
            "$set": {"updated_at": now},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


def _add_node_to_docs(coll: Collection, doc_ids: list[str], node_id: Any) -> None:
    if not doc_ids:
        return
    now = datetime.now(timezone.utc)
    coll.bulk_write([_add_op(d, node_id, now) for d in doc_ids], ordered=False)


def register(coll: Collection, doc_ids: list[str], node_id: Any) -> None:
    """Root registration: the document is already known (node inserts)."""
    _add_node_to_docs(coll, doc_ids, node_id)


def register_nodes_for_docs(
    bridge_coll: Collection,
    main_coll: Collection,
    doc_ids: list[str] | set[str],
    *,
    batch: int = 1000,
) -> int:
    """Batched root registration for t03.

    Looks up the ``_id`` of every node under ``doc_ids`` and registers each
    under its own document in chunks, so a whole corpus costs a handful of
    round trips instead of one ``register`` per node. Returns the number of
    ``(document, node)`` pairs written.

    Idempotent (``$addToSet``), and it also re-covers nodes written by an
    earlier run of t03 — registration is derived from what is actually in the
    collection, not from this run's write results.
    """
    wanted = sorted(set(doc_ids))
    if not wanted:
        return 0

    pairs: list[tuple[str, Any]] = []
    for start in range(0, len(wanted), batch):
        chunk = wanted[start : start + batch]
        for doc in main_coll.find(
            {"doc_type": "node", "properties.doc_id": {"$in": chunk}},
            {"_id": 1, "properties.doc_id": 1},
        ):
            doc_id = (doc.get("properties") or {}).get("doc_id")
            if doc_id:
                pairs.append((doc_id, doc["_id"]))

    if not pairs:
        return 0
    now = datetime.now(timezone.utc)
    for start in range(0, len(pairs), batch):
        bridge_coll.bulk_write(
            [_add_op(doc_id, node_id, now) for doc_id, node_id in pairs[start : start + batch]],
            ordered=False,
        )
    return len(pairs)


def propagate(coll: Collection, new_id: Any, constituent_ids: list[Any]) -> None:
    """Union the documents already attached to any constituent, register
    ``new_id`` under each. Called right after every BE/MB write."""
    if not constituent_ids:
        return
    docs = [d["_id"] for d in coll.find({"node_ids": {"$in": constituent_ids}}, {"_id": 1})]
    _add_node_to_docs(coll, docs, new_id)


def propagate_up_chain(
    bridge_coll: Collection, main_coll: Collection, leaf_id: Any, new_doc_ids: list[str]
) -> None:
    """An existing node gained document(s) it didn't have before (a merge or a
    corrected re-ingest). Register at ``leaf_id`` itself, then walk
    ``parent_edge_id`` upward through whatever's already been built above it,
    registering at every level — that structure predates this document's
    arrival at the leaf.

    Works uniformly for a node, a BaryEdge, or a MetaBary as the starting
    ``leaf_id``, since every doc type carries ``parent_edge_id``.

    Not called by the current stage set: t03's upsert key embeds ``doc_id``
    (``<doc_id>:p<n>:s<n>``), so a node's document never changes today. Kept
    for parity with :mod:`lib.doi_bridge` and for the merge path a future
    re-ingest/absorb stage would take.
    """
    if not new_doc_ids:
        return
    _add_node_to_docs(bridge_coll, new_doc_ids, leaf_id)
    current_id = leaf_id
    seen: set[Any] = {leaf_id}
    while True:
        doc = main_coll.find_one({"_id": current_id}, {"parent_edge_id": 1})
        parent_id = (doc or {}).get("parent_edge_id")
        if parent_id is None or parent_id in seen:
            break
        _add_node_to_docs(bridge_coll, new_doc_ids, parent_id)
        seen.add(parent_id)
        current_id = parent_id


def docs_for_node(coll: Collection, node_id: Any) -> list[str]:
    """Reverse lookup for query-time use: which documents does this node trace to."""
    return [d["_id"] for d in coll.find({"node_ids": node_id}, {"_id": 1})]


def docs_for_nodes(coll: Collection, node_ids: list[Any]) -> dict[Any, list[str]]:
    """Batched reverse lookup for several nodes at once (e.g. a search result
    page) — one query instead of one per node. A document's ``node_ids`` array
    can span many unrelated docs, so this fans each matched document back out
    only to the requested ids it actually contains, not its full list.
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
