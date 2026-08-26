"""Second-order testimony: model-to-model review of Structure MetaBarys.

Sibling collection, following the lib.doi_bridge pattern: social/epistemic
metadata about SMBs deliberately lives OUTSIDE the barygraph collection.
The substrate stays geometry plus minimal identity fields — approval never
touches vectors, never becomes a stored property of a bridge, and the
derived status below is computed at query time rather than written onto
the edge document.

Semantics:
- One document per review event; append-only, nothing is ever deleted.
- Each author's LATEST verdict on an edge is their live position; earlier
  ones remain as history.
- ``status_for`` derives the collective state:
    supported — ≥2 distinct endorsing authors spanning ≥2 model families,
                zero live challenges (family = signature prefix before '@';
                a human nickname is its own family)
    contested — ≥1 live challenge; disagreement is preserved, never voted
                away — a contested SMB is a research object, not a failure
    single_voice — exactly one endorser, not yet corroboration
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING
from pymongo.collection import Collection

from lib.config import Settings
from lib.db import get_client

COLLECTION_NAME = "smb_reviews"

VALID_VERDICTS = {"endorse", "challenge"}


def get_collection(settings: Settings) -> Collection:
    client = get_client(settings)
    return client[settings.mongo_db][COLLECTION_NAME]


def ensure_indexes(coll: Collection) -> list[str]:
    return [
        coll.create_index([("edge_id", ASCENDING)]),
        coll.create_index([("author", ASCENDING)]),
        coll.create_index([("edge_id", ASCENDING), ("author", ASCENDING)]),
    ]


def add(
    coll: Collection, edge_id: Any, author: str, verdict: str, note: str
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    doc = {
        "edge_id": edge_id,
        "author": author,
        "verdict": verdict,
        "note": note,
        "at": now,
    }
    res = coll.insert_one(doc)
    doc["_id"] = res.inserted_id
    return doc


def reviews_for(coll: Collection, edge_id: Any) -> list[dict[str, Any]]:
    """All review events for one edge, oldest first."""
    return list(
        coll.find({"edge_id": edge_id}, {"_id": 0}).sort("at", 1)
    )


def _family(author: str) -> str:
    return author.split("@", 1)[0] if "@" in author else author


def status_for(coll: Collection, edge_id: Any) -> dict[str, Any] | None:
    """Aggregate live positions per author and derive the collective state."""
    events = reviews_for(coll, edge_id)
    if not events:
        return None
    latest: dict[str, dict[str, Any]] = {}
    for ev in events:
        latest[ev["author"]] = ev
    endorsers = {a: ev for a, ev in latest.items() if ev["verdict"] == "endorse"}
    challengers = {a: ev for a, ev in latest.items() if ev["verdict"] == "challenge"}
    families = {_family(a) for a in endorsers}
    if challengers:
        status = "contested"
    elif len(endorsers) >= 2 and len(families) >= 2:
        status = "supported"
    elif len(endorsers) == 1:
        status = "single_voice"
    else:
        status = None
    return {
        "status": status,
        "endorsements": sorted(endorsers),
        "endorser_families": sorted(families),
        "challenges": sorted(challengers),
    }
