"""The цех paper shelf: sibling collection of unclaimed academic papers.

Agents choose what to read by will, never from an ordered queue, so the only
query this module exposes for selection is a randomized ``$sample`` of
``status == 'available'`` documents — deliberately no ranking, no topic
recommendation, no recency ordering at read time (the fetcher inserts newest
first, but ``$sample`` ignores insertion order).

Lifecycle: fetcher upserts ``available`` docs keyed by arXiv id → an agent
``claim``s one (mandatory one-line reason = provocation record + interest
profile) → ``mark_processed`` after term ingestion closes the loop. Claims are
append-only history in ``claims``: forced concurrent claims (deliberate
same-paper convergence reads) and post-processing re-reads are recorded, never
erased.

The clean ``available → claimed`` transition is atomic via
``find_one_and_update``; the held/idempotent/forced branches read-then-write,
which is safe enough for three sequential cron-driven agents.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, ReturnDocument, UpdateOne
from pymongo.collection import Collection

from lib.config import Settings
from lib.db import get_client

COLLECTION_NAME = "papers"


def get_collection(settings: Settings) -> Collection:
    client = get_client(settings)
    return client[settings.mongo_db][COLLECTION_NAME]


def ensure_indexes(coll: Collection) -> list[str]:
    return [
        coll.create_index([("status", ASCENDING)]),
        coll.create_index([("claims.author", ASCENDING)]),
        coll.create_index([("fetched_at", ASCENDING)]),
    ]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def make_paper_doc(
    *,
    arxiv_id: str,
    title: str,
    abstract: str,
    authors: list[str],
    primary_category: str,
    categories: list[str],
    published: str,
    updated: str,
    link: str,
    doi: str | None,
) -> dict[str, Any]:
    """Shelf document. Content fields go through ``$setOnInsert`` on upsert so
    re-fetches never mutate an already-shelved paper."""
    ts = _now()
    return {
        "_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "primary_category": primary_category,
        "categories": categories,
        "published": published,
        "updated": updated,
        "link": link,
        "doi": doi,
        "status": "available",
        "claims": [],
        "touched_word_ids": [],
        "fetched_at": ts,
    }


def upsert_many(coll: Collection, docs: list[dict[str, Any]]) -> tuple[int, int]:
    """Insert-if-absent. Returns (n_new, n_already_shelved)."""
    if not docs:
        return 0, 0
    ops = [
        UpdateOne({"_id": d["_id"]}, {"$setOnInsert": d}, upsert=True)
        for d in docs
    ]
    res = coll.bulk_write(ops, ordered=False)
    n_new = res.upserted_count
    return n_new, len(docs) - n_new


def available_count(coll: Collection) -> int:
    return coll.count_documents({"status": "available"})


def sample_available(coll: Collection, n: int) -> list[dict[str, Any]]:
    """Randomized slice of the unclaimed shelf — THE volitional-choice query."""
    return list(coll.aggregate([
        {"$match": {"status": "available"}},
        {"$sample": {"size": n}},
    ]))


def claim(
    coll: Collection, paper_id: str, author: str, reason: str, *, force: bool = False
) -> dict[str, Any]:
    """Append a claim event; returns {outcome, ...}.

    Outcomes:
      claimed      — clean available→claimed transition
      already_yours— same author re-claiming their own held paper (idempotent)
      held         — another author holds it; retry needs force=True
      reclaimed    — forced claim while held (convergence read; recorded)
      reopened     — forced claim of a processed paper (re-read; recorded)
      missing      — unknown paper_id
    """
    reason = reason.strip()
    entry = {"author": author, "reason": reason, "at": _now(), "forced": bool(force)}

    fresh = coll.find_one_and_update(
        {"_id": paper_id, "status": "available"},
        {"$push": {"claims": entry}, "$set": {"status": "claimed"}},
        return_document=ReturnDocument.AFTER,
    )
    if fresh is not None:
        return {"outcome": "claimed", "title": fresh["title"]}

    doc = coll.find_one({"_id": paper_id})
    if doc is None:
        return {"outcome": "missing"}

    claims = doc.get("claims") or []
    last = claims[-1] if claims else None
    holder = last["author"] if last else None

    if doc.get("status") == "claimed":
        if holder == author:
            return {"outcome": "already_yours", "title": doc["title"],
                    "previous_reason": last["reason"]}
        if not force:
            return {"outcome": "held", "title": doc["title"], "held_by": holder}
        coll.update_one(
            {"_id": paper_id},
            {"$push": {"claims": {**entry, "forced": True}}},
        )
        return {"outcome": "reclaimed", "title": doc["title"], "previously_held_by": holder}

    # status == 'processed'
    if not force:
        return {"outcome": "processed", "title": doc["title"],
                "processed_by": doc.get("processed_by")}
    coll.update_one(
        {"_id": paper_id},
        {"$push": {"claims": {**entry, "forced": True}}, "$set": {"status": "claimed"}},
    )
    return {"outcome": "reopened", "title": doc["title"]}


def mark_processed(
    coll: Collection, paper_id: str, author: str, touched_word_ids: list[Any]
) -> None:
    coll.update_one(
        {"_id": paper_id},
        {
            "$set": {
                "status": "processed",
                "processed_by": author,
                "processed_at": _now(),
            },
            "$addToSet": {"touched_word_ids": {"$each": [str(w) for w in touched_word_ids]}},
        },
    )
