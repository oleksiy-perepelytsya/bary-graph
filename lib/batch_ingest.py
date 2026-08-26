"""Shared academic-term ingestion write path (CLI and MCP server).

Extracted from scripts/ingest_batch.py so the цех ``ingest_terms`` MCP tool
and the offline batch CLI run byte-identical logic: exact dedup → embed →
per-term existing-word match → near-duplicate merge or insert, with DOI
provenance registered at every new/merged node (lib.doi_bridge).

New docs may additionally be stamped with ``properties.author`` — testimony
tagging for agent-created senses/words. Merge paths deliberately leave the
existing node untouched apart from DOI bookkeeping: a merge is a provenance
event, not authorship of the pre-existing sense.

Always writes (no dry-run here — the CLI keeps its own dry-run accounting).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from lib.batch_merge import (
    dedupe_exact,
    find_existing_word_candidates,
    near_duplicate_sense,
    resolve_pos,
)
from lib.config import Settings
from lib.docs import sense_node, word_node
from lib.doi_bridge import propagate_up_chain, register


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _stamp_author(coll: Collection, doc_id: Any, author: str, now: datetime) -> None:
    coll.update_one(
        {"_id": doc_id},
        {"$set": {"properties.author": author, "updated_at": now}},
    )


def ingest_entries(
    coll: Collection,
    bridge_coll: Collection,
    embedder: Any,
    settings: Settings,
    entries: list[tuple],
    *,
    stamp_author: str | None = None,
) -> dict[str, Any]:
    """Ingest (ParsedWord, ParsedSense) pairs into the live graph.

    Returns {"n_after_exact_dedup", "n_new_words", "n_new_senses_existing_word",
    "n_merged_dup", "touched_word_ids"}.
    """
    entries = dedupe_exact(entries)

    vectors: list = []
    for chunk in _batched([ps.embed_text for _, ps in entries], settings.embed_batch_size):
        vectors.extend(embedder.embed(chunk))

    now = datetime.now(timezone.utc)
    n_new_words = n_new_senses_existing_word = n_merged_dup = 0
    touched_word_ids: list = []

    for (pw, ps), vec in zip(entries, vectors, strict=True):
        candidates = find_existing_word_candidates(coll, ps.word)

        if not candidates:
            sense_doc = sense_node(ps, vec)
            sense_res = coll.insert_one(sense_doc)
            word_doc = word_node(pw)
            word_res = coll.insert_one(word_doc)
            if stamp_author:
                _stamp_author(coll, sense_res.inserted_id, stamp_author, now)
                _stamp_author(coll, word_res.inserted_id, stamp_author, now)
            register(bridge_coll, ps.doi, sense_res.inserted_id)
            register(bridge_coll, ps.doi, word_res.inserted_id)
            touched_word_ids.append(word_res.inserted_id)
            n_new_words += 1
            continue

        target = resolve_pos(coll, vec, candidates) or candidates[0]
        word, pos = target["properties"]["word"], target["properties"]["pos"]

        dup_id = near_duplicate_sense(coll, word, pos, vec, settings.batch_dup_threshold)
        if dup_id is not None:
            coll.update_one(
                {"_id": dup_id},
                {
                    "$addToSet": {"properties.doi": {"$each": ps.doi}},
                    "$set": {"updated_at": now},
                },
            )
            propagate_up_chain(bridge_coll, coll, dup_id, ps.doi)
            propagate_up_chain(bridge_coll, coll, target["_id"], ps.doi)
            n_merged_dup += 1
            continue

        ps2 = dataclasses.replace(ps, word=word, pos=pos)
        sense_doc = sense_node(ps2, vec)
        sense_res = coll.insert_one(sense_doc)
        if stamp_author:
            _stamp_author(coll, sense_res.inserted_id, stamp_author, now)
        coll.update_one(
            {"_id": target["_id"]},
            {
                "$push": {"properties.sense_ids": ps2.sense_id},
                "$inc": {"surface": 1},
                "$set": {"updated_at": now},
            },
        )
        register(bridge_coll, ps2.doi, sense_res.inserted_id)
        touched_word_ids.append(target["_id"])
        n_new_senses_existing_word += 1

    return {
        "n_after_exact_dedup": len(entries),
        "n_new_words": n_new_words,
        "n_new_senses_existing_word": n_new_senses_existing_word,
        "n_merged_dup": n_merged_dup,
        "touched_word_ids": touched_word_ids,
    }
