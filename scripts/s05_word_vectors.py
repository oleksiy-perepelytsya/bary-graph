"""Compute L14 word vectors from BE-centroids + orphan senses.

    v(W) = normalize( Σ v(BE_i) + Σ v(orphan_sense_j) )

No embedding call. Strict stage boundary: depends on the *finalized* L15
BE set from stage 04 (including orphan re-entry). See v0.5 §2.4.

``--word-ids-file`` (written by scripts.ingest_batch) scopes the recompute
to a specific set of word docs instead of scanning the whole L14 word
collection — a small academic-batch ingestion pass otherwise forces a full
recompute over every kaikki word every time. A scoped run does not touch
the shared stage checkpoint (it isn't a partial/resumable slice of the full
recompute, so persisting its processed/total counts there would corrupt the
"has 05_word_vectors completed" state used by the ordinary full-build path).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne

from lib import checkpoint as cp_mod
from lib import doi_bridge
from lib.bary_vec import word_vector
from lib.db import get_collection
from lib.vector import pack_vec, unpack_vec
from scripts._base import bootstrap, finish

STAGE = "05_word_vectors"

BE_CHUNK = 2000   # _id $in chunks for BE / orphan-sense vector fetches


def _process_window(coll, win: list[dict]) -> list[tuple[ObjectId, np.ndarray, list]]:
    """Batch BE + orphan-sense vector fetches for *win* (list of records).

    Returns a list of (word_id, vector, sense_ids) for every word whose
    vector can be computed (skips words where both be_vecs and orphan_vecs
    are empty — shouldn't happen post-stage-03).
    """
    want_be: set[ObjectId] = set()
    want_orph: set[ObjectId] = set()
    for rec in win:
        want_be |= rec["be_ids"]
        want_orph.update(rec["orphan_ids"])

    be_cache: dict[ObjectId, np.ndarray] = {}
    for i in range(0, len(want_be), BE_CHUNK):
        for be in coll.find(
            {"_id": {"$in": sorted(want_be)[i:i + BE_CHUNK]}},
            {"vector": 1},
        ):
            be_cache[be["_id"]] = unpack_vec(be["vector"])

    orph_cache: dict[ObjectId, np.ndarray] = {}
    for i in range(0, len(want_orph), BE_CHUNK):
        for s in coll.find(
            {"_id": {"$in": sorted(want_orph)[i:i + BE_CHUNK]}},
            {"vector": 1},
        ):
            orph_cache[s["_id"]] = unpack_vec(s["vector"])

    out: list[tuple[ObjectId, np.ndarray, list]] = []
    for rec in win:
        be_vecs = [be_cache[i] for i in rec["be_ids"] if i in be_cache]
        orphan_vecs = [orph_cache[i] for i in rec["orphan_ids"] if i in orph_cache]
        if not be_vecs and not orphan_vecs:
            continue
        vec = word_vector(be_vecs, orphan_vecs)
        out.append((rec["w_id"], vec, rec["sense_ids"]))
    return out


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    bridge_coll = doi_bridge.get_bridge_collection(settings)
    log.info("start processed=%d dry_run=%s word_ids_file=%s",
              cp.processed, args.dry_run, args.word_ids_file)

    scoped = bool(args.word_ids_file)
    if scoped:
        word_ids = [ObjectId(i) for i in json.loads(Path(args.word_ids_file).read_text())]
        q: dict = {"doc_type": "node", "node_type": "word", "level": 14,
                   "_id": {"$in": word_ids}}
        total = len(word_ids)
        n = 0
    else:
        q = {"doc_type": "node", "node_type": "word", "level": 14}
        total = coll.count_documents({"doc_type": "node", "node_type": "word", "level": 14})
        n = cp.processed

    batch_n = args.batch_size or settings.batch_size
    ops: list[UpdateOne] = []
    dois_active = not args.dry_run and bridge_coll.count_documents({}) > 0
    if not dois_active:
        log.info("doi_bridges absent/empty -> per-word DOI propagation skipped")

    # Enumeration: natural (physical) order. Sorting by _id (the original
    # design) forces the server to fully sort ~10M word docs — viable only
    # while physical order still aligns with _id. Once a 10M-doc bulk vector
    # pass has scrambled physical order, that sort cannot stream even its
    # first batch within the client socket timeout, and every resume aborts
    # with NetworkTimeout before processing anything. Natural order streams
    # immediately, and resume correctness holds because the skip below is by
    # _id VALUE (idempotent), not by stream position; the head is re-walked
    # cheaply (no sense lookups for skipped words). A compound
    # (doc_type, node_type, level, _id) index exists but this mongot treats
    # $gt/$range seeks are unreliable here, so stay with full natural scans.
    skip_to = ObjectId(cp.last_id) if (cp.last_id and not scoped) else None
    cur = coll.find(q, {"_id": 1}).batch_size(5000)
    # Head words (≤ last_id) are skipped by _id value before their properties
    # are ever needed, so stream only _id for the walk; fetch properties
    # lazily (a handful of tail documents) instead of dragging 10M IPA/
    # etymology blobs across the wire — ~30x faster resume walk.
    win: list[dict] = []
    run_count = 0
    # Words that can never carry a vector (no senses, or no vector-able source)
    # are skipped rather than counted in `n`; total() still counts them, so the
    # completion gate `n >= total` would never fire and `done` stays False
    # forever. Track them separately and gate on n + dead >= total. These are
    # deterministic across resumes, so the count is idempotent.
    dead = 0

    def _drain() -> None:
        nonlocal ops, n, win, run_count, dead
        computed = _process_window(coll, win)
        out_ids = set()
        for w_id, vec, sense_ids in computed:
            out_ids.add(w_id)
            ops.append(
                UpdateOne(
                    {"_id": w_id},
                    {"$set": {"vector": pack_vec(vec), "updated_at": datetime.now(timezone.utc)}},
                )
            )
            if dois_active:
                doi_bridge.propagate(bridge_coll, w_id, sense_ids)
            n += 1
            run_count += 1
            cp.last_id = str(w_id)
        dead += sum(1 for rec in win if rec["w_id"] not in out_ids)
        if len(ops) >= batch_n:
            if not args.dry_run:
                coll.bulk_write(ops, ordered=False)
            ops = []
            if not scoped and not args.dry_run:
                cp.processed = n
                cp_mod.save(cp, settings)
            log.info("… %d/%d word vectors", n, total)
        win = []

    for w in cur:
        if args.limit and run_count >= args.limit:
            break
        if skip_to is not None and w["_id"] <= skip_to:
            continue
        props = coll.find_one({"_id": w["_id"]}, {"properties": 1, "_id": 0})["properties"]
        word, pos = props["word"], props["pos"]
        lang = props.get("lang", "en")

        # Senses of W and their parent BEs (no vector payload — orphan senses
        # are rare and their vectors are fetched batched per window below).
        sense_docs = list(
            coll.find(
                {
                    "doc_type": "node",
                    "node_type": "sense",
                    "properties.word": word,
                    "properties.pos": pos,
                    "properties.lang": lang,
                },
                {"_id": 1, "parent_edge_id": 1},
            )
        )
        be_ids = {s["parent_edge_id"] for s in sense_docs if s.get("parent_edge_id")}
        orphan_ids = [s["_id"] for s in sense_docs if not s.get("parent_edge_id")]
        if not be_ids and not orphan_ids:
            dead += 1  # word with zero senses (shouldn't happen post-stage-03)
            continue
        win.append(
            {
                "w_id": w["_id"],
                "be_ids": be_ids,
                "orphan_ids": orphan_ids,
                "sense_ids": [s["_id"] for s in sense_docs],
            }
        )
        if len(win) >= batch_n:
            _drain()

    if win:
        _drain()
    if ops and not args.dry_run:
        coll.bulk_write(ops, ordered=False)

    log.info("computed %d/%d L14 word vectors (dead=%d)", n, total, dead)
    if scoped:
        return  # scoped runs don't touch the shared stage checkpoint

    cp.processed = n
    cp.total = total
    if not args.dry_run and n + dead >= total:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
