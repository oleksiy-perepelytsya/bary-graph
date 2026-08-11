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
from scripts._base import bootstrap, finish

STAGE = "05_word_vectors"


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
        if cp.last_id:
            q["_id"] = {"$gt": ObjectId(cp.last_id)}
        total = coll.count_documents({"doc_type": "node", "node_type": "word", "level": 14})
        n = cp.processed

    batch_n = args.batch_size or settings.batch_size
    ops: list[UpdateOne] = []

    cur = coll.find(q, {"_id": 1, "properties": 1}).sort("_id", 1)
    for w in cur:
        if args.limit and n - cp.processed >= args.limit:
            break
        props = w["properties"]
        word, pos = props["word"], props["pos"]

        # Senses of W and their parent BEs.
        sense_docs = list(
            coll.find(
                {
                    "doc_type": "node",
                    "node_type": "sense",
                    "properties.word": word,
                    "properties.pos": pos,
                },
                {"_id": 1, "vector": 1, "parent_edge_id": 1},
            )
        )
        be_ids = {s["parent_edge_id"] for s in sense_docs if s.get("parent_edge_id")}
        be_vecs = [
            np.asarray(be["vector"], dtype=np.float32)
            for be in coll.find({"_id": {"$in": list(be_ids)}}, {"vector": 1})
        ]
        orphan_vecs = [
            np.asarray(s["vector"], dtype=np.float32)
            for s in sense_docs
            if not s.get("parent_edge_id")
        ]
        if not be_vecs and not orphan_vecs:
            continue  # word with zero senses (shouldn't happen post-stage-03)

        vec = word_vector(be_vecs, orphan_vecs)

        ops.append(
            UpdateOne(
                {"_id": w["_id"]},
                {"$set": {"vector": vec.tolist(), "updated_at": datetime.now(timezone.utc)}},
            )
        )
        if not args.dry_run:
            doi_bridge.propagate(bridge_coll, w["_id"], [s["_id"] for s in sense_docs])
        n += 1
        cp.last_id = str(w["_id"])
        if len(ops) >= batch_n:
            if not args.dry_run:
                coll.bulk_write(ops, ordered=False)
            ops = []
            if not scoped:
                cp.processed = n
                cp_mod.save(cp, settings)
            log.info("… %d/%d word vectors", n, total)

    if ops and not args.dry_run:
        coll.bulk_write(ops, ordered=False)

    log.info("computed %d/%d L14 word vectors", n, total)
    if scoped:
        return  # scoped runs don't touch the shared stage checkpoint

    cp.processed = n
    cp.total = total
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
