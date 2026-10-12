"""t05_title_vectors — derive L14 title vectors from their L15 BaryEdges.

    v(title) = normalize( Σ v(BE_i) + Σ v(orphan_sentence_j) )

Mirrors the PoC word-vector stage (s05 / lib.bary_vec.word_vector): no
embedding call — a title's vector is the centroid of the BaryEdges that parent
its sentences, plus the vectors of its still-orphan sentences. Sentence→title
membership comes from the ``properties.section_idx`` metadata written by t01,
not from pairwise edges.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib.vector import pack_vec, unpack_vec
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.db import get_collection
from tensegrity_pipeline.tg_vec import title_vector

STAGE = "05_title_vectors"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start dry_run=%s", args.dry_run)

    # L15 BaryEdge vectors, keyed by BE id.
    be_vecs: dict = {}
    for d in coll.find({"doc_type": "baryedge", "level": 15}, {"_id": 1, "vector": 1}):
        v = d.get("vector")
        if v is not None:
            be_vecs[d["_id"]] = unpack_vec(v)
    log.info("loaded %d L15 BaryEdge vectors", len(be_vecs))

    # Aggregate each title's contributing vectors from its sentences.
    title_be_ids: dict[tuple, set] = defaultdict(set)
    title_orphan_vecs: dict[tuple, list[np.ndarray]] = defaultdict(list)
    n_sentences = 0
    for s in coll.find(
        {"doc_type": "node", "node_type": "sentence", "level": 15},
        {"_id": 1, "parent_edge_id": 1, "vector": 1,
         "properties.doc_id": 1, "properties.section_idx": 1},
    ):
        props = s.get("properties", {})
        key = (props.get("doc_id"), props.get("section_idx"))
        if key[1] is None or key[1] < 0:
            continue
        n_sentences += 1
        parent = s.get("parent_edge_id")
        if parent is not None and parent in be_vecs:
            title_be_ids[key].add(parent)
        elif s.get("vector") is not None:
            title_orphan_vecs[key].append(unpack_vec(s["vector"]))
    log.info("aggregated %d sentences into titles", n_sentences)

    batch_n = args.batch_size or settings.batch_size
    ops: list[UpdateOne] = []
    now = datetime.now(timezone.utc)
    n = dead = 0
    for t in coll.find(
        {"doc_type": "node", "node_type": "title", "level": 14},
        {"_id": 1, "properties.doc_id": 1, "properties.section_idx": 1},
    ):
        props = t.get("properties", {})
        key = (props.get("doc_id"), props.get("section_idx"))
        be_ids = title_be_ids.get(key, set())
        be_vecs_for_title = [be_vecs[b] for b in be_ids if b in be_vecs]
        orphan_vecs = title_orphan_vecs.get(key, [])
        if not be_vecs_for_title and not orphan_vecs:
            dead += 1
            continue
        vec = title_vector(be_vecs_for_title, orphan_vecs)
        ops.append(
            UpdateOne({"_id": t["_id"]}, {"$set": {"vector": pack_vec(vec), "updated_at": now}})
        )
        n += 1
        if len(ops) >= batch_n:
            if not args.dry_run:
                coll.bulk_write(ops, ordered=False)
            ops = []
    if ops and not args.dry_run:
        coll.bulk_write(ops, ordered=False)

    log.info("computed %d L14 title vectors (dead=%d)", n, dead)
    cp.processed = n
    cp.total = n + dead
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
