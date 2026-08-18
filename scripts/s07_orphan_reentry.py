"""Absorb orphan L14 word CMs into nearest existing L14 BE.

L15 orphan re-entry happens earlier, inside s04_l15_edges (it must
precede s05_word_vectors). This stage handles L14 only.

Each orphan word is paired with its nearest existing L14 BE; the new BE
inherits ``edge_type`` / ``type_vector`` / ``q`` from that partner (no new
embedding call) — see CLAUDE.md Stage 6.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib import doi_bridge
from lib.bary_vec import compute_bary_vec
from lib.db import get_collection
from lib.docs import baryedge
from scripts._base import bootstrap, finish

STAGE = "07_orphan_reentry"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    bridge_coll = doi_bridge.get_bridge_collection(settings)
    log.info("start processed=%d dry_run=%s", cp.processed, args.dry_run)

    orphan_ids: list = []
    OV: np.ndarray = np.empty((2_000_000, settings.embed_dim), dtype=np.float32)
    for i, doc in enumerate(coll.find(
        {"doc_type": "node", "node_type": "word", "level": 14,
         "parent_edge_id": None, "vector": {"$ne": None}},
        {"_id": 1, "vector": 1},
    )):
        orphan_ids.append(doc["_id"])
        OV[i] = doc["vector"]
    n_orphans = len(orphan_ids)
    OV = OV[:n_orphans]

    # Vectors only in this pass — metadata (type_vector etc.) is fetched below
    # just for the BEs that actually win a match. Materializing type_vector
    # (768 raw Python floats) for every existing L14 BE here would hold tens
    # of GB for the full BE pool, when at most n_orphans of them are ever used.
    be_ids: list = []
    n_bes_count = coll.count_documents({"doc_type": "baryedge", "level": 14})
    BEV: np.ndarray = np.empty((n_bes_count, settings.embed_dim), dtype=np.float32)
    for i, doc in enumerate(coll.find(
        {"doc_type": "baryedge", "level": 14},
        {"_id": 1, "vector": 1},
    )):
        be_ids.append(doc["_id"])
        BEV[i] = doc["vector"]
    n_bes = len(be_ids)
    BEV = BEV[:n_bes]

    log.info("L14 orphans=%d existing L14 BEs=%d", n_orphans, n_bes)

    if not n_orphans or not n_bes:
        cp.total = n_orphans
        if not args.dry_run:
            finish(cp, settings, log)
        return

    # Chunked nearest-BE search: full OV@BEV.T would be ~(300K×150K×4 B)=180 GB.
    # Process orphans in chunks so the intermediate sims matrix stays small.
    CHUNK = 1024
    best_bi = np.empty(n_orphans, dtype=np.int64)
    for start in range(0, n_orphans, CHUNK):
        end = min(start + CHUNK, n_orphans)
        best_bi[start:end] = np.argmax(OV[start:end] @ BEV.T, axis=1)

    # Now that we know which BEs actually won a match, fetch metadata
    # (type_vector, edge_type, q, accumulated_weight) for just those —
    # a small fraction of the full BE pool.
    used_be_ids = [be_ids[bi] for bi in sorted(set(best_bi.tolist()))]
    be_meta: dict = {
        doc["_id"]: doc
        for doc in coll.find(
            {"_id": {"$in": used_be_ids}},
            {"edge_type": 1, "type_vector": 1, "q": 1, "accumulated_weight": 1},
        )
    }

    # Stream inserts: building all 1.35M docs before inserting would hold ~66 GB of
    # Python-list float data in RAM (_vec() expands each 768-dim vector to a list of
    # Python floats at 32 bytes each × 2 vectors × 1.35M docs). Process batch_n at a time.
    batch_n = args.batch_size or settings.batch_size
    n_written = 0
    if not args.dry_run:
        now = datetime.now(timezone.utc)
        batch_docs: list[dict] = []
        batch_oids: list = []
        batch_partner_ids: list = []
        for idx in range(n_orphans):
            oid = orphan_ids[idx]
            bi = int(best_bi[idx])
            partner = be_meta[be_ids[bi]]
            tv = np.asarray(partner["type_vector"], dtype=np.float32)
            q = float(partner["q"])
            acc_w = float(partner.get("accumulated_weight", q))
            bv = compute_bary_vec(OV[idx], BEV[bi], tv, q)
            batch_docs.append(
                baryedge(oid, be_ids[bi], 14, bv, q,
                         accumulated_weight=acc_w,
                         edge_type=partner.get("edge_type"), type_vector=tv,
                         source="inferred", confidence=q)
            )
            batch_oids.append(oid)
            batch_partner_ids.append(be_ids[bi])
            if len(batch_docs) >= batch_n:
                res = coll.insert_many(batch_docs)
                ups = [UpdateOne({"_id": o}, {"$set": {"parent_edge_id": e, "updated_at": now}})
                       for o, e in zip(batch_oids, res.inserted_ids, strict=True)]
                coll.bulk_write(ups, ordered=False)
                for o, p, e in zip(batch_oids, batch_partner_ids, res.inserted_ids, strict=True):
                    doi_bridge.propagate(bridge_coll, e, [o, p])
                n_written += len(batch_docs)
                batch_docs = []
                batch_oids = []
                batch_partner_ids = []
        if batch_docs:
            res = coll.insert_many(batch_docs)
            ups = [UpdateOne({"_id": o}, {"$set": {"parent_edge_id": e, "updated_at": now}})
                   for o, e in zip(batch_oids, res.inserted_ids, strict=True)]
            coll.bulk_write(ups, ordered=False)
            for o, p, e in zip(batch_oids, batch_partner_ids, res.inserted_ids, strict=True):
                doi_bridge.propagate(bridge_coll, e, [o, p])
            n_written += len(batch_docs)

    cp.processed = n_written if not args.dry_run else n_orphans
    cp.total = n_orphans
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
