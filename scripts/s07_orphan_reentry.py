"""Absorb orphan L14 word CMs into nearest existing L14 BE (bounded).

L15 orphan re-entry happens earlier, inside s04_l15_edges (it must
precede s05_word_vectors). This stage handles L14 only.

Each orphan word is paired with its nearest existing L14 BE; the new BE
inherits ``edge_type`` / ``type_vector`` / ``q`` from that partner (no new
embedding call) — see CLAUDE.md Stage 6. Multiple orphans may absorb into
the same partner BE (each picks its own nearest partner independently),
matching the original exact-argmax semantics.

This stage doubles as the post-s07b coverage sweep: after
s07b_pair_orphans drains every _id window, ~5 % of words per window are left
unpaired (plus band-tail words) because s07b applies a strict 0.70 cosine
floor. Re-running this stage over the remaining orphans absorbs *all* of
them (no similarity floor), so every word node ends up parented. Raise the
cap with ``--limit N`` / ``S07_ORPHAN_LIMIT`` (default 100 000).

All-build fixes (patched 2026-09-18):
  - NO DOI/provenance propagation (user decision; per-edge reverse lookups
    were also the write throttle). Writes: insert_many acknowledged +
    w=0 parent stamps, batch 2900 by default (S07_WRITE_BATCH / --batch-size).
  - NO count_documents: mongot stalls on doc_type/level fat-counts on the
    35M-doc collection. Both pools stream from the proven pure-``_id``
    primitive (8-way id-sliced ranges, ``hint(_id_)``, client-side filters).
    Orphan rows go to OV/OVP disk memmaps ((limit, embed_dim) sparse up
    front, truncated to the real count). The L14 BE pool is streamed as
    projected + L2-normalised rows into a binary file reopened as an exact-size
    memmap — a ~5M-BE pool never materialises in RAM.
  - Nearest-BE search is HNSW on the projected pool (k=1, no mark_deleted)
    above ANN_THRESHOLD, exact chunked argmax below — the plain OV@BEV.T
    argmax against a ~5M-BE pool would be ~1e16 MACs (~a day of CPU).
  - The BE-pool scan filters ``doc_type == baryedge`` and
    ``source != structural`` (words that fall above the BE floor must not
    leak into the pool as fake partners).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne
from pymongo.write_concern import WriteConcern

from lib.bary_vec import compute_bary_vec
from lib.db import get_collection
from lib.docs import baryedge
from lib.match import (
    ANN_EF_CONSTRUCTION,
    ANN_M,
    ANN_THRESHOLD,
    MATCH_DIM,
    _gaussian_projector,
)
from lib.vector import unpack_vec
from scripts._base import bootstrap, finish

_log = logging.getLogger(__name__)

STAGE = "07_orphan_reentry"

S07_MMAP_PATH = os.environ.get("S07_MMAP_PATH", "/storage/bary/s07_OV.mmap")
# Projected L14-BE pool binary file (reopened as an exact-size memmap).
S07_BEV_PATH = os.environ.get("S07_BEV_PATH", "/storage/bary/s07_be_pool.bin")
# Word _id block for this build — same bounds s06 uses (words are contiguous
# in _id space; first word 6a9a2b40…, block ends before 6a9a53…).
S07_WORD_LO = os.environ.get("S07_WORD_LO", "6a9a2b400000000000000000")
S07_WORD_HI = os.environ.get("S07_WORD_HI", "6a9a53000000000000000000")
S07_ORPHAN_LIMIT = int(os.environ.get("S07_ORPHAN_LIMIT", "100000"))
S07_LOAD_WORKERS = int(os.environ.get("S07_LOAD_WORKERS", "8"))
S07_WRITE_BATCH = int(os.environ.get("S07_WRITE_BATCH", "2900"))
S07_STAMPS_UNACK = os.environ.get("S07_STAMPS_UNACK", "1") not in ("0", "false")
# Nearest-BE query ef — k=1, so a generous ef keeps the top-1 close to exact.
_S07_NN_EF = 400

# L14 BE band floor for this build (s06/s07/s07b BEs all minted after it).
_EDGE_LO_DEFAULT = ObjectId.from_datetime(
    datetime(2026, 9, 17, 11, 50, tzinfo=timezone.utc))
_S07_BE_LO = os.environ.get("S07_BE_LO")  # test override only
_S07_BE_HI = os.environ.get("S07_BE_HI")  # test override only


def _id_int(oid: ObjectId) -> int:
    return int.from_bytes(oid.binary, "big")


def _split_ranges(lo: ObjectId, hi: ObjectId, n: int) -> list[tuple[ObjectId, ObjectId]]:
    lo_i = _id_int(lo)
    hi_i = _id_int(hi)
    span = (hi_i - lo_i) // n
    out = []
    for k in range(n):
        a = lo_i + k * span
        b = lo_i + (k + 1) * span if k < n - 1 else hi_i
        out.append((ObjectId(a.to_bytes(12, "big")),
                    ObjectId(b.to_bytes(12, "big"))))
    return out


def _load_orphans(coll, limit: int, embed_dim: int, OV, OVP, proj,
                  log: logging.Logger) -> tuple[list, int]:
    """Stream up to ``limit`` unparented L14 words into OV/OVP memmaps.

    Returns (orphan_ids, n_skipped). Each worker walks one _id slice of the
    word block; rows land under a lock so the memmap fill is one shared
    stream (order within the cap is irrelevant for nearest-BE absorption).
    """
    ids: list = []
    lock = threading.Lock()
    n = 0
    skipped = 0
    fields = {"_id": 1, "vector": 1, "parent_edge_id": 1,
              "doc_type": 1, "node_type": 1, "level": 1}

    def _scan(lo, hi):
        nonlocal n, skipped
        q = {"_id": {"$gte": lo, "$lt": hi}}
        for doc in coll.find(q, fields).sort("_id", 1).hint("_id_").batch_size(10_000):
            if doc.get("doc_type") != "node" or doc.get("node_type") != "word" \
                    or doc.get("level") != 14 or doc.get("vector") is None \
                    or doc.get("parent_edge_id") is not None:
                skipped += 1
                continue
            row = unpack_vec(doc["vector"])
            rp = row @ proj.T
            norm = float(np.linalg.norm(rp))
            rp = rp / norm if norm else rp
            with lock:
                if n >= limit:
                    break
                ids.append(doc["_id"])
                OV[n] = row
                OVP[n] = rp
                n += 1
                if n % 500_000 == 0:
                    log.info("  %d orphans loaded", n)

    ranges = _split_ranges(ObjectId(S07_WORD_LO), ObjectId(S07_WORD_HI),
                           S07_LOAD_WORKERS)
    with ThreadPoolExecutor(max_workers=S07_LOAD_WORKERS) as ex:
        list(ex.map(lambda r: _scan(*r), ranges))
    return ids, skipped


def _stream_be_pool(coll, embed_dim: int, proj, out_path: Path,
                    log: logging.Logger) -> list:
    """Stream the L14 BE pool as projected-normalised rows into a binary file.

    Returns the BE _ids (parallel to the rows written). The caller reopens
    ``out_path`` as an exact-size (n, MATCH_DIM) memmap. No count_documents.
    """
    lo = ObjectId(_S07_BE_LO) if _S07_BE_LO else _EDGE_LO_DEFAULT
    if _S07_BE_HI:
        hi = ObjectId(_S07_BE_HI)
    else:
        max_id = coll.find_one(sort=[("_id", -1)], projection={"_id": 1})
        hi = max_id["_id"] if max_id else None
    if hi is None or _id_int(hi) <= _id_int(lo):
        return []
    ids: list = []
    lock = threading.Lock()
    n = 0
    fields = {"_id": 1, "vector": 1, "level": 1, "source": 1, "doc_type": 1}

    log.info("streaming L14 BE pool band [%s, %s)", lo, hi)
    ranges = _split_ranges(lo, hi, S07_LOAD_WORKERS)
    with open(out_path, "wb") as f:
        def _scan(a, b):
            nonlocal n
            q = {"_id": {"$gte": a, "$lt": b}}
            for doc in coll.find(q, fields).sort("_id", 1).hint("_id_").batch_size(10_000):
                if doc.get("doc_type") != "baryedge" or doc.get("level") != 14 \
                        or doc.get("vector") is None or doc.get("source") == "structural":
                    continue
                row = unpack_vec(doc["vector"])
                rp = row @ proj.T
                norm = float(np.linalg.norm(rp))
                rp = rp / norm if norm else rp
                with lock:
                    ids.append(doc["_id"])
                    f.write(rp.tobytes())
                    n += 1
                    if n % 500_000 == 0:
                        log.info("  %d L14 BEs in pool", n)

        with ThreadPoolExecutor(max_workers=S07_LOAD_WORKERS) as ex:
            list(ex.map(lambda r: _scan(*r), ranges))
    return ids


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    limit = args.limit if args.limit is not None else S07_ORPHAN_LIMIT
    if limit < 1:
        log.info("orphan limit < 1 (%d) — nothing to do", limit)
        cp.processed = 0
        cp.total = 0
        if not args.dry_run:
            finish(cp, settings, log)
        return

    embed_dim = settings.embed_dim
    proj = _gaussian_projector(embed_dim)
    log.info("start processed=%d dry_run=%s orphan_limit=%d embed_dim=%d MATCH_DIM=%d",
             cp.processed, args.dry_run, limit, embed_dim, MATCH_DIM)

    ov_path = Path(S07_MMAP_PATH)
    ovp_path = Path(f"{S07_MMAP_PATH}_P")
    bev_path = Path(S07_BEV_PATH)
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (ov_path, ovp_path, bev_path):
        p.unlink(missing_ok=True)

    try:
        OV = np.memmap(ov_path, mode="w+", dtype=np.float32,
                       shape=(limit, embed_dim))
        OVP = np.memmap(ovp_path, mode="w+", dtype=np.float32,
                        shape=(limit, MATCH_DIM))
        orphan_ids, n_skipped = _load_orphans(
            coll, limit, embed_dim, OV, OVP, proj, log)
        n_orphans = len(orphan_ids)
        if n_orphans == 0:
            log.info("no L14 orphans found (skipped=%d)", n_skipped)
            cp.processed = 0
            cp.total = 0
            if not args.dry_run:
                finish(cp, settings, log)
            return
        OV = OV[:n_orphans]
        OVP = OVP[:n_orphans]
        log.info("L14 orphans=%d (skipped=%d)", n_orphans, n_skipped)

        be_ids = _stream_be_pool(coll, embed_dim, proj, bev_path, log)
        n_bes = len(be_ids)
        if n_bes == 0:
            log.info("no L14 BE pool — nothing to absorb into")
            cp.processed = 0
            cp.total = n_orphans
            if not args.dry_run:
                finish(cp, settings, log)
            return
        log.info("L14 BE pool=%d", n_bes)

        # Nearest-BE per orphan (k=1, no mark_deleted → independent absorption).
        if n_bes <= ANN_THRESHOLD:
            BVP = np.memmap(bev_path, dtype=np.float32, mode="r",
                            shape=(n_bes, MATCH_DIM))
            CHUNK = 1024
            best_bi = np.empty(n_orphans, dtype=np.int64)
            for start in range(0, n_orphans, CHUNK):
                end = min(start + CHUNK, n_orphans)
                best_bi[start:end] = np.argmax(
                    OVP[start:end].copy() @ BVP.T, axis=1)
            del BVP
        else:
            try:
                import hnswlib
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("hnswlib required for ANN nearest-BE search") from e
            BVP = np.memmap(bev_path, dtype=np.float32, mode="r",
                            shape=(n_bes, MATCH_DIM))
            index = hnswlib.Index(space="cosine", dim=MATCH_DIM)
            index.init_index(max_elements=max(n_bes, 16),
                             ef_construction=ANN_EF_CONSTRUCTION, M=ANN_M)
            log.info("building HNSW on %d BEs (ef_c=%d M=%d)",
                     n_bes, ANN_EF_CONSTRUCTION, ANN_M)
            index.add_items(BVP)
            log.info("HNSW built")
            del BVP
            bev_path.unlink(missing_ok=True)
            index.set_ef(_S07_NN_EF)
            best_bi = np.empty(n_orphans, dtype=np.int64)
            for start in range(0, n_orphans, 16_384):
                end = min(start + 16_384, n_orphans)
                labs, _ = index.knn_query(
                    np.ascontiguousarray(OVP[start:end]), k=1)
                best_bi[start:end] = labs[:, 0]
            del index
        del OVP

        bev_path.unlink(missing_ok=True)

        used_be_ids = [be_ids[bi] for bi in sorted(set(best_bi.tolist()))]
        log.info("winning partners=%d", len(used_be_ids))
        # Chunk the $in lookup — a single 2M-id query would exceed the 16 MB
        # BSON document limit.
        be_meta: dict = {}
        _FIELDS = {"vector": 1, "edge_type": 1, "type_vector": 1,
                   "q": 1, "accumulated_weight": 1}
        for s in range(0, len(used_be_ids), 50_000):
            chunk = used_be_ids[s:s + 50_000]
            be_meta.update(
                {doc["_id"]: doc
                 for doc in coll.find({"_id": {"$in": chunk}}, _FIELDS)}
            )
        pvec = {pid: unpack_vec(meta["vector"]) for pid, meta in be_meta.items()
                if meta.get("vector") is not None}

        batch_n = args.batch_size or S07_WRITE_BATCH
        n_written = 0
        if not args.dry_run:
            now = datetime.now(timezone.utc)
            stamps_coll = coll.with_options(
                write_concern=WriteConcern(w=0)) if S07_STAMPS_UNACK else coll
            batch_docs: list[dict] = []
            batch_oids: list = []
            for idx in range(n_orphans):
                oid = orphan_ids[idx]
                bi = int(best_bi[idx])
                partner = be_meta[be_ids[bi]]
                tv = unpack_vec(partner["type_vector"])
                q = float(partner["q"])
                acc_w = float(partner.get("accumulated_weight", q))
                bv = compute_bary_vec(OV[idx], pvec[be_ids[bi]], tv, q)
                batch_docs.append(
                    baryedge(oid, be_ids[bi], 14, bv, q,
                             accumulated_weight=acc_w,
                             edge_type=partner.get("edge_type"), type_vector=tv,
                             source="inferred", confidence=q)
                )
                batch_oids.append(oid)
                if len(batch_docs) >= batch_n:
                    res = coll.insert_many(batch_docs)
                    ups = [UpdateOne({"_id": o},
                                     {"$set": {"parent_edge_id": e, "updated_at": now}})
                           for o, e in zip(batch_oids, res.inserted_ids, strict=True)]
                    stamps_coll.bulk_write(ups, ordered=False)
                    n_written += len(batch_docs)
                    log.info("  written %d (%d/%d)", n_written, idx + 1, n_orphans)
                    batch_docs = []
                    batch_oids = []
            if batch_docs:
                res = coll.insert_many(batch_docs)
                ups = [UpdateOne({"_id": o},
                                 {"$set": {"parent_edge_id": e, "updated_at": now}})
                       for o, e in zip(batch_oids, res.inserted_ids, strict=True)]
                stamps_coll.bulk_write(ups, ordered=False)
                n_written += len(batch_docs)

        if args.dry_run:
            log.info("dry-run: would absorb %d orphan words into %d existing BEs",
                     n_orphans, n_bes)
        else:
            log.info("absorbed %d orphan words into %d existing BEs",
                     n_written, n_bes)
        cp.processed = n_written if not args.dry_run else n_orphans
        cp.total = n_orphans
        if not args.dry_run:
            finish(cp, settings, log)
    finally:
        for p in (ov_path, ovp_path, bev_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                log.warning("could not remove memmap file %s", p)


if __name__ == "__main__":
    run()
