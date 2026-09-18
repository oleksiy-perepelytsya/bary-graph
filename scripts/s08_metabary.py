"""Form L13 triads and recurse upward toward L1.

For each unparented bridge BE at level L-1, find two unparented BEs at
level L with mutual cos > ``meta_bary_cos_threshold``. Form a MetaBary at
L-2 using the Born-rule q_MB and set ``parent_edge_id`` on all three.
Stop when a pass produces zero new triads.

User decisions (2026-09-17):
  - No DOI/provenance propagation in this stage (the per-MB reverse lookups
    were the write throttle, and inferred MB triads over a pure-kaikki corpus
    carry no DOIs to propagate anyway).
  - ``source="structural"`` nodes (SMBs) are NOT consumed by this stage. They
    are created by the cognitive/SMB pipeline to be folded in on a LATER
    rerun, so every load path and the level<=13 safeguard excludes them —
    they keep ``parent_edge_id: None`` and stay untouched by the pipeline
    build. The ``_S08_TODAY_LO`` _id floor also excludes the current strays.

Scale/memory design: the L15 child pass is 12,051,296 BEs x 4096 x 4B
(~197 GB) and must never touch RAM. Each level's children and bridges are
loaded into DISK memmaps at full embed_dim (for write) plus a second memmap
pre-projected and normalized to MATCH_DIM (for matching/bridge selection), so
the big matrices stay file-backed and the anonymous peak is just the two
HNSW indexes, built sequentially: child index (~49 GB anon, dropped right
after greedy matching) then bridge index on the projected matrix (~10 GB
anon). Writes re-read the full-dim rows in BATCH chunks with
posix_fadvise(DONTNEED) afterwards, bounding write-phase RSS.

``--children-cap`` (env S08_CHILDREN_CAP) is a DEV-only knob: it caps the
pass-1 child pool (default: no cap = full pass) and forces a single-threaded
load scan, so the memmap lifecycle / matching / dry-run write can be smoke
tested in ~2 minutes before the full run.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne

from lib.bary_vec import compute_metabary_vec, level_factor
from lib.db import get_collection
from lib.docs import metabary
from lib.match import (
    ANN_M,
    ANN_THRESHOLD,
    MATCH_DIM,
    _gaussian_projector,
    greedy_unique_match,
    top_k_pairs,
)
from lib.vector import unpack_vec
from scripts._base import bootstrap, finish

_log = __import__("logging").getLogger(__name__)

STAGE = "08_metabary"

# L15 baryedges were minted by s04 (2026-09-04T06:00Z …) and sit below the
# s06/s07 L14 band; everything s08 consumes above L15 was minted from the
# s06 start onward, so a single _id floor covers all non-L15 children/bridges.
# Verified on the all-build: count({_id: [09-04T06:00Z, 09-17T11:50Z)}) =
# 12,051,299 ≈ 12,051,296 L15 unparented BEs (3 strays, client-filtered).
_S08_BARY_LO = ObjectId.from_datetime(datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc))
_S08_TODAY_LO = ObjectId.from_datetime(datetime(2026, 9, 17, 11, 50, tzinfo=timezone.utc))
_S08_LOAD_WORKERS = int(os.environ.get("S08_LOAD_WORKERS", "8"))
_S08_MMAP_DIR = os.environ.get("S08_MMAP_DIR", "/storage/bary")
_CHILDREN_CAP = os.environ.get("S08_CHILDREN_CAP")

# Pipeline MBs carry no ``source`` field (or None); structural SMBs carry
# source="structural" and must never be absorbed.
_SC = {"source": {"$ne": "structural"}}


def _cleanup_mmaps(paths: Sequence[Path | None], log: logging.Logger) -> None:
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not remove memmap file %s: %s", path, e)


def _split_custom_args(argv: Sequence[str] | None) -> tuple[list[str], int | None]:
    """Strip s08b-specific flags (--children-cap) so bootstrap's parser stays happy."""
    out: list[str] = []
    cap = None
    it = iter(argv or [])
    for a in it:
        if a == "--children-cap":
            cap = int(next(it))
        elif a.startswith("--children-cap="):
            cap = int(a.split("=", 1)[1])
        else:
            out.append(a)
    return out, cap


def _load_unparented_bes(coll, level: int, embed_dim: int, tag: str,
                         cap: int | None = None,
                         ) -> tuple[list, list[dict], np.ndarray, np.ndarray,
                                     Path | None, Path | None]:
    """Return (ids, meta_list, V, VP, V_path, VP_path) for unparented BEs/MBs.

    ``V`` is the full embed_dim disk memmap (needed for metabary vectors);
    ``VP`` is the projected + L2-normalised MATCH_DIM memmap (used for the
    child match and bridge selection, so the big full-dim matrix stays cold
    until write). ``cap`` (dev) bounds children and forces a single-threaded
    scan so a smoke run stops after a couple of minutes.

    mongot's planner stalls BOTH fat-filter counts and fat-projection filter
    finds on the big all-build collection (the ``count_documents`` below timed
    out at 120 s), so this uses the s06/s07 pure-``_id`` primitive: disjoint
    ``_id`` sub-ranges over the level's band, ``hint(_id_)`` cursors,
    client-side level/parent/vector/source filtering. There is NO count: the
    accepted rows stream as raw float32 into append-only binary files which
    are reopened as exact-size memmaps at the end, so the L15 child pass
    (12,051,296 × 4096 × 4B ≈ 197 GB) never touches RAM and needs no index.
    """
    band_lo, band_hi = (_S08_BARY_LO, _S08_TODAY_LO) if level == 15 \
        else (_S08_TODAY_LO, None)
    v_path = Path(_S08_MMAP_DIR) / f"s08_{tag}.mmap"
    vp_path = Path(_S08_MMAP_DIR) / f"s08_{tag}_P.mmap"
    v_path.parent.mkdir(parents=True, exist_ok=True)
    proj = _gaussian_projector(embed_dim)  # (MATCH_DIM, embed_dim), float32
    ids: list = []
    meta: list[dict] = []
    lock = threading.Lock()
    n = 0
    dropped = 0
    proj_fields = {"_id": 1, "doc_type": 1, "level": 1, "parent_edge_id": 1,
                   "vector": 1, "accumulated_weight": 1, "source": 1}

    def _scan(lo, hi):
        nonlocal n, dropped
        q = {"_id": {"$gte": lo}}
        if hi is not None:
            q["_id"]["$lt"] = hi
        for doc in coll.find(q, proj_fields).sort("_id", 1).hint("_id_").batch_size(10_000):
            if doc.get("doc_type") != "baryedge" or doc.get("level") != level \
                    or doc.get("parent_edge_id") is not None \
                    or doc.get("source") == "structural":
                dropped += 1
                continue
            vec = doc.get("vector")
            if vec is None:
                dropped += 1
                continue
            # unpack + projection are CPU-heavy: do them OUTSIDE the lock so 8
            # workers actually run in parallel (they used to serialize on the
            # lock, collapsing the bridge load to ~145 rows/s aggregate).
            row = unpack_vec(vec)
            rp = row @ proj.T
            norm = float(np.linalg.norm(rp))
            rp = rp / norm if norm else rp
            with lock:
                if cap is not None and n >= cap:
                    break
                ids.append(doc["_id"])
                meta.append({"_id": doc["_id"],
                             "accumulated_weight": doc["accumulated_weight"]})
                vf.write(row.tobytes())
                pf.write(rp.tobytes())
                n += 1
                if n % 500_000 == 0:
                    _log.info("  loaded %d unparented L%d (dropped=%d)", n, level, dropped)

    with open(v_path, "wb") as vf, open(vp_path, "wb") as pf:
        if cap is None and _S08_LOAD_WORKERS > 1 and band_hi is not None:
            lo_i = int.from_bytes(band_lo.binary, "big")
            hi_i = int.from_bytes(band_hi.binary, "big")
            span = (hi_i - lo_i) // _S08_LOAD_WORKERS
            ranges = []
            for k in range(_S08_LOAD_WORKERS):
                a = lo_i + k * span
                b = lo_i + (k + 1) * span if k < _S08_LOAD_WORKERS - 1 else hi_i
                ranges.append((ObjectId(a.to_bytes(12, "big")),
                               ObjectId(b.to_bytes(12, "big"))))
            with ThreadPoolExecutor(max_workers=_S08_LOAD_WORKERS) as ex:
                list(ex.map(lambda r: _scan(*r), ranges))
        else:
            _scan(band_lo, band_hi)
    if n == 0:
        _cleanup_mmaps([v_path, vp_path], _log)
        return [], [], np.empty((0, embed_dim), dtype=np.float32), \
            np.empty((0, MATCH_DIM), dtype=np.float32), None, None
    V = np.memmap(v_path, mode="r+", dtype=np.float32, shape=(n, embed_dim))
    VP = np.memmap(vp_path, mode="r+", dtype=np.float32, shape=(n, MATCH_DIM))
    _log.info("loaded %d unparented L%d BEs/MBs (dropped=%d, V %s, VP %s)",
              n, level, dropped, v_path, vp_path)
    return ids, meta, V, VP, v_path, vp_path


def _form_level(coll, child_level: int, bridge_level: int, threshold: float,
                alpha: float, dry_run: bool, embed_dim: int,
                children_cap: int | None = None) -> int:
    """Form MetaBary at ``child_level - 2`` from children@L and bridges@L-1."""
    child_ids, child_meta, CV, CVP, cv_path, cvp_path = _load_unparented_bes(
        coll, child_level, embed_dim, f"C{child_level}", cap=children_cap
    )
    bridge_ids, bridge_meta, BV, BVP, bv_path, bvp_path = _load_unparented_bes(
        coll, bridge_level, embed_dim, f"B{bridge_level}"
    )
    if len(child_ids) < 2 or not bridge_ids:
        _cleanup_mmaps([cv_path, cvp_path, bv_path, bvp_path], _log)
        return 0

    # 1. Greedy mutual-cosine matching among children (similarity is between
    #    the two children, NOT bridge↔child — see CLAUDE.md Stage 7). Runs on
    #    the projected memmap: dim == MATCH_DIM so top_k_pairs does not
    #    re-project into an anonymous matrix.
    pairs_iter = top_k_pairs(CVP, min_score=threshold)
    pairs = greedy_unique_match(pairs_iter, threshold=threshold)
    del pairs_iter
    gc.collect()  # free the child HNSW (anonymous, ~49 GB at 12M children)
    if not pairs:
        _cleanup_mmaps([cv_path, cvp_path, bv_path, bvp_path], _log)
        return 0

    _log.info("greedy_unique_match: %d pairs from %d children", len(pairs), len(child_ids))

    # 2. Assign each pair the nearest unused bridge (by projected centroid
    #    cosine). Centroids come from CVP rows (full-dim CV stays cold until
    #    write); the bridge HNSW is built on the projected memmap BVP.
    n_bridges = len(bridge_ids)
    _K = min(200, n_bridges)
    bridge_taken: set[int] = set()
    triads: list[tuple[int, int, int, float]] = []  # (ci, cj, bi, q_pair)

    if n_bridges > ANN_THRESHOLD:
        import hnswlib
        # Use ef_construction=100 regardless of ANN_EF_CONSTRUCTION env var:
        # ANN_EF_CONSTRUCTION=50 speeds up the child HNSW but is too sparse
        # to reliably satisfy knn_query when k is large.
        _BRIDGE_EF_C = 100
        # k=50 is sufficient — with 1.39M bridges and at most n_pairs taken,
        # the nearest untaken bridge is almost always in the top-50.
        _BRIDGE_K = min(50, n_bridges)
        _bridge_ef_q = max(200, _BRIDGE_K * 4)
        _log.info("building bridge HNSW index: n=%d dim=%d ef_construction=%d "
                  "M=%d k=%d",
                  n_bridges, BVP.shape[1], _BRIDGE_EF_C, ANN_M, _BRIDGE_K)
        bidx = hnswlib.Index(space="cosine", dim=BVP.shape[1])
        bidx.init_index(max_elements=n_bridges, ef_construction=_BRIDGE_EF_C, M=ANN_M)
        bidx.add_items(BVP)
        bidx.set_ef(_bridge_ef_q)
        _log.info("bridge HNSW ready")

        for ci, cj, q_pair in pairs:
            centroid = CVP[ci] + CVP[cj]
            n = float(np.linalg.norm(centroid))
            centroid = centroid / n if n else centroid
            found = False
            try:
                labels, _ = bidx.knn_query(centroid.reshape(1, -1), k=_BRIDGE_K)
                for bi in labels[0]:
                    bi = int(bi)
                    if bi not in bridge_taken:
                        bridge_taken.add(bi)
                        bidx.mark_deleted(bi)
                        triads.append((ci, cj, bi, q_pair))
                        found = True
                        break
            except RuntimeError:
                pass  # M=16 graph too sparse for this query; fall through to brute force
            if not found:
                # Brute-force fallback: covers both "all top-K taken" and hnswlib
                # RuntimeError (M=16 connectivity insufficient for this query point).
                for bi in (int(x) for x in np.argsort(-(BVP @ centroid))):
                    if bi not in bridge_taken:
                        bridge_taken.add(bi)
                        bidx.mark_deleted(bi)
                        triads.append((ci, cj, bi, q_pair))
                        break
        del bidx
        gc.collect()
    else:
        for ci, cj, q_pair in pairs:
            centroid = CVP[ci] + CVP[cj]
            n = float(np.linalg.norm(centroid))
            centroid = centroid / n if n else centroid
            sims = BVP @ centroid
            if len(sims) > _K:
                cands = np.argpartition(-sims, _K)[:_K]
                order: np.ndarray = cands[np.argsort(-sims[cands])]
            else:
                order = np.argsort(-sims)
            found = False
            for bi in order:
                if int(bi) not in bridge_taken:
                    bridge_taken.add(int(bi))
                    triads.append((ci, cj, int(bi), q_pair))
                    found = True
                    break
            if not found:
                for bi in np.argsort(-sims):
                    if int(bi) not in bridge_taken:
                        bridge_taken.add(int(bi))
                        triads.append((ci, cj, int(bi), q_pair))
                        break

    # Projected memmaps done: drop them (and their pages) before the write
    # phase, which re-reads only the full-dim V/BV rows it needs.
    _cleanup_mmaps([cvp_path, bvp_path], _log)

    if not triads or dry_run:
        _cleanup_mmaps([cv_path, bv_path], _log)
        return len(triads)

    mb_level = child_level - 2
    now = datetime.now(timezone.utc)
    # Stream in batches: each metabary doc holds an embed_dim vector as a Python
    # list of floats (~24 KB); building all docs at once can exhaust RAM at scale.
    BATCH = 1000
    cv_fd = os.open(cv_path, os.O_RDONLY) if cv_path else None
    bv_fd = os.open(bv_path, os.O_RDONLY) if bv_path else None
    try:
        for start in range(0, len(triads), BATCH):
            batch = triads[start : start + BATCH]
            docs = []
            for ci, cj, bi, _ in batch:
                w1 = float(child_meta[ci]["accumulated_weight"])
                w2 = float(child_meta[cj]["accumulated_weight"])
                w3 = float(bridge_meta[bi]["accumulated_weight"])
                vec, q_mb_raw = compute_metabary_vec(CV[ci], CV[cj], BV[bi],
                                                     w1, w2, w3)
                acc_w = q_mb_raw * level_factor(mb_level, alpha)
                docs.append(metabary(child_ids[ci], child_ids[cj], mb_level, vec,
                                     q_mb_raw, acc_w))
            res = coll.insert_many(docs)
            ups: list[UpdateOne] = []
            for (ci, cj, bi, _), eid in zip(batch, res.inserted_ids, strict=True):
                for cm_id in (child_ids[ci], child_ids[cj], bridge_ids[bi]):
                    ups.append(UpdateOne({"_id": cm_id},
                                         {"$set": {"parent_edge_id": eid,
                                                   "updated_at": now}}))
            coll.bulk_write(ups, ordered=False)
            # Evict the batch's full-dim pages so write-phase RSS stays bounded
            # (~the batch's rows, not the 197 GB child matrix).
            if cv_fd is not None:
                os.posix_fadvise(cv_fd, 0, 0, os.POSIX_FADV_DONTNEED)
            if bv_fd is not None:
                os.posix_fadvise(bv_fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        if cv_fd is not None:
            os.close(cv_fd)
        if bv_fd is not None:
            os.close(bv_fd)
    _cleanup_mmaps([cv_path, bv_path], _log)
    return len(triads)


def run(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    clean, cap = _split_custom_args(argv)
    settings, args, log, cp = bootstrap(STAGE, clean)
    coll = get_collection(settings)
    thr = settings.meta_bary_cos_threshold
    alpha = settings.level_factor_alpha
    children_cap = cap if cap is not None else (
        int(_CHILDREN_CAP) if _CHILDREN_CAP else None
    )
    log.info("start processed=%d dry_run=%s cos_threshold=%.2f alpha=%.2f "
             "children_cap=%s",
             cp.processed, args.dry_run, thr, alpha, children_cap)

    if not args.dry_run and not args.force:
        if coll.count_documents({"doc_type": "baryedge", "level": {"$lte": 13},
                                 **_SC}, limit=1):
            raise RuntimeError(
                "MetaBary docs (level ≤13, pipeline-built) already present — "
                "re-running would double-parent BEs. Drop them and use --reset, "
                "or pass --force. (source=\"structural\" SMBs are excluded.)"
            )

    total = 0
    child_level = 15
    while child_level - 2 >= 1:
        bridge_level = child_level - 1
        pass_cap = children_cap  # dev-only; None in production
        n = _form_level(coll, child_level, bridge_level, thr, alpha,
                        args.dry_run, settings.embed_dim, children_cap=pass_cap)
        log.info("L%d MetaBary: children@L%d bridges@L%d → %d triads",
                 child_level - 2, child_level, bridge_level, n)
        total += n
        if n == 0:
            break
        child_level -= 1

    cp.processed = total
    cp.total = total
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
