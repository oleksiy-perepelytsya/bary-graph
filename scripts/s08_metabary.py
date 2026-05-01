"""Form L13 triads and recurse upward toward L1.

For each unparented bridge BE at level L-1, find two unparented BEs at
level L with mutual cos > ``meta_bary_cos_threshold``. Form a MetaBary at
L-2 using the Born-rule q_MB and set ``parent_edge_id`` on all three.
Stop when a pass produces zero new triads.

Safeguard: refuses to run if any L≤13 baryedge already exists.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib.bary_vec import compute_metabary_vec, level_factor
from lib.db import get_collection
from lib.docs import metabary
from lib.match import ANN_EF, ANN_EF_CONSTRUCTION, ANN_M, ANN_THRESHOLD, greedy_unique_match, top_k_pairs
from scripts._base import bootstrap, finish

_log = __import__("logging").getLogger(__name__)

STAGE = "08_metabary"


def _load_unparented_bes(coll, level: int) -> tuple[list, list[dict], np.ndarray]:
    """Return (ids, meta_list, V) for unparented BEs at level, streaming into pre-alloc numpy."""
    ids: list = []
    meta: list[dict] = []
    V = np.empty((2_000_000, 768), dtype=np.float32)
    for i, doc in enumerate(coll.find(
        {"doc_type": "baryedge", "level": level, "parent_edge_id": None},
        {"_id": 1, "vector": 1, "accumulated_weight": 1},
    )):
        ids.append(doc["_id"])
        meta.append({"_id": doc["_id"], "accumulated_weight": doc["accumulated_weight"]})
        V[i] = doc["vector"]
    n = len(ids)
    return ids, meta, V[:n]


def _form_level(coll, child_level: int, bridge_level: int, threshold: float,
                alpha: float, dry_run: bool) -> int:
    """Form MetaBary at ``child_level - 2`` from children@L and bridges@L-1."""
    child_ids, child_meta, CV = _load_unparented_bes(coll, child_level)
    bridge_ids, bridge_meta, BV = _load_unparented_bes(coll, bridge_level)
    if len(child_ids) < 2 or not bridge_ids:
        return 0

    # 1. Greedy mutual-cosine matching among children (similarity is between
    #    the two children, NOT bridge↔child — see CLAUDE.md Stage 7).
    pairs = greedy_unique_match(top_k_pairs(CV, min_score=threshold), threshold=threshold)
    if not pairs:
        return 0

    _log.info("greedy_unique_match: %d pairs from %d children", len(pairs), len(child_ids))

    # 2. Assign each pair the nearest unused bridge (by pair-centroid cosine).
    # For large bridge sets build an HNSW index to replace the O(n_bridges*dim)
    # full scan — otherwise bridge assignment would take hours at 1.4M bridges.
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
        _BRIDGE_K_EXPAND = min(n_bridges, _BRIDGE_K * 10)  # used in fallback
        # Phase 1 only needs ef >= k=50; Phase 2 expand needs ef >= k=500.
        _p1_ef = max(100, _BRIDGE_K + 1)
        _bridge_ef_q = max(_BRIDGE_K_EXPAND, _BRIDGE_K + 1)
        _log.info("building bridge HNSW index: n=%d ef_construction=%d M=%d k=%d p1_ef=%d p2_ef=%d",
                  n_bridges, _BRIDGE_EF_C, ANN_M, _BRIDGE_K, _p1_ef, _bridge_ef_q)
        bidx = hnswlib.Index(space="cosine", dim=BV.shape[1])
        bidx.init_index(max_elements=n_bridges, ef_construction=_BRIDGE_EF_C, M=ANN_M)
        bidx.add_items(BV)
        bidx.set_ef(_p1_ef)
        _log.info("bridge HNSW ready")

        # Pre-compute all pair centroids for a batched knn_query.
        # Chunked so we get progress logs and can see the query is making progress.
        dim = BV.shape[1]
        n_pairs = len(pairs)
        centroids = np.empty((n_pairs, dim), dtype=np.float32)
        for idx, (ci, cj, _) in enumerate(pairs):
            c = CV[ci] + CV[cj]
            nn = float(np.linalg.norm(c))
            centroids[idx] = (c / nn if nn else c).astype(np.float32)

        # Phase 1: chunked batched knn_query — parallelised per chunk, logged per chunk.
        _BRIDGE_CHUNK = 50_000
        _num_threads = os.cpu_count() or 4
        _log.info("bridge knn_query: %d pairs k=%d ef=%d chunk=%d threads=%d",
                  n_pairs, _BRIDGE_K, _p1_ef, _BRIDGE_CHUNK, _num_threads)
        all_labels_chunks: list[np.ndarray] = []
        phase1_ok = True
        t0 = time.monotonic()
        try:
            for start in range(0, n_pairs, _BRIDGE_CHUNK):
                end = min(start + _BRIDGE_CHUNK, n_pairs)
                chunk_labels, _ = bidx.knn_query(
                    centroids[start:end], k=_BRIDGE_K, num_threads=_num_threads)
                all_labels_chunks.append(chunk_labels)
                elapsed = time.monotonic() - t0
                rate = end / elapsed if elapsed > 0 else 0
                eta = (n_pairs - end) / rate if rate > 0 else float("inf")
                _log.info("bridge knn_query: %d/%d (%.0f%%) elapsed=%.0fs rate=%.0f/s eta=%.0fs",
                          end, n_pairs, 100 * end / n_pairs, elapsed, rate, eta)
        except RuntimeError:
            phase1_ok = False
        all_labels = np.vstack(all_labels_chunks) if (phase1_ok and all_labels_chunks) else None

        needs_expand: list[int] = []
        for idx, (ci, cj, q_pair) in enumerate(pairs):
            found = False
            if phase1_ok:
                for bi in all_labels[idx]:
                    bi = int(bi)
                    if bi not in bridge_taken:
                        bridge_taken.add(bi)
                        triads.append((ci, cj, bi, q_pair))
                        found = True
                        break
            else:
                sims = BV @ centroids[idx]
                top = np.argpartition(-sims, _K)[:_K]
                for bi in top[np.argsort(-sims[top])]:
                    bi = int(bi)
                    if bi not in bridge_taken:
                        bridge_taken.add(bi)
                        triads.append((ci, cj, bi, q_pair))
                        found = True
                        break
            if not found:
                needs_expand.append(idx)

        # Phase 2: expand query only for the rare pairs that exhausted top-50.
        if needs_expand:
            _log.info("expand batch knn_query: %d pairs k=%d ef=%d",
                      len(needs_expand), _BRIDGE_K_EXPAND, _bridge_ef_q)
            bidx.set_ef(_bridge_ef_q)
            exp_centroids = centroids[needs_expand]
            try:
                exp_labels, _ = bidx.knn_query(exp_centroids, k=_BRIDGE_K_EXPAND, num_threads=_num_threads)
                for i, idx in enumerate(needs_expand):
                    ci, cj, q_pair = pairs[idx]
                    for bi in exp_labels[i]:
                        bi = int(bi)
                        if bi not in bridge_taken:
                            bridge_taken.add(bi)
                            triads.append((ci, cj, bi, q_pair))
                            break
            except RuntimeError:
                for idx in needs_expand:
                    ci, cj, q_pair = pairs[idx]
                    sims = BV @ centroids[idx]
                    for bi in np.argsort(-sims):
                        bi = int(bi)
                        if bi not in bridge_taken:
                            bridge_taken.add(bi)
                            triads.append((ci, cj, bi, q_pair))
                            break
    else:
        for ci, cj, q_pair in pairs:
            centroid = CV[ci] + CV[cj]
            n = float(np.linalg.norm(centroid))
            centroid = centroid / n if n else centroid
            sims = BV @ centroid
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

    if not triads or dry_run:
        return len(triads)

    mb_level = child_level - 2
    now = datetime.now(timezone.utc)
    # Stream in batches: each metabary doc holds a 768-dim vector as a Python
    # list of floats (~24 KB); building all docs at once can exhaust RAM at scale.
    BATCH = 1000
    for start in range(0, len(triads), BATCH):
        batch = triads[start : start + BATCH]
        docs = []
        for ci, cj, bi, _ in batch:
            w1 = float(child_meta[ci]["accumulated_weight"])
            w2 = float(child_meta[cj]["accumulated_weight"])
            w3 = float(bridge_meta[bi]["accumulated_weight"])
            vec, q_mb_raw = compute_metabary_vec(CV[ci], CV[cj], BV[bi], w1, w2, w3)
            acc_w = q_mb_raw * level_factor(mb_level, alpha)
            docs.append(metabary(child_ids[ci], child_ids[cj], mb_level, vec,
                                 q_mb_raw, acc_w))
        res = coll.insert_many(docs)
        ups: list[UpdateOne] = []
        for (ci, cj, bi, _), eid in zip(batch, res.inserted_ids, strict=True):
            for cm_id in (child_ids[ci], child_ids[cj], bridge_ids[bi]):
                ups.append(UpdateOne({"_id": cm_id},
                                     {"$set": {"parent_edge_id": eid, "updated_at": now}}))
        coll.bulk_write(ups, ordered=False)
    return len(triads)


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    thr = settings.meta_bary_cos_threshold
    alpha = settings.level_factor_alpha
    log.info("start processed=%d dry_run=%s cos_threshold=%.2f alpha=%.2f",
             cp.processed, args.dry_run, thr, alpha)

    if not args.dry_run and not args.force:
        if coll.count_documents({"doc_type": "baryedge", "level": {"$lte": 13}}, limit=1):
            raise RuntimeError(
                "MetaBary docs (level ≤13) already present — re-running would "
                "double-parent BEs. Drop them and use --reset, or pass --force."
            )

    total = 0
    child_level = 15
    while child_level - 2 >= 1:
        bridge_level = child_level - 1
        n = _form_level(coll, child_level, bridge_level, thr, alpha, args.dry_run)
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
