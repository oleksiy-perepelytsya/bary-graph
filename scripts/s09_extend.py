"""Form MetaBary from orphan BaryEdges iteratively until convergence.

Picks up where s08 left off.  For each child level, sweeps the cosine
threshold from the base value (meta_bary_cos_threshold, default 0.9) down
to a per-level floor of ``0.9 - level × 0.01``:

  L15 floor = 0.75  (15 rounds of 0.01 relaxation)
  L14 floor = 0.76
  L13 floor = 0.77
  … and so on upward

Each child_level does ONE MongoDB fetch then sweeps all thresholds in
memory — 14× faster than re-fetching from MongoDB each threshold round.
Pairs are sorted by cosine descending so high-quality matches are greedy-
claimed first, producing identical results to the multi-round approach.

Repeats full descending passes until a complete pass produces zero new
triads across all levels.

No embedding calls.  All reads are from existing BE vectors; the only
writes are new MetaBary docs and parent_edge_id updates on existing BEs.
Safe on a CPU-only instance: the bridge HNSW index builds with hnswlib
(CPU) and the working vectors fit comfortably in 64 GB RAM.

Resumable: MongoDB parent_edge_id=None is the ground truth for which BEs
are still unparented.  The checkpoint records cumulative triads formed for
observability but does not gate which BEs are re-processed on resume.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib import checkpoint as cp_mod
from lib.bary_vec import compute_metabary_vec, level_factor
from lib.db import get_collection
from lib.docs import metabary
from lib.match import ANN_M, ANN_THRESHOLD, greedy_unique_match, top_k_pairs
from scripts._base import bootstrap, finish
from scripts.s08_metabary import _load_unparented_bes

_log = __import__("logging").getLogger(__name__)

STAGE = "09_extend"

_BASE_THR = 0.9


def _min_threshold(level: int) -> float:
    """Per-level floor: 0.9 - level × 0.01  (L15→0.75, L14→0.76, …)."""
    return round(_BASE_THR - level * 0.01, 2)


def _form_level_sweep(
    coll,
    child_level: int,
    bridge_level: int,
    base_thr: float,
    min_thr: float,
    alpha: float,
    dry_run: bool,
    outer_pass: int,
) -> int:
    """Load children+bridges once, sweep all thresholds in memory.

    Single MongoDB fetch per child_level instead of one per threshold round.
    Pair selection is cosine-descending greedy — identical results to the
    original per-round approach because high-cosine pairs are claimed first.
    """
    child_ids, child_meta, CV = _load_unparented_bes(coll, child_level)
    bridge_ids, bridge_meta, BV = _load_unparented_bes(coll, bridge_level)

    def _log_zeros() -> None:
        thr = base_thr
        while thr >= min_thr:
            _log.info(
                "pass %d L%d thr=%.2f: children@L%d bridges@L%d → 0 triads",
                outer_pass, child_level - 2, thr, child_level, bridge_level,
            )
            thr = round(thr - 0.01, 2)

    if len(child_ids) < 2 or not bridge_ids:
        _log_zeros()
        return 0

    # Single HNSW build at floor threshold — all candidate pairs in one query.
    all_pairs = list(top_k_pairs(CV, min_score=min_thr))
    pairs = greedy_unique_match(all_pairs, threshold=min_thr)

    if not pairs:
        _log_zeros()
        return 0

    # Bridge assignment: same logic as _form_level in s08_metabary.
    n_bridges = len(bridge_ids)
    _K = min(200, n_bridges)
    bridge_taken: set[int] = set()
    triads: list[tuple[int, int, int, float]] = []

    if n_bridges > ANN_THRESHOLD:
        import hnswlib

        _BRIDGE_EF_C = 100
        _BRIDGE_K = min(50, n_bridges)
        _bridge_ef_q = max(200, _BRIDGE_K * 4)
        _log.info(
            "building bridge HNSW index: n=%d ef_construction=%d M=%d k=%d",
            n_bridges, _BRIDGE_EF_C, ANN_M, _BRIDGE_K,
        )
        bidx = hnswlib.Index(space="cosine", dim=BV.shape[1])
        bidx.init_index(max_elements=n_bridges, ef_construction=_BRIDGE_EF_C, M=ANN_M)
        bidx.add_items(BV)
        bidx.set_ef(_bridge_ef_q)
        _log.info("bridge HNSW ready")

        for ci, cj, q_pair in pairs:
            centroid = CV[ci] + CV[cj]
            n_norm = float(np.linalg.norm(centroid))
            centroid = centroid / n_norm if n_norm else centroid
            found = False
            try:
                labels, _ = bidx.knn_query(centroid.reshape(1, -1), k=_BRIDGE_K)
                for bi in labels[0]:
                    bi = int(bi)
                    if bi not in bridge_taken:
                        bridge_taken.add(bi)
                        triads.append((ci, cj, bi, q_pair))
                        found = True
                        break
            except RuntimeError:
                pass
            if not found:
                for bi in (int(x) for x in np.argsort(-(BV @ centroid))):
                    if bi not in bridge_taken:
                        bridge_taken.add(bi)
                        triads.append((ci, cj, bi, q_pair))
                        break
    else:
        for ci, cj, q_pair in pairs:
            centroid = CV[ci] + CV[cj]
            n_norm = float(np.linalg.norm(centroid))
            centroid = centroid / n_norm if n_norm else centroid
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

    # Log per-threshold bucket counts for diagnostics.
    # Top bucket (base_thr) is open-ended [base_thr, 1.0]; lower buckets are [thr, thr+0.01).
    thr = base_thr
    while thr >= min_thr:
        lo = thr
        if thr == base_thr:
            count = sum(1 for _, _, _, q in triads if q >= lo)
        else:
            hi = round(thr + 0.01, 2)
            count = sum(1 for _, _, _, q in triads if lo <= q < hi)
        _log.info(
            "pass %d L%d thr=%.2f: children@L%d bridges@L%d → %d triads",
            outer_pass, child_level - 2, thr, child_level, bridge_level, count,
        )
        thr = round(thr - 0.01, 2)

    if dry_run:
        return len(triads)

    # Write MetaBary docs + update parent_edge_id in batches.
    mb_level = child_level - 2
    now = datetime.now(timezone.utc)
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
            docs.append(metabary(child_ids[ci], child_ids[cj], mb_level, vec, q_mb_raw, acc_w))
        res = coll.insert_many(docs)
        ups: list[UpdateOne] = []
        for (ci, cj, bi, _), eid in zip(batch, res.inserted_ids, strict=True):
            for cm_id in (child_ids[ci], child_ids[cj], bridge_ids[bi]):
                ups.append(UpdateOne({"_id": cm_id}, {"$set": {"parent_edge_id": eid, "updated_at": now}}))
        coll.bulk_write(ups, ordered=False)

    return len(triads)


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    base_thr = settings.meta_bary_cos_threshold
    alpha = settings.level_factor_alpha

    log.info(
        "start processed=%d dry_run=%s cos_threshold=%.2f alpha=%.2f",
        cp.processed, args.dry_run, base_thr, alpha,
    )

    total = cp.processed
    outer_pass = 0

    while True:
        outer_pass += 1
        pass_total = 0
        child_level = 15

        while child_level - 2 >= 1:
            bridge_level = child_level - 1
            min_thr = _min_threshold(child_level)

            n = _form_level_sweep(
                coll, child_level, bridge_level,
                base_thr, min_thr, alpha, args.dry_run,
                outer_pass,
            )
            pass_total += n
            log.info(
                "pass %d L%d sweep done: %d triads (children@L%d bridges@L%d)",
                outer_pass, child_level - 2, n, child_level, bridge_level,
            )

            if n == 0:
                break
            child_level -= 1

        total += pass_total
        log.info("pass %d complete: %d new MBs (cumulative=%d)", outer_pass, pass_total, total)

        cp.processed = total
        if not args.dry_run:
            cp_mod.save(cp, settings)

        if pass_total == 0:
            break

    log.info("converged after %d passes, %d total new MBs", outer_pass, total)
    cp.total = total
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
