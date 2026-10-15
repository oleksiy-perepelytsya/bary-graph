"""t08_metabary — form L13 triads and recurse upward toward L1.

Mirrors s08's algorithm (unparented children at level L, unparented bridge at
L-1, mutual-cosine child matching, nearest-available-bridge assignment, Born-rule
``q_mb_raw``) with one tensegrity addition: the tev1 decision model classifies
each MetaBary's **bridge** into one of the kaikki relation types (Step IV), and
that role sign is applied to the bridge term of the MetaBary vector.

Differences from s08, by design:

* In-memory loading instead of disk memmaps — the pipeline targets document
  corpora, whose BE pools are orders of magnitude smaller than kaikki's L15 set.
* Bridge selection uses a seeded top-K window (``_BRIDGE_WINDOW``) with no
  brute-force rescue, matching s08's "ANN-window-only, else drop" rule so the
  children stay available for t09's relaxed rescue sweep.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import islice

import numpy as np
from pymongo import UpdateOne

from lib.bary_vec import level_factor
from lib.match import greedy_unique_match, top_k_pairs
from tensegrity_pipeline import doc_bridge
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.db import get_collection
from tensegrity_pipeline.docs import metabary
from tensegrity_pipeline.pairing import (
    classify_states,
    load_unparented_bes,
    text_resolver,
)
from tensegrity_pipeline.tev1 import get_decision_model
from tensegrity_pipeline.tg_vec import compute_metabary_vec

STAGE = "08_metabary"
_BRIDGE_WINDOW = 50

_log = logging.getLogger(__name__)


def _form_level(
    coll,
    child_level: int,
    bridge_level: int,
    threshold: float,
    alpha: float,
    *,
    model,
    resolve,
    bridge_coll,
    dry_run: bool,
    log: logging.Logger,
    workers: int = 1,
) -> int:
    """Form MetaBary at ``child_level - 2`` from children@L and bridges@L-1."""
    child_ids, child_w, CV = load_unparented_bes(coll, child_level)
    bridge_ids, bridge_w, BV = load_unparented_bes(coll, bridge_level)
    if len(child_ids) < 2 or not bridge_ids:
        return 0

    pairs = greedy_unique_match(top_k_pairs(CV, min_score=threshold), threshold=threshold)
    if not pairs:
        return 0
    log.info(
        "L%d: %d pairs from %d children, %d bridges available",
        child_level - 2, len(pairs), len(child_ids), len(bridge_ids),
    )

    n_bridges = len(bridge_ids)
    window = min(_BRIDGE_WINDOW, n_bridges)
    bridge_taken: set[int] = set()
    triads: list[tuple[int, int, int]] = []
    dropped = 0
    for ci, cj, _q in pairs:
        centroid = CV[ci] + CV[cj]
        norm = float(np.linalg.norm(centroid))
        centroid = centroid / norm if norm else centroid
        sims = BV @ centroid
        if len(sims) > window:
            cands = np.argpartition(-sims, window)[:window]
            order = cands[np.argsort(-sims[cands])]
        else:
            order = np.argsort(-sims)
        for bi in order:
            bi = int(bi)
            if bi not in bridge_taken:
                bridge_taken.add(bi)
                triads.append((ci, cj, bi))
                break
        else:
            dropped += 1
    log.info(
        "L%d: %d triads, %d pairs dropped (no unparented bridge in window)",
        child_level - 2, len(triads), dropped,
    )
    if not triads or dry_run:
        return len(triads)

    mb_level = child_level - 2
    now = datetime.now(timezone.utc)
    states = [
        "\n".join(
            [
                f"A: {resolve(child_ids[ci])}",
                f"B: {resolve(child_ids[cj])}",
                f"BRIDGE: {resolve(bridge_ids[bi])}",
            ]
        )
        for ci, cj, bi in triads
    ]
    workers = max(1, workers)
    log.info("L%d MetaBary: classifying %d triads (workers=%d)", mb_level, len(states), workers)
    classified = classify_states(model, states, workers=workers)

    total = 0
    for start in range(0, len(triads), 1000):
        batch = triads[start : start + 1000]
        batch_cls = list(islice(classified, len(batch)))
        docs = []
        for (ci, cj, bi), cls in zip(batch, batch_cls, strict=True):
            role = cls["role"]
            vec, q_mb_raw = compute_metabary_vec(
                CV[ci], CV[cj], BV[bi],
                float(child_w[ci]), float(child_w[cj]), float(bridge_w[bi]),
                role,
            )
            acc_w = q_mb_raw * level_factor(mb_level, alpha)
            docs.append(
                metabary(
                    child_ids[ci], child_ids[cj], mb_level, vec, q_mb_raw, acc_w,
                    role=role,
                    role_probability=cls["role_probability"],
                    role_source=cls["role_source"],
                )
            )
        res = coll.insert_many(docs)
        ups: list[UpdateOne] = []
        for (ci, cj, bi), eid in zip(batch, res.inserted_ids, strict=True):
            for cm_id in (child_ids[ci], child_ids[cj], bridge_ids[bi]):
                ups.append(
                    UpdateOne({"_id": cm_id}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
                )
            # A MetaBary traces to the union of both children's and the bridge's
            # documents (doc_bridge.propagate, mirroring doi_bridge in s08).
            doc_bridge.propagate(
                bridge_coll, eid, [child_ids[ci], child_ids[cj], bridge_ids[bi]]
            )
        coll.bulk_write(ups, ordered=False)
        total += len(docs)
    return total


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    thr = settings.meta_bary_cos_threshold
    alpha = settings.level_factor_alpha
    log.info("start dry_run=%s cos_threshold=%.2f alpha=%.2f max_level=%d",
             args.dry_run, thr, alpha, args.max_level)

    if not args.dry_run and not args.force:
        if coll.count_documents({"doc_type": "baryedge", "level": {"$lte": 13},
                                 "source": {"$ne": "structural"}}, limit=1):
            raise RuntimeError(
                "MetaBary docs (level ≤13, pipeline-built) already present — "
                "re-running would double-parent BEs. Drop them and use --reset, "
                "or pass --force."
            )

    if args.dry_run:
        cp.processed = cp.total = 0
        return

    model = get_decision_model(settings)
    resolve = text_resolver(coll)
    bridge_coll = doc_bridge.get_bridge_collection(settings)

    total = 0
    child_level = 15
    while child_level - 2 >= args.max_level and child_level - 2 >= 1:
        bridge_level = child_level - 1
        n = _form_level(
            coll, child_level, bridge_level, thr, alpha,
            model=model, resolve=resolve, bridge_coll=bridge_coll,
            dry_run=args.dry_run, log=log,
            workers=max(1, int(getattr(settings, "tev1_workers", 1))),
        )
        log.info("L%d MetaBary: children@L%d bridges@L%d → %d triads",
                 child_level - 2, child_level, bridge_level, n)
        total += n
        child_level -= 1

    cp.processed = total
    cp.total = total
    finish(cp, settings, log)


if __name__ == "__main__":
    run()
