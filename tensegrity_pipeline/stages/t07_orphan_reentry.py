"""t07_orphan_reentry — recover sentences/titles whose pair never formed.

Mirrors s07's two phases, specialized to the tensegrity role model:

* **pair** — pair still-unparented nodes at a level with each other (cosine
  greedy matching), exactly like t04 but restricted to orphans.
* **absorb** — for each remaining orphan, find the nearest *existing* BaryEdge
  at that level and classify the orphan against the BaryEdge's two participants
  with tev1 (Step III: "classify the orphan sentence against the two sentences
  of the candidate BE"). A new BaryEdge ``(orphan, BE)`` is written and the
  orphan is stamped with its ``parent_edge_id``.

Runs L15 then L14. ``--phase pair|absorb|both`` selects the phase(s).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib.embed import get_embedder
from lib.vector import unpack_vec
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.db import get_collection
from tensegrity_pipeline.docs import tensegrity_baryedge
from tensegrity_pipeline.pairing import (
    form_level_edges,
    load_level_nodes,
    role_type_vectors,
    text_resolver,
)
from tensegrity_pipeline.tev1 import classify_role, get_decision_model
from tensegrity_pipeline.tg_vec import COMPLEMENT, compute_bary_vec

STAGE = "07_orphan_reentry"
LEVELS = (15, 14)


def absorb_orphans(
    coll,
    settings,
    embedder,
    model,
    level: int,
    *,
    dry_run: bool = False,
    log: logging.Logger | None = None,
) -> int:
    """Attach each orphan at ``level`` to its nearest existing BaryEdge."""
    log = log or logging.getLogger(__name__)
    ids, texts, V = load_level_nodes(coll, level, parents_only=True)
    if not ids:
        log.info("L%d: no orphans to absorb", level)
        return 0

    be_vecs: dict = {}
    for d in coll.find(
        {"doc_type": "baryedge", "level": level}, {"_id": 1, "vector": 1}
    ):
        v = d.get("vector")
        if v is not None:
            be_vecs[d["_id"]] = unpack_vec(v)
    if not be_vecs:
        log.info("L%d: no BaryEdges to absorb into — skipping", level)
        return 0

    resolve = text_resolver(coll)
    role_vecs = role_type_vectors(embedder)
    be_ids = list(be_vecs)
    BE = np.stack([be_vecs[b] for b in be_ids]).astype(np.float32)

    now = datetime.now(timezone.utc)
    docs: list[dict] = []
    orphan_ids: list = []
    n = 0

    def _flush() -> None:
        nonlocal docs, orphan_ids
        if not docs:
            return
        if not dry_run:
            res = coll.insert_many(docs)
            coll.bulk_write(
                [
                    UpdateOne({"_id": oid}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
                    for oid, eid in zip(orphan_ids, res.inserted_ids, strict=True)
                ],
                ordered=False,
            )
        docs = []
        orphan_ids = []

    for k, oid in enumerate(ids):
        qv = V[k]
        sims = BE @ qv  # vectors are unit-normalized
        bi = int(np.argmax(sims))
        q = float(sims[bi])
        be_id = be_ids[bi]
        state = "\n".join(t for t in (texts[k], resolve(be_id)) if t)
        cls = classify_role(model, state)
        role = cls["role"]
        v_role = role_vecs.get(role or COMPLEMENT, role_vecs[COMPLEMENT])
        bv = compute_bary_vec(qv, be_vecs[be_id], v_role, q, role)
        docs.append(
            tensegrity_baryedge(
                oid,
                be_id,
                level,
                bv,
                q,
                role=role,
                role_probability=cls["role_probability"],
                role_source=cls["role_source"],
                accumulated_weight=q,
                type_vector=v_role,
                source="reentry",
                confidence=q,
            )
        )
        orphan_ids.append(oid)
        n += 1
        if len(docs) >= settings.batch_size:
            _flush()
    _flush()
    log.info("L%d: absorbed %d orphans into nearest BaryEdges", level, n)
    return n


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start phase=%s dry_run=%s", args.phase, args.dry_run)

    embedder = get_embedder(settings)
    model = get_decision_model(settings)
    total = 0

    for level in LEVELS:
        if args.phase in ("pair", "both"):
            total += form_level_edges(
                coll, settings, embedder, model, level,
                limit=args.limit, parents_only=True,
                dry_run=args.dry_run, log=log,
            )
        if args.phase in ("absorb", "both"):
            total += absorb_orphans(
                coll, settings, embedder, model, level,
                dry_run=args.dry_run, log=log,
            )

    cp.processed = total
    cp.total = total
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
