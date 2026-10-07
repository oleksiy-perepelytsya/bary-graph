"""Repair s07's absorb phase after the float64-projection corruption.

2026-10-08 — The s07 absorb ``_stream_be_pool`` wrote projected rows as
float64 (``row @ proj.T`` with a float64 projector) into a binary file that
was reopened as a float32 memmap, silently misreading every projected BE row
as garbage. The absorb phase then matched the 70,938 orphan words against
that garbage BE pool, so each absorbed word got an *arbitrary* partner BE
and a derived BE (``source="inferred"``, ``cm2_id`` = a baryedge id).

This worker removes exactly those artifacts so ``--phase absorb`` can be
re-run cleanly with the fixed writer:

  * classify L14 ``source="inferred"`` BEs: absorb BEs are the ones whose
    ``cm2_id`` resolves to a baryedge doc (pair-phase BEs point at word
    nodes).
  * delete the absorb BEs and set ``parent_edge_id = None`` on the orphan
    words that pointed at them (their only children).

Dry-run by default; pass ``--apply`` to mutate. Refuses to run when the
classifier counts look implausible (absorb set outside ~[60k, 85k]).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pymongo.errors  # noqa: F401  (import hygiene for interpolated errors)

from lib.config import Settings
from lib.db import get_collection

STAGE = "s07_absorb_repair"
MIN_ABSORB, MAX_ABSORB = 60_000, 85_000
_DEL_CHUNK = 20_000


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete/unparent (default: dry-run)")
    args, _ = ap.parse_known_args(argv)
    settings = Settings.load()
    coll = get_collection(settings)
    print(f"db={coll.database.name} coll={coll.name}")

    # 1. Load every L14 source="inferred" BE's cm ids.
    bes = list(coll.find(
        {"doc_type": "baryedge", "level": 14, "source": "inferred"},
        {"cm1_id": 1, "cm2_id": 1},
    ))
    print(f"L14 inferred BEs: {len(bes)}")
    cm2_ids = {b["cm2_id"] for b in bes if b.get("cm2_id")}
    print(f"distinct cm2 ids: {len(cm2_ids)}")

    # 2. Resolve cm2 doc_type to separate absorb (cm2=baryedge) from pair BEs.
    cm2_type: dict = {}
    for s in range(0, len(cm2_ids), _DEL_CHUNK):
        chunk = list(cm2_ids)[s:s + _DEL_CHUNK]
        for d in coll.find({"_id": {"$in": chunk}}, {"doc_type": 1}):
            cm2_type[d["_id"]] = d.get("doc_type")
    absorb = [b for b in bes
              if cm2_type.get(b.get("cm2_id")) == "baryedge"]
    pair = [b for b in bes
            if cm2_type.get(b.get("cm2_id")) == "node"]
    unmatched = [b for b in bes if b.get("cm2_id") not in cm2_type]
    print(f"absorb BEs (cm2=baryedge): {len(absorb)}")
    print(f"pair BEs   (cm2=node):    {len(pair)}")
    print(f"unresolved cm2:           {len(unmatched)}")

    if not (MIN_ABSORB <= len(absorb) <= MAX_ABSORB):
        print(f"ABORT: absorb count {len(absorb)} outside [{MIN_ABSORB}, "
              f"{MAX_ABSORB}] — refusing to continue.")
        return 2
    n_words = len({b["cm1_id"] for b in absorb})
    print(f"orphan words attached to absorb BEs: {n_words}")

    if not args.apply:
        print("dry-run: nothing mutated. Pass --apply to repair.")
        return 0

    # 3. Unparent the orphan words, then delete the absorb BEs.
    word_ids = [b["cm1_id"] for b in absorb]
    del_ids = [b["_id"] for b in absorb]
    for s in range(0, len(word_ids), _DEL_CHUNK):
        chunk = word_ids[s:s + _DEL_CHUNK]
        coll.update_many(
            {"_id": {"$in": chunk}, "parent_edge_id": {"$in": del_ids}},
            {"$set": {"parent_edge_id": None, "updated_at": None}},
        )
    print(f"unparented {len(word_ids)} words")
    for s in range(0, len(del_ids), _DEL_CHUNK):
        coll.delete_many({"_id": {"$in": del_ids[s:s + _DEL_CHUNK]}})
    print(f"deleted {len(del_ids)} absorb BEs")

    # 4. Report post-state.
    left = coll.count_documents(
        {"doc_type": "baryedge", "level": 14, "source": "inferred"})
    orphans = coll.count_documents(
        {"doc_type": "node", "node_type": "word", "level": 14,
         "parent_edge_id": None})
    print(f"post: L14 inferred BEs={left} (expect {len(pair)}), "
          f"unparented L14 words={orphans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())