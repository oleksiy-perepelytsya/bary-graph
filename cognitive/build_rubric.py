#!/usr/bin/env python3
"""build_rubric.py — one-pass SMB analysis rubric (DISTANCE / DISTINCTNESS /
MECHANISM / GROUNDING).

A light analysis-quality check run *before* a build is decided. It computes
mechanical evidence from the graph (cosines, shared leaves, pivot presence)
and scores four axes 0-5. It is a DECISION AID for the agent's value
judgment — explicitly NOT a gate and NOT an oracle:

  - it never blocks a build (value judgment is the sole admission criterion),
  - it has no q threshold and no verdict of its own,
  - its scores land in smb_builds.jsonl `analysis` so every build carries the
    analysis that admitted it, making near-misses auditable.

Axes (mirror the older cog step-3 axes, one pass, no re-probe):
  distance     — child_cosine between the two branches (low = distant = good):
                 <0.30→5, <0.45→4, <0.60→3, <0.75→2, else→1
  distinctness — shared leaf words between cm1 and cm2 (*excluding* the
                 pivot) — 0→5, 1→4, 2-3→3, 4-5→2, >5→1
  mechanism    — bridge=symmetric real middle, not equal to either branch and
                 carrying the pivot: pivot-in-bridge (3) + min(cos(br,cm)) map.
                 The 1-sentence mechanism test ("what operation/state makes
                 the two cohere?") is the agent's job — the score is a hint.
  grounding    — every term you quote actually appears in the retrieved
                 leaves of its side (pass --cm1-terms/--bridge-terms/
                 --cm2-terms to check; otherwise scores from pivot presence
                 only).

Usage
-----
    build_rubric.py CM1 CM2 BRIDGE [--pivot WORD]
        [--cm1-terms a,b,c] [--bridge-terms x,y] [--cm2-terms p,q]
        [--json]          # machine-readable for the build manifest

Importable: `rubric(coll, cm1_id, cm2_id, bridge_id, pivot, terms=...)`
returns {"scores": {axis: 0-5}, "evidence": {...}, "mechanism_test": "..."}.
"""
from __future__ import annotations

import argparse
import json

from bson import ObjectId

from lib.config import Settings
from lib.db import cm_leaf_words, get_collection
from lib.vector import unpack_vec


def _cos(v1, v2) -> float:
    n1 = sum(x * x for x in v1) ** 0.5
    n2 = sum(y * y for y in v2) ** 0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    return sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)


def _map_dist(cc: float) -> int:
    if cc < 0.30: return 5
    if cc < 0.45: return 4
    if cc < 0.60: return 3
    if cc < 0.75: return 2
    return 1


def _leaves(coll, oid, max_words=80):
    try:
        return set(cm_leaf_words(coll, ObjectId(oid), max_depth=15, max_words=max_words))
    except Exception:
        return set()


def rubric(coll, cm1_id, cm2_id, bridge_id, pivot: str | None = None,
           terms: dict | None = None) -> dict:
    def fetch(oid):
        try:
            d = coll.find_one({"_id": ObjectId(oid)},
                              {"vector": 1, "level": 1, "doc_type": 1})
            return d
        except Exception:
            return None

    cm1, cm2, bridge = fetch(cm1_id), fetch(cm2_id), fetch(bridge_id)
    if not (cm1 and cm2 and bridge) or not all(d.get("vector") for d in (cm1, cm2, bridge)):
        return {"ok": False, "error": "missing doc(s) or vector(s)"}

    v1 = [float(x) for x in unpack_vec(cm1["vector"])]
    v2 = [float(x) for x in unpack_vec(cm2["vector"])]
    vb = [float(x) for x in unpack_vec(bridge["vector"])]
    cc = _cos(v1, v2)
    cbr, cbr2 = _cos(v1, vb), _cos(v2, vb)

    l1, l2, lb = _leaves(coll, cm1_id), _leaves(coll, cm2_id), _leaves(coll, bridge_id)
    p = pivot.lower() if pivot else None
    shared_l1l2 = sorted(l1 & l2)
    shared_l1b  = sorted(l1 & lb)
    shared_l2b  = sorted(l2 & lb)

    # ---- distance ----
    distance = _map_dist(cc)

    # ---- distinctness (shared words excluding pivot) ----
    excl = {p} if p else set()
    n_shared = len(set(shared_l1l2) - excl)
    distinctness = 5 if n_shared == 0 else 4 if n_shared == 1 else 3 if n_shared <= 3 else 2 if n_shared <= 5 else 1

    # ---- mechanism (hint only — the 1-sentence test is the agent's) ----
    pivot_in_bridge = p in {w.lower() for w in lb}
    symmetric = min(cbr, cbr2)
    mechanism = (3 if pivot_in_bridge else 0) + (2 if symmetric >= 0.15 else 1 if symmetric >= 0.05 else 0)
    mechanism = min(5, mechanism)

    # ---- grounding ----
    def side_hit(terms_list, leaves):
        if not terms_list:
            return None
        leaves_l = {w.lower() for w in leaves}
        missing = [t for t in terms_list if t.lower() not in leaves_l]
        return len(missing)

    grounding_acc = 0.0
    grounding_n = 0
    for key in ("cm1_terms", "bridge_terms", "cm2_terms"):
        tl = (terms or {}).get(key)
        if tl:
            m = side_hit(tl, {"cm1_terms": l1, "bridge_terms": lb, "cm2_terms": l2}[key])
            if m is not None:
                grounding_acc += max(0, 5 - m)
                grounding_n += 1
    if grounding_n:
        grounding = round(grounding_acc / grounding_n)
    else:
        # no term lists: pivot presence as a weak grounding signal
        hits = int(p in {w.lower() for w in l1}) + int(p in {w.lower() for w in l2}) + int(pivot_in_bridge)
        grounding = 1 if hits == 0 else 2 if hits == 1 else 3 if hits == 2 else 4

    evidence = {
        "child_cosine": round(cc, 4),
        "cos(bridge,cm1)": round(cbr, 4),
        "cos(bridge,cm2)": round(cbr2, 4),
        "levels": {"cm1": cm1.get("level"), "cm2": cm2.get("level"),
                   "bridge": bridge.get("level")},
        "doc_types": {"cm1": cm1.get("doc_type"), "cm2": cm2.get("doc_type"),
                      "bridge": bridge.get("doc_type")},
        "pivot_in_bridge": pivot_in_bridge,
        "shared(c1,c2)": shared_l1l2[:12],
        "shared(c1,b)": shared_l1b[:12],
        "shared(c2,b)": shared_l2b[:12],
    }
    return {
        "ok": True,
        "scores": {"distance": distance, "distinctness": distinctness,
                   "mechanism": mechanism, "grounding": grounding},
        "evidence": evidence,
        "mechanism_test": ("What operation/state/artifact makes cm1 and cm2 cohere? "
                           "The bridge must be that mechanism, not a synonym of either "
                           "branch."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cm1_id")
    ap.add_argument("cm2_id")
    ap.add_argument("bridge_id")
    ap.add_argument("--pivot", default=None)
    ap.add_argument("--cm1-terms", default="")
    ap.add_argument("--bridge-terms", default="")
    ap.add_argument("--cm2-terms", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    terms = {k: [t.strip() for t in v.split(",") if t.strip()]
             for k, v in (("cm1_terms", args.cm1_terms),
                          ("bridge_terms", args.bridge_terms),
                          ("cm2_terms", args.cm2_terms)) if v}

    coll = get_collection(Settings.load())
    r = rubric(coll, args.cm1_id, args.cm2_id, args.bridge_id, args.pivot, terms)
    if not r.get("ok"):
        print(r.get("error"))
        return 1
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
    else:
        print("RUBRIC", json.dumps(r["scores"]))
        ev = r["evidence"]
        print(f"  child_cosine={ev['child_cosine']}  "
              f"cos(bridge,cm1)={ev['cos(bridge,cm1)']}  "
              f"cos(bridge,cm2)={ev['cos(bridge,cm2)']}")
        print(f"  levels cm1/cm2/bridge = {ev['levels']}")
        print(f"  pivot_in_bridge = {ev['pivot_in_bridge']}")
        print(f"  shared(c1,c2) = {ev['shared(c1,c2)']}")
        print(f"  shared(c1,b)  = {ev['shared(c1,b)']}")
        print(f"  shared(c2,b)  = {ev['shared(c2,b)']}")
        print(f"  mechanism test: {r['mechanism_test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())