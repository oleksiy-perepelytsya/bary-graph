#!/usr/bin/env python3
"""find_bridge.py — locate the L-1 bridge doc for a pivot word.

The admission rule for a structural SMB is: two *distant* children whose
clusters share a pivot word, joined by a bridge doc one level below the
children that itself carries the pivot word. This tool does the bridge hunt:

  * take the union of the L<level> neighbourhoods of the given seed docs
    (the two child clusters, or their representative nodes);
  * keep only docs whose leaf-word set contains the pivot word (or any of
    --keys);
  * print each match (id + its leaf words), lowest-cosine first.

Usage
-----
    find_bridge.py PIVOT SEED_ID [SEED_ID ...] [options]

    PIVOT                 the shared word to look for (also a default --key)
    SEED_ID...            one or more doc ids whose neighbourhoods to search
                          (typically one node from each child cluster)

Options
-------
    --keys a,b,c          extra/alternative leaf words to match (default: PIVOT)
    --level 12            level of the bridge docs to search (default 12)
    --limit 70            neighbours per seed ($vectorSearch limit)
    --num-candidates 800  ANN candidates per seed
    --max-words 24        leaf words fetched per candidate
    --seed N              RNG seed (kept for symmetry; search is deterministic)

Exit status is 0 even when nothing matches (an empty hunt is a normal HOLD).
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/home/alex/Projects/bary-vector")

from bson import ObjectId  # noqa: E402

from lib.config import Settings  # noqa: E402
from lib.db import cm_leaf_words, get_collection, vector_search  # noqa: E402
from lib.vector import unpack_vec  # noqa: E402


def words(coll, oid, k: int) -> list[str]:
    try:
        return sorted(cm_leaf_words(coll, ObjectId(str(oid)), max_depth=13, max_words=k))
    except Exception:
        return ["ERR"]


def vec(coll, oid):
    d = coll.find_one({"_id": ObjectId(str(oid))}, {"vector": 1})
    if d and d.get("vector") is not None:
        return [float(x) for x in unpack_vec(d["vector"])]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Find the L-1 bridge doc for a pivot word.")
    ap.add_argument("pivot", help="the shared pivot word")
    ap.add_argument("seeds", nargs="+", help="seed doc id(s) whose neighbourhoods to search")
    ap.add_argument("--keys", help="comma-separated leaf words to match (default: pivot)")
    ap.add_argument("--level", type=int, default=12, help="bridge level to search (default 12)")
    ap.add_argument("--limit", type=int, default=70)
    ap.add_argument("--num-candidates", type=int, default=800)
    ap.add_argument("--max-words", type=int, default=24)
    args = ap.parse_args()

    keys = [k.strip().lower() for k in (args.keys or args.pivot).split(",") if k.strip()]

    s = Settings.load()
    coll = get_collection(s)

    seen: dict[str, dict] = {}
    for sd in args.seeds:
        qv = vec(coll, sd)
        if qv is None:
            print(f"# seed {sd}: no stored vector — skipped", file=sys.stderr)
            continue
        for h in vector_search(coll, qv, limit=args.limit,
                               num_candidates=args.num_candidates,
                               filter={"level": args.level}):
            seen[str(h["_id"])] = h

    print(f"=== {args.pivot!r}: {len(seen)} L{args.level} docs near "
          f"{len(args.seeds)} seed(s); matching keys={keys} ===", flush=True)
    matches = 0
    for i in seen:
        w = words(coll, i, args.max_words)
        low = " ".join(w).lower()
        if any(k in low for k in keys):
            print("  MATCH", i, json.dumps(w, ensure_ascii=False), flush=True)
            matches += 1
    if not matches:
        print(f"  (no L{args.level} doc carries the pivot — HOLD, do not force a bridge)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
