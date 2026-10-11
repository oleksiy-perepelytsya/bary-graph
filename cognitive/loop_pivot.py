#!/usr/bin/env python3
"""Batch W enrichment: pivot hunt.

For each independently-drawn seed, vector-search its neighbourhood and keep the
neighbours that SHARE a leaf word with the seed, ranked by cosine (ascending).
A shared word + low cosine = a pivot word with two lives (the productive signal
found in batch V: yarn 0.514, tanker 0.617).

  argv[1] = seeds (default 15), argv[2] = levels (default "13,12")
  --seed N = seed the RNG so a run is reproducible (for replay/diff)
  --top N  = print the N most distant pairs (default 24; use a large N to
             see every shared-word pair, needed to file per-seed outcomes)
  All drawn seeds are emitted to stderr as {"drawn": true, ...} lines so a
  batch is fully recordable even when a seed yields no shared-word neighbour.
"""
import sys, json, random
import numpy as np
sys.path.insert(0, "/home/alex/Projects/bary-vector")
from bson import ObjectId
from lib.config import Settings
from lib.db import get_collection, vector_search, cm_leaf_words
from lib.vector import unpack_vec

_argv = list(sys.argv[1:])
_rng_seed = None
if "--seed" in _argv:
    _i = _argv.index("--seed")
    _rng_seed = int(_argv[_i + 1])
    del _argv[_i:_i + 2]
if _rng_seed is not None:
    random.seed(_rng_seed)
_TOP = 24
if "--top" in _argv:
    _i = _argv.index("--top")
    _TOP = int(_argv[_i + 1])
    del _argv[_i:_i + 2]

N = int(_argv[0]) if len(_argv) > 0 else 15
LEVELS = [int(x) for x in (_argv[1].split(",") if len(_argv) > 1 else ["13", "12"])]
K = 50
s = Settings.load(); c = get_collection(s)

_wc, _vc = {}, {}
def words(oid, k=8):
    key = str(oid)
    if key not in _wc:
        _wc[key] = sorted(cm_leaf_words(c, ObjectId(key), max_depth=13, max_words=k))
    return _wc[key]
def vec(oid):
    key = str(oid)
    if key not in _vc:
        d = c.find_one({"_id": ObjectId(key)}, {"vector": 1})
        _vc[key] = [float(x) for x in unpack_vec(d["vector"])] if d and d.get("vector") else None
    return _vc[key]
def cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = (a.dot(a) ** 0.5) * (b.dot(b) ** 0.5)
    return float(a.dot(b) / n) if n else 0.0

# independent seed draw (same fix now in _random_metabary_docs)
seeds = []
for lvl in LEVELS:
    filt = {"doc_type": "baryedge", "level": lvl}
    cnt = c.count_documents(filt)
    per = max(1, N // len(LEVELS))
    for _ in range(per):
        k = random.randrange(cnt) if cnt > 1 else 0
        d = next(iter(c.find(filt, {"cm1_id": 1, "cm2_id": 1, "connection_strength": 1}).skip(k).limit(1)), None)
        if d:
            seeds.append((lvl, d))

hits = []
for lvl, d in seeds:
    sid = str(d["_id"]); sv = vec(sid)
    print(json.dumps({"drawn": True, "seed": {"id": sid, "lvl": lvl},
                      "words": words(sid)}), file=sys.stderr)
    if sv is None: continue
    sw = set(words(sid))
    if not sw: continue
    for h in vector_search(c, sv, limit=K, num_candidates=500):
        nid = str(h["_id"])
        if nid == sid: continue
        shared = sw & set(words(nid))
        if not shared: continue
        cc = cos(sv, vec(nid)) if vec(nid) else 1.0
        hits.append((round(cc, 3), sorted(shared), lvl, sid, h.get("level"), nid, words(sid), words(nid)))

hits.sort()
for cc, shared, sl, sid, nl, nid, sw, nw in hits[:_TOP]:
    print(json.dumps({"cos": cc, "shared": shared, "seed": {"id": sid, "lvl": sl, "w": sw},
                      "nb": {"id": nid, "lvl": nl, "w": nw}}, ensure_ascii=False))
print(f"# seeds={len(seeds)} shared-word neighbour pairs={len(hits)}", file=sys.stderr)
