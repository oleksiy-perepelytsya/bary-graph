#!/usr/bin/env python3
"""Find an L12 bridge doc whose leaf words contain the pivot word."""
import sys, json
sys.path.insert(0, "/home/alex/Projects/bary-vector")
from bson import ObjectId
from lib.config import Settings
from lib.db import get_collection, vector_search, cm_leaf_words
from lib.vector import unpack_vec

s = Settings.load(); c = get_collection(s)

def w(oid, k=20):
    try: return sorted(cm_leaf_words(c, ObjectId(str(oid)), max_depth=13, max_words=k))
    except Exception: return ["ERR"]

def vec(oid):
    d = c.find_one({"_id": ObjectId(str(oid))}, {"vector": 1})
    return [float(x) for x in unpack_vec(d["vector"])] if d and d.get("vector") is not None else None

JOBS = [
    ("crab", ["6ac6e1555715285783884f42", "6ac6e19b5715285783888a64"], ["crab", "claw", "pincer", "nipper"]),
    ("derby", ["6ac6df7e571528578386afb2", "6ac7c31c51ff4d8f79c7dab4"], ["derby"]),
]
for name, seeds, keys in JOBS:
    seen = {}
    for sd in seeds:
        for h in vector_search(c, vec(sd), limit=70, num_candidates=800, filter={"level": 12}):
            seen[str(h["_id"])] = h
    print(f"=== {name}: {len(seen)} L12 docs near both clusters; matches for {keys} ===", flush=True)
    for i in seen:
        words = w(i, k=24)
        low = " ".join(words).lower()
        if any(k in low for k in keys):
            print("  MATCH", i, json.dumps(words, ensure_ascii=False), flush=True)
