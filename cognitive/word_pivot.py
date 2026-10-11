#!/usr/bin/env python3
"""word_pivot.py — word-anchored pivot scan ("one word, two lives").

Alternative to loop_pivot.py's seed-neighbourhood scan. Where that scan looks
at docs NEAR a random seed (so the shared words are the cluster's own
texture, cosine >= 0.84), this scan anchors on a WORD and enumerates the
word's own lives directly:

  1. draw random words (random L14 word nodes),
  2. load every L15 sense node of the word (`properties.word` is indexed),
  3. cluster senses by cosine; find the most distant pair of SUBSTANTIVE
     senses (drops obsolete/archaic/rare/slang senses),
  4. rank words ascending by that distance (low = a real two-lives spread).

The L14 word node of the word is reported as `bridge`: if two sense-lives are
distant, that node literally carries the word and sits between them — the
same shape as the accepted "lawn" SMB (L15 sense-lives -> L14 word node).
Level arithmetic for a build from this scan: children L15, bridge L14, SMB L13.

A scan hit is a CANDIDATE, not a build: the agent still judges substantiveness,
hunts the two lives as real graph clusters, and passes build_rubric/build_gate.
No q rule, no automatic admission.

  argv = explicit words to scan (primary; pass seed leaf-words or candidates)
  --draw N        instead draw N random words from the graph (noisier)
  --seed N        reproducible draw
  --thr F         sense-cluster cosine threshold (default 0.80)
  --max-skew N    cap senses per word (default 60)
"""
import sys, json, random, ast, string
sys.path.insert(0, "/home/alex/Projects/bary-vector")
import numpy as np
from bson import ObjectId
from lib.config import Settings
from lib.db import get_collection
from lib.vector import unpack_vec

_argv = list(sys.argv[1:])
def _flag(name, cast):
    if name in _argv:
        i = _argv.index(name); v = cast(_argv[i + 1]); del _argv[i:i + 2]; return v
    return None
_rng_seed = _flag("--seed", int)
THR = _flag("--thr", float) or 0.80
MAXSKEW = _flag("--max-skew", int) or 60
DRAW = _flag("--draw", int)
if _rng_seed is not None:
    random.seed(_rng_seed)
# positional args = explicit words to scan (the primary mode: pass seed
# leaf-words, candidate words, or any word list). --draw N instead draws N
# random words from the graph (noisier; mostly rare words -- prefer explicit).
EXPLICIT = [a for a in _argv if not a.startswith("--")]
DRAW_N = DRAW if DRAW is not None else (12 if not EXPLICIT else 0)

c = get_collection(Settings.load())
TRIVIAL = {"obsolete", "archaic", "nonstandard", "rare", "slang",
           "informal", "dialectal", "dated", "historical", "misspelling"}

def parse_tags(t):
    try:
        return set(x.lower() for x in ast.literal_eval(t))
    except Exception:
        return set()

def load_senses(w):
    out = []
    for d in c.find({"doc_type": "node", "level": 15, "properties.word": w},
                    {"vector": 1, "properties": 1}).limit(MAXSKEW):
        v = d.get("vector")
        if not v:
            continue
        p = d.get("properties") or {}
        tags = parse_tags(p.get("tags") or "[]")
        gloss = p.get("gloss") or ""
        out.append({"id": str(d["_id"]), "pos": p.get("pos"), "gloss": gloss,
                    "topics": p.get("topics") or "[]",
                    "substantive": (not (tags & TRIVIAL) and len(gloss) >= 12),
                    "v": np.asarray([float(x) for x in unpack_vec(v)])})
    return out

def cluster_count(senses):
    n = len(senses)
    if n == 0:
        return 0
    M = np.vstack([s["v"] for s in senses]); M = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = M @ M.T
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= THR or senses[i]["gloss"].strip().lower() == senses[j]["gloss"].strip().lower():
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    return len({find(i) for i in range(n)})

def scan_word(w, bridge_id):
    senses = load_senses(w)
    sub = [s for s in senses if s["substantive"] and not is_noise(s["pos"], s["gloss"])]
    if len(sub) < 2:
        return {"word": w, "senses": len(senses), "substantive": len(sub),
                "min_inter": None, "verdict": "few-substantive"}
    n = len(sub)
    M = np.vstack([s["v"] for s in sub]); M = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = M @ M.T
    best = 1.0; pair = None
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] < best:
                best = float(C[i, j]); pair = (i, j)
    i, j = pair
    return {"word": w, "senses": len(senses), "substantive": n,
            "clusters": cluster_count(sub), "min_inter": round(best, 3),
            "A": f"{sub[i]['pos']}: {sub[i]['gloss'][:80]}",
            "B": f"{sub[j]['pos']}: {sub[j]['gloss'][:80]}",
            "A_topics": sub[i]["topics"][:60], "B_topics": sub[j]["topics"][:60],
            "bridge_id": bridge_id, "A_sense_id": sub[i]["id"], "B_sense_id": sub[j]["id"]}

# words to scan: explicit list, else draw N random words (filtered to real
# word tokens). Random-id / $sample / skip are all O(collection) here, so a
# random `properties.word` prefix seek is used for --draw.
def draw_words(n, tries=4000):
    out, seen = [], set()
    for _ in range(tries):
        if len(out) >= n:
            break
        ln = random.choice([3, 3, 4, 4, 5])
        p = "".join(random.choice(string.ascii_lowercase) for _ in range(ln))
        d = c.find_one({"doc_type": "node", "level": 15, "node_type": "sense",
                        "properties.word": {"$gte": p}}, {"properties.word": 1})
        w = (d.get("properties") or {}).get("word") if d else None
        if w and w.isalpha() and len(w) >= 3 and w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out

if EXPLICIT:
    drawn = EXPLICIT
else:
    drawn = draw_words(DRAW_N)
print(json.dumps({"scanning": drawn}), file=sys.stderr)

def word_node_id(w):
    d = c.find_one({"doc_type": "node", "level": 14, "node_type": "word",
                    "properties.word": w}, {"_id": 1})
    return str(d["_id"]) if d else None

INFLECT_PREFIX = ("plural of", "third-person", "simple past", "past tense",
                  "present participle", "comparative of", "superlative of",
                  "gerund of", "alternative spelling", "alternative form",
                  "misspelling", "obsolete form", "archaic form")

def is_noise(pos, gloss):
    if pos == "name":
        return True
    g = gloss.strip().lower()
    return any(g.startswith(p) for p in INFLECT_PREFIX)

rows = []
for w in drawn:
    try:
        r = scan_word(w, word_node_id(w))
    except Exception as e:
        r = {"word": w, "error": str(e)}
    r["seed_word"] = True
    rows.append(r)

rows.sort(key=lambda r: (r.get("min_inter") is None, r.get("min_inter") or 1.0))
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
