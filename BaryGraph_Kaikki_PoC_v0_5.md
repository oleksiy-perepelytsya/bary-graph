# BaryGraph PoC: English Kaikki Dictionary
## A Local-First Proof of Concept
**Version 0.5 · April 2026**

> **v0.5 changes from v0.4:**
> - Removed word-length feature φ(W) and λ entirely. Word length is a
>   property of orthographic form, not semantic content. Register
>   differences (e.g. "use" vs "utilization") are already captured by
>   gloss embeddings. L14 word vector is now a clean BE-centroid + orphan
>   senses only.
> - MetaBary vector construction now uses children's `accumulated_weight`
>   (not raw `q`) as pull weights, so structurally authoritative children
>   tilt the MetaBary vector toward their semantic space.
> - New field `accumulated_weight` on every BaryEdge and MetaBary. At
>   L14/L15 it equals `q`. At L13 and above it equals
>   `q_MB_raw · level_factor(L)`, compounding upward with each level.
> - `level_factor(L) = 1 + α · (14 − L) / 13` (default α = 0.5). L13
>   gets no boost; L1 gets maximum boost of 1 + α. `accumulated_weight`
>   can exceed 1 above L13.
> - `connection_strength` is preserved as raw `q` / `q_MB_raw` (always
>   in [0,1]). `accumulated_weight` is the separate structural signal
>   passed upward.
> - Removed: `char_len`, `syllable_ct` from word node properties (no
>   longer needed for vector computation).
> - Removed: parameter R6 (λ tuning). Removed open question §19.8–9
>   (word-length syllable counter, λ tuning).
> - Stage 9 `s09_extend.py` added and run: MetaBary extension with
>   per-level cosine threshold relaxation. Picks up where s08 left off,
>   sweeping the threshold from the 0.9 base down to a per-level floor
>   (`0.9 − child_level × 0.01`; L15 floor = 0.75). Single MongoDB
>   fetch per child_level, then in-memory greedy sweep — 14× faster
>   than re-fetching per threshold round. Repeats full descending passes
>   until a complete pass produces zero new triads.

> **Level orientation:** L1 = top (most abstract), L15 = bottom (most
> concrete sense). MetaBary climbs from L to L-2 using L-1 as bridge.

---

## 1. Objective

Validate the core BaryGraph hypothesis — that relationship-aware vector
retrieval outperforms flat nearest-neighbour search — using the English
machine-readable dictionary from kaikki.org as corpus. The PoC runs
entirely on local hardware: MongoDB Community Edition with mongot for
storage and vector search, and nomic-embed-text for embeddings.

### 1.1 Why This Corpus

- **Pre-labeled relations.** Synonyms, derived forms, hypernyms, and
  etymology are explicit. BaryEdge `edge_type` comes from the corpus.
- **Built-in ground truth.** Hold out 10% of synonym links; measure
  recall against BaryEdge retrieval with zero human annotation.
- **Rich polysemy.** Words like "bank", "crane", "bark" have senses so
  distant in meaning that no embedding makes them neighbours — yet they
  are deeply related. BaryGraph surfaces this via MetaBary triads.
- **Bounded scale.** ~800K headwords, ~2–2.5M senses.
- **Translations as future cross-language bridges.** Each translation
  entry carries a `sense` gloss string that disambiguates which sense
  it corresponds to — enabling precise cross-language BaryEdge
  construction without cross-dump word matching.

### 1.2 What This PoC Validates

| BaryGraph Claim | How Kaikki Tests It |
|---|---|
| BaryEdge retrieval > flat retrieval | Hold-out recall on synonym links |
| `bary_vec` formula is useful | NN search on `bary_vec` retrieves correct CM pair |
| Forest-structure MetaBary encodes polysemy | Triad paths vs. WordNet sense clusters |
| Cross-level hierarchy via single `$graphLookup` | Forest traversal from L15 to root |
| Sense disambiguation from `_dis1` weights | Precision of sense-level edge assignment |
| Fermion-ordered matching preserves rare signals | Antonym edges survive synonym flood |
| Accumulated weight compounds structural authority | Higher-level MBs dominate retrieval ranking |

### 1.3 What This PoC Does Not Cover

- Antonym `contradicts` edges as primary eval signal (low noun coverage).
- Cross-language bridges (deferred to multi-language expansion).
- Production deployment, sharding, or cloud migration.

---

## 2. Core Equations

### 2.1 BaryEdge Vector (L14, L15)

```
bary_vec = normalize( q·v(CM₁) + q·v(CM₂) + (1−q)·v(type) )
```

where `q` is connection quality (0–1) and `v(type)` is level-dependent
(see §2.3).

At L14 and L15, `accumulated_weight = q` (base case — no compounding yet).

### 2.2 MetaBary Vector (L13 and above)

MetaBary construction uses two separate computations per new MetaBary:

**Step 1 — vector direction** (what this MB *is*):

```
meta_bary = normalize( w₁·v(BE₁) + w₂·v(BE₂) + w₃·v(bridge) )
```

where `w₁`, `w₂`, `w₃` are the `accumulated_weight` values of the three
children. Children with higher structural authority pull the MetaBary
vector toward their semantic space. After `normalize()` the magnitudes
are discarded — only direction survives.

**Step 2 — accumulated weight** (what this MB passes upward):

```
q_MB_raw        = w₃² / √(w₁⁴ + w₂⁴ + w₃⁴)
level_factor(L) = 1 + α · (14 − L) / 13        # α default: 0.5
accumulated_weight = q_MB_raw · level_factor(L)
```

**Born rule interpretation:** `w` is amplitude, `w²` is connection
probability. `q_MB_raw` is the bridge probability measured against the
combined probability mass of all three inputs. The bridge (`w₃`) dominates
— a weak bridge weakens the MetaBary regardless of BE₁/BE₂ strength.
`level_factor` then amplifies this raw quality by how high in the
hierarchy the MetaBary sits.

**Role separation:**

| Role | Consumes | Produces |
|---|---|---|
| Vector direction | children's `accumulated_weight` as pull weights | direction only — magnitude discarded by normalize |
| `accumulated_weight` | children's `accumulated_weight` + `level_factor(L)` | scalar stored on this MB, passed to parent level |

The chain is clean: L15 `q` seeds the base case; each level up consumes
what the level below stored, never reaching back further.

**Level factor table (α = 0.5):**

| Level | level_factor | Effect |
|---|---|---|
| L13 | 1.00 | No boost — just formed |
| L11 | 1.15 | Mild boost |
| L9  | 1.31 | Moderate |
| L7  | 1.46 | Strong |
| L5  | 1.62 | Very strong |
| L3  | 1.77 | Near-maximum |
| L1  | 1.50 | Maximum (1 + α) |

`accumulated_weight` can exceed 1.0 above L13. This is intentional —
it encodes structural authority that has compounded through multiple
rounds of triadic selection.

**At L13 specifically** (`BE₁`, `BE₂` are L15 BEs, bridge is L14 BE):

```
w₁ = be1.accumulated_weight   # = be1.q  (L15 base case)
w₂ = be2.accumulated_weight   # = be2.q  (L15 base case)
w₃ = bridge.accumulated_weight  # = bridge.q  (L14 base case)
```

Above L13, all three children are MetaBarys whose `accumulated_weight`
may already exceed 1.

### 2.3 v(type) Construction

**L15 — per-pair lexical neighborhood:**

```
type_text = "W_A (antonyms: a₁, a₂, …; synonyms: s₁, s₂, …); W_B (antonyms: b₁, b₂, …; synonyms: t₁, t₂, …)"
v(type)   = embed(type_text)
```

This anchors every L15 BE in the lexical neighborhood of both parent
words. The antonyms inject polarity contrast; the synonyms inject
cluster membership. The result captures relational context around the
pairing, not just the pairing itself.

For same-headword sense pairs (polysemy), the type text degenerates to
one word's neighborhood. Still valid but less informative — q will
typically be lower for these pairs anyway.

For words with empty synonym/antonym sets (rare words), falls back to
`embed("W_A; W_B")`.

**L14 — TYPE_SENTENCES (fixed per edge type):**

```python
TYPE_SENTENCES = {
    'same_phenomenon': 'these two words describe the same concept',
    'contradicts':     'these two words have opposite meanings',
    'extends':         'one word is derived from or extends the other',
    'applies_to':      'these two words share a common origin or root',
    'is_instance_of':  'this relationship is a specific instance of the broader relationship',
}
v_type = embed(TYPE_SENTENCES[edge_type])
```

L14 is the only level where TYPE_SENTENCES is used.

**L13+ — bridge vector (no embedding call):**

Above L14, the bridge BaryEdge's vector serves directly as the third
component. No embedding call. The bridge already encodes relational
information from the levels below.

### 2.4 Word Vector (L14)

```
v(word_W) = normalize( Σᵢ v(BE_i) + Σⱼ v(sense_j) )
```

where:
- `BE_i` are L15 BaryEdges in which one of W's senses participates
- `sense_j` are orphan senses of W that found no partner at L15

**Why BE-centroid, not sense-centroid:**
Each L15 BE vector already encodes three signals — both senses plus the
word-pair-context type. The word vector absorbs relational information
from every pairing its senses participated in. A word whose senses
paired with diverse partners gets a vector that sits where its senses'
relational neighborhoods overlap — exactly the property needed to bridge
L15 BEs at L13 triad formation.

**Dependency:** L14 word vectors cannot be computed until all L15 BEs
are finalized (including orphan re-entry). Strict stage boundary.

---

## 3. Invariants

1. **Unique parent (soft):** every CM has at most one `parent_edge_id`.
   Orphans allowed. `parent_edge_id` always references a `baryedge`
   document — nodes are CMs, never parent edges.
2. **Triadic recursion only above L14.** No lateral edges, no cross-level
   BEs outside triads.
3. **Forest structure** — single `$graphLookup` climbs to root.
4. **BE and MetaBary interchangeable above L14** — same doc type, same role.
5. **Algebraically closed** — vector construction is always
   `normalize(w₁·a + w₂·b + w₃·c)`, recursive. At L14/L15, `w = q`.
   Above L13, `w = accumulated_weight`.
6. **`connection_strength` always in [0,1].** `accumulated_weight` may
   exceed 1 above L13 — these are distinct fields with distinct roles.

---

## 4. Kaikki Data Structure

One JSONL line = one `(word, pos)` entry. Example: `dictionary (noun)`.

### 4.1 Top-Level Fields

```
word            "dictionary"
pos             "noun"
lang_code       "en"
forms[]         [{form: "dictionaries", tags: ["plural"]}, ...]
etymology_text  full etymology string
sounds[]        [{ipa: "/ˈdɪk.ʃə.nə.ɹi/", tags: ["Received-Pronunciation"]}, ...]
senses[]        array of sense objects (see §4.2)
translations[]  array of translation objects (see §4.3)

— Relations (word-level, NOT sense-level) —
synonyms[]         [{word: "dict", _dis1: "0 0 0 0 0 0"}, ...]
antonyms[]         (sparse — mainly adjectives/verbs, rare for nouns)
hypernyms[]        [{word: "catalog", _dis1: "0 0 0 0 0 0"}, ...]
hyponyms[]         [{word: "bilingual dictionary", _dis1: "0 0 0 0 0 0"}, ...]
meronyms[]         [{word: "vocabularium", _dis1: "0 0 0 0 0 0"}, ...]
derived[]          [{word: "dictionarial", _dis1: "0 0 0 0 0 0"}, ...]
related[]          [{word: "lexicon", _dis1: "0 0 0 0 0 0"}, ...]
coordinate_terms[] [{word: "thesaurus", _dis1: "0 0 0 0 0 0"}, ...]
```

**Critical:** All semantic relations live at the word entry level.
The `_dis1` field carries sense distribution weights for disambiguation.

### 4.2 Sense-Level Fields

```
senses[i]:
  id          "en-dictionary-en-noun-en:Q23622"  (stable unique ID)
  glosses[]   ["A reference work listing words..."]
  examples[]  [{text: "...", type: "example"|"quotation"}]
  tags[]      ["broadly", "figuratively", "derogatory", ...]
  topics[]    ["computing", "engineering", "mathematics"]  (sparse)
  categories[]{name: "Computing", kind: "other", ...}
  wikidata[]  ["Q23622"]  (only on some senses)

  — Sense-level relations (sparse, minority of senses only) —
  hypernyms[]        (e.g. sense[0]: [{word: "wordbook"}])
  coordinate_terms[] (e.g. sense[0]: [{word: "thesaurus"}])
  hyponyms[]         (e.g. sense[5]: [{word: "hash table"}])
```

When a sense carries its own relation fields, these take priority over
word-level relations for that sense — no disambiguation needed.

### 4.3 Translation Structure

```
translations[i]:
  lang       "Abkhaz"
  lang_code  "ab"
  sense      "publication that explains the meanings of an ordered list of words"
  word       "ажәар"
  _dis1      "24 8 2 28 25 12"   ← non-zero: dominant sense = index 3
```

Translations carry a `sense` gloss string and non-zero `_dis1` weights —
the most precisely sense-disambiguated cross-language signal in the corpus.

### 4.4 `_dis1` Sense Disambiguation

`_dis1` is a space-separated string of integers, one per sense.

```python
def assign_sense(item: dict, sense_vectors: list, threshold: float = 0.72) -> int | None:
    weights = [int(x) for x in item['_dis1'].split()]
    if max(weights) > 0:
        return weights.index(max(weights))   # use _dis1 directly → L15
    target_vec = embed(item['word'])
    sims = [cosine(target_vec, sv) for sv in sense_vectors]
    if max(sims) > threshold:
        return sims.index(max(sims))         # cosine fallback → L15
    return None                              # assign to word level (L14)
```

---

## 5. Hierarchy Mapping

| Level | Scale | Kaikki Source | PoC Status |
|---|---|---|---|
| 1–3 | Language family / Paradigm | Fixed: "English", "Germanic", "Indo-European" | Static scaffolding |
| 4–6 | Register / Period | `tags[]`: formal, archaic, slang, technical | Sparse — collapse to L7 if <5% coverage |
| 7–9 | Semantic field / POS cluster | `topics[]`, `pos`, sense `categories[]` | Active |
| 10–12 | Concept cluster (synset) | Clustered senses sharing hypernyms | Active |
| 13 | MetaBary (polysemy bridge) | L15 BE pairs bridged by L14 BE | Active |
| 14 | Word entry | One node per `(word, pos)` — kaikki relation matching | Active |
| 15 | Individual sense / Gloss | Each `senses[]` entry — cosine-matched BEs | Active |

---

## 6. Data Schema

Single collection `barygraph`. Two document types: `node`, `baryedge`.

### 6.1 Node

```python
{
    '_id':            ObjectId(),
    'doc_type':       'node',
    'node_type':      'sense' | 'word' | 'synset' | 'field' | 'register' | 'stub',
    'level':          int,          # 1–15
    'label':          str,
    'vector':         list[float],  # 768-dim
    'surface':        int,
    'rotation':       0.0,
    'parent_edge_id': ObjectId() | None,  # ≤1 parent BE; None = orphan
    'properties':     dict,         # see node_type table below
    'created_at':     datetime,
    'updated_at':     datetime,
}
```

| node_type | level | key properties | vector source |
|---|---|---|---|
| `sense` | 15 | word, pos, sense_id, sense_idx, gloss, examples, tags, topics, wikidata | embed(gloss + examples[:2]) |
| `word` | 14 | word, pos, etymology, forms, ipa | BE-centroid + orphan senses |
| `synset` | 10–12 | hypernym, member_count | cluster centroid |
| `field` | 7–9 | name, pos_group | cluster centroid |
| `register` | 4–6 | name, tag — **may collapse to L7** | cluster centroid |
| `stub` | any | word, reason — no vector | none |

**Word node `properties`:**

```python
'properties': {
    'word':        'dictionary',
    'pos':         'noun',
    'etymology':   'From Middle English dixionare...',
    'forms':       ['dictionaries'],
    'ipa':         '/ˈdɪk.ʃə.nə.ɹi/',
}
```

### 6.2 BaryEdge

```python
{
    '_id':                ObjectId(),
    'doc_type':           'baryedge',
    'cm1_id':             ObjectId(),   # → node (L14/L15) or baryedge (L≤13)
    'cm2_id':             ObjectId(),   # → node (L14/L15) or baryedge (L≤13)
    'level':              int,          # same as CMs at L14/L15;
                                        # = cm1.level - 2 for MetaBary (L≤13)
    'vector':             list[float],  # bary_vec — algebraic
    'parent_edge_id':     ObjectId() | None,  # ≤1 parent; always → baryedge; None = orphan
    'connection_strength': float,       # q (L14/L15) or q_MB_raw (L≤13); always in [0,1]
    'accumulated_weight': float,        # = q at L14/L15; = q_MB_raw·level_factor at L≤13
                                        # may exceed 1 above L13; passed to parent level

    # L14/L15 ONLY:
    'edge_type':          str | None,   # kaikki relation (L14) or None (L15 cosine-matched)
    'type_vector':        list[float],  # v(type) — per-pair embed (L15) or TYPE_SENTENCES (L14)
    'q':                  float,        # 0–1
    'source':             str,          # 'ingested' | 'inferred' | 'manual' | 'placeholder'
    'confidence':         float,

    # ABSENT above L14:
    # edge_type, type_vector, q, source, confidence

    'created_at':  datetime,
    'updated_at':  datetime,
}
```

**What's dropped above L14:**

| Removed | Why |
|---|---|
| `edge_type` | Type is implicit in bridge BE vector |
| `is_metabary` flag | Everything above L14 is MB by construction |
| `hierarchy_direction` | Always upward; no lateral edges |
| `common_ancestor_id` | Forest structure makes traversal trivial |
| Lateral MetaBary | Eliminated by unique-parent constraint |

---

## 7. Edge Types (L14/L15 only)

### 7.1 Fermion Order (L14 matching priority)

L14 BaryEdge matching follows fermion order — rarer, more informative
relations matched first. Once a word has `parent_edge_id` set, it is
skipped at lower-priority tiers.

| Priority | edge_type | kaikki field | q_seed |
|---|---|---|---|
| 1 | `contradicts` | `antonyms[]` | 0.85 |
| 2 | `applies_to` | `meronyms[]`, `holonyms[]` | 0.55 |
| 3 | `is_instance_of` | `hypernyms[]`, `hyponyms[]` | 0.65 |
| 4 | `extends` | `derived[]`, `related[]` | 0.60 |
| 5 | `same_phenomenon` | `coordinate_terms[]` | 0.70 |
| 6 | `same_phenomenon` | `synonyms[]` | 0.90 |

Tiers 5 and 6 share `edge_type` but differ in q_seed — keyed separately
in `Settings.q_seeds` by kaikki field name.

### 7.2 L15 Cosine-Matched Edges

L15 BaryEdges are formed by greedy highest-cosine matching across all
sense pairs. No edge_type — the relationship is captured entirely by
`v(type)` (lexical neighborhood embed).

| Parameter | Value |
|---|---|
| Matching method | Greedy highest-cosine first |
| q | `cos(s_A, s_B)` directly |
| Threshold (q_min) | 0.72 |
| v(type) | embed(word neighborhood text) |
| Orphan v(type) | embed(orphan's word neighborhood only) |

### 7.3 Polysemy Edges (L15, same headword)

Same-headword sense pairs enter the L15 greedy-match candidate pool
with a q floor:

```
q = max(0.40, cosine(sense_vec_i, sense_vec_j))
```

Greedy matching still selects at most one parent BE per sense.
The floor ensures polysemous senses remain eligible even when their
raw cosine is low.

---

## 8. Construction Pipeline

### Stage 1 — Sense Nodes (L15)

- Parse kaikki JSONL → extract senses
- `v(sense) = embed(gloss + examples[:2])`
- Store as node: `node_type: 'sense'`, `level: 15`, `parent_edge_id: None`

### Stage 2 — Embed

- Batch embed all sense glosses via nomic-embed-text (~10 min GPU)
- One embedding call per sense

### Stage 3 — Insert Nodes

- Insert L15 sense nodes into MongoDB
- Insert L14 word nodes with placeholder vectors (updated in Stage 5)

### Stage 4 — L15 BE Formation (cosine-driven)

```
4a. Pairwise cosine among all L15 sense vectors (ANN-accelerated)
4b. Greedy match: highest cosine first, skip already-paired
4c. For each pair: build type_text (parent words + antonyms + synonyms),
    batch-embed → v(type)
4d. Compute bary_vec, set q, set accumulated_weight = q, set parent_edge_id
4e. Orphan re-entry: unpaired senses match with existing L15 BEs
```

**Scale:** ~300K senses → use ANN (FAISS or hnswlib). Top-k neighbors
per sense, then greedy match from ranked pairs.

**Embedding cost:** ~1–2M v(type) calls, batchable at 1K → ~1–2K batches.

### Stage 5 — L14 Word Vectors

```python
for word_node in word_nodes:
    be_vecs     = [be['vector'] for be in get_baryedges_for_word(word_node)]
    orphan_vecs = [s['vector'] for s in get_orphan_senses(word_node)]
    raw = sum(be_vecs + orphan_vecs)
    word_node['vector'] = normalize(raw)
```

No embedding call. Strict stage boundary — runs after L15 BE formation.

### Stage 6 — L14 BE Formation (kaikki-driven, fermion order)

```
6a. Iterate kaikki relations in fermion order (§7.1)
6b. Skip words already paired at this priority tier
6c. v(type) = embed(TYPE_SENTENCES[edge_type])
6d. Compute bary_vec, set q, set accumulated_weight = q, set parent_edge_id
```

L14 orphan re-entry (`s06_orphan_reentry.py`): each unpaired word
matches the nearest existing L14 BE; new BE inherits that partner's
`edge_type`, `type_vector`, `q`, and `accumulated_weight` (no new
embedding call).

### Stage 7 — L13 MetaBary (polysemy bridge)

For each L14 BE (bridge), find two unparented L15 BEs with mutual
`cos > 0.9`:

```python
ALPHA = 0.5  # level_factor tuning parameter

def level_factor(level: int) -> float:
    return 1.0 + ALPHA * (14 - level) / 13

def compute_metabary(be1: dict, be2: dict, bridge: dict, level: int) -> dict:
    w1 = be1['accumulated_weight']
    w2 = be2['accumulated_weight']
    w3 = bridge['accumulated_weight']

    # Vector direction: pull toward highest-authority children
    raw_vec = w1 * be1['vector'] + w2 * be2['vector'] + w3 * bridge['vector']
    bary_vec = normalize(raw_vec)

    # Accumulated weight: Born rule + level amplification
    q_mb_raw = w3**2 / (w1**4 + w2**4 + w3**4) ** 0.5
    acc_w    = q_mb_raw * level_factor(level)

    return {
        'vector':             bary_vec,
        'connection_strength': q_mb_raw,   # always in [0,1]
        'accumulated_weight': acc_w,        # may exceed 1 above L13
        'level':              level,
        ...
    }
```

Example polysemy triad:

```
word "bank" + word "flow"  → L14 BE (via kaikki: related)
    ↑ bridge
sense "bank/financial" + sense "credit/loan"  → L15 BE₁
sense "bank/riverbank" + sense "flow/stream"  → L15 BE₂
    → L13 MB: financial↔river polysemy bridged by flow concept
```

### Stage 8 — L12→L1 Recursive

```python
while True:
    new_triads = 0
    for bridge in get_unparented_bes(level=current_level - 1):
        candidates = find_unparented_bes_near(bridge, level=current_level, threshold=0.9)
        if len(candidates) >= 2:
            be1, be2 = candidates[:2]
            mb = compute_metabary(be1, be2, bridge, level=current_level - 2)
            insert_metabary(mb)
            new_triads += 1
    if new_triads == 0:
        break
    current_level -= 2
```

Pure geometry — no kaikki, no TYPE_SENTENCES. `accumulated_weight`
compounds at each level via `level_factor`.

### Stage 9 — MetaBary Extension (Threshold Relaxation)

Picks up where Stage 8 left off (`s09_extend.py`). Rather than stopping
at a fixed 0.9 cosine threshold, this stage sweeps the threshold down to
a per-level floor until no new triads form:

```python
def _min_threshold(level: int) -> float:
    """Floor per child_level: 0.9 - level × 0.01."""
    return round(0.9 - level * 0.01, 2)
    # L15 → 0.75, L14 → 0.76, L13 → 0.77, L12 → 0.78, …
```

For each child level, all unparented BEs and bridges are loaded from
MongoDB **once**, then the full threshold sweep is applied in memory —
14× faster than re-fetching per threshold round. Pairs are sorted
cosine-descending so greedy selection is identical to a round-by-round
approach.

When the bridge count exceeds `ANN_THRESHOLD`, a **CPU HNSW index**
(hnswlib) is built from bridge vectors to accelerate centroid lookup.
Otherwise a brute-force dot product is used.

The outer loop repeats full descending passes (L15 children first,
upward) until a complete pass produces **zero new triads** across all
levels — natural convergence, no hard cap.

No embedding calls. **Resumable:** `parent_edge_id = None` is the
ground truth for unparented BEs; the checkpoint records cumulative
triads for observability but does not gate re-processing on resume.

**Per-level floor table:**

| child_level | floor threshold | bridge_level |
|---|---|---|
| L15 | 0.75 | L14 |
| L14 | 0.76 | L13 |
| L13 | 0.77 | L12 |
| L12 | 0.78 | L11 |
| L11 | 0.79 | L10 |

### Stage 10 — Index

- Build mongot vector index (~4–8 hours)

### Pipeline Timing

| Stage | Script | Duration | Blocking |
|---|---|---|---|
| 1. Parse | `s01_parse.py` | ~10 min | Yes |
| 2. Embed | `s02_embed.py` | ~20 min | Yes |
| 3. Insert nodes | `s03_insert_nodes.py` | ~30 min | Yes |
| 4. L15 BE formation | `s04_l15_edges.py` | ~45 min | Yes |
| 5. L14 word vectors | `s05_word_vectors.py` | ~5 min | Yes |
| 6. L14 BE formation | `s06_l14_edges.py` | ~30 min | Yes |
| 7. L14 orphan re-entry | `s07_orphan_reentry.py` | ~10 min | Yes |
| 8. MetaBary L13→L1 | `s08_metabary.py` | ~1–2 hours | Yes |
| 9. MetaBary extension | `s09_extend.py` | ~variable | Yes |
| 10. Index | `s10_index.py` | ~4–8 hours | Yes |

**Queryable: ~7–12 hours.**

### Resumability

```json
// pipeline_state/{stage_name}.json
{ "last_id": "ObjectId(...)", "processed": 1240000, "total": 2500000 }
```

---

## 9. MongoDB Infrastructure

Database: `barygraph_poc`  
Collection: `barygraph`

### 9.1 Standard Indexes

```python
# lib/db.py — run once at setup
db.barygraph.create_index([('doc_type', 1), ('level', 1)])
db.barygraph.create_index([('cm1_id', 1)])
db.barygraph.create_index([('cm2_id', 1)])
db.barygraph.create_index([('node_type', 1)])
db.barygraph.create_index([('edge_type', 1), ('level', 1)])
db.barygraph.create_index([('parent_edge_id', 1)])
db.barygraph.create_index([('properties.word', 1), ('properties.pos', 1)])
db.barygraph.create_index([('properties.sense_id', 1)])
```

### 9.2 Vector Index

```json
{
  "fields": [
    { "type": "vector", "path": "vector", "numDimensions": 768, "similarity": "cosine" },
    { "type": "filter", "path": "doc_type" },
    { "type": "filter", "path": "level" },
    { "type": "filter", "path": "edge_type" },
    { "type": "filter", "path": "node_type" }
  ]
}
```

---

## 10. Query Patterns

MongoDB aggregation syntax is language-agnostic — shown as query
structures, executed via `pymongo`'s `aggregate()`.

### Baseline (flat)
```
filter: { doc_type: 'node', level: { $in: [14, 15] } }
```

### BaryGraph retrieval
```
filter: { doc_type: { $in: ['node', 'baryedge'] }, level: { $in: [14, 15] } }
# + $lookup on cm1_id, cm2_id
```

### Hierarchy traversal (forest walk via parent_edge_id)
```
$graphLookup:
  from: 'barygraph'
  startWith: '$parent_edge_id'
  connectFromField: 'parent_edge_id'
  connectToField: '_id'
  as: 'upward_chain'
  maxDepth: 15
```

Forest structure means no cycle handling, no `restrictSearchWithMatch`
needed. Single `$graphLookup` walks from any node to root.

---

## 11. Evaluation

### 11.1 Primary: Held-Out Synonym Recall@20

1. Hold out 10% of `synonyms[]` links → `data/holdout.json`
   (**synonyms only** — antonyms too sparse for noun-heavy corpus)
2. Ingest 90%
3. Query `embed(word_A gloss)`, filter `doc_type: 'baryedge'`, top-20
4. **Success:** word_B in CM lineage of any returned BaryEdge
5. BaryGraph recall@20 vs. flat recall@20

### 11.2 Secondary Metrics

| Metric | Method |
|---|---|
| `bary_vec` precision | BaryEdge 5-NN should include cm1 and cm2 |
| Disambiguation accuracy | Sample 100 word-level relations; manually verify sense assignment |
| `_dis1` vs cosine | Precision when `_dis1 > 0` vs cosine fallback |
| Forest coherence | Manual top-50 triad inspection at L13 |
| Hierarchy correctness | Forest paths vs. `topics[]` / `categories[]` |
| Fermion order impact | Recall with/without antonym-first priority |
| Orphan rate per level | Track % of CMs with parent_edge_id = None |
| Accumulated weight distribution | Histogram of `accumulated_weight` per level; verify compounding |
| α sensitivity | Recall@20 at α ∈ {0.0, 0.25, 0.5, 1.0} |

---

## 12. Defaults and Tuning Parameters

| ID | Parameter | Default | Rationale |
|---|---|---|---|
| R1 | L15 matching order | Greedy highest-cosine first | Sense-level kaikki too sparse to drive ordering |
| R2 | L15 q value | `cos(s_A, s_B)` directly | Natural quality signal |
| R3 | L15 orphan v(type) | embed(orphan's word neighborhood only) | One-sided but consistent |
| R4 | Antonym/synonym _dis1 filtering | No filtering initially | Simpler; revisit if eval shows noise |
| R5 | L15 matching threshold | 0.72 | Below this, sense remains orphan |
| R6 | Level factor α | 0.5 | L1 MB gets 1.5× raw q_MB; tune from secondary eval |

---

## 13. Resource Budget

| Stage | Duration | Blocking? |
|---|---|---|
| Parse + Embed | ~30 min | Yes |
| Insert nodes | ~30 min | Yes |
| L15 BE formation | ~45 min | Yes |
| L14 word vectors | ~5 min | Yes |
| L14 BE formation | ~30 min | Yes |
| L14 orphan re-entry | ~10 min | Yes |
| MetaBary L13→L1 | ~1–2 hours | Yes |
| MetaBary extension | ~variable | Yes |
| Build indexes | ~4–8 hours | Yes |
| **Queryable** | **~8–14 hours** | |

Hardware: 8–16 GB GPU VRAM, 32–64 GB RAM, 150–200 GB disk, 8+ cores.  
Cost: zero (all open-source).

---

## 14. Deviations From Parent Spec (BaryGraph v1.2)

| v1.2 Section | Deviation | Reason |
|---|---|---|
| §4.2 — pairwise cosine scan all levels | Explicit-relation seeding at L14/L15 | O(n²) infeasible |
| §7 — MongoDB Atlas | Community + mongot | Local-first PoC |
| Embedding dimensions | 768-dim (not 1536) | Glosses are short |
| `v(type)` as edge label | Per-pair neighborhood embed (L15), TYPE_SENTENCES (L14 only) | Bare labels embed poorly |
| All relations at sense level | Word-level with `_dis1`/cosine disambiguation | Actual kaikki structure |
| Lateral MetaBary | Eliminated | Forest structure with unique-parent |
| `is_metabary` flag | Eliminated | Redundant — level > 14 implies MetaBary |
| Word vectors = sense centroid | BE-centroid + orphan senses | Carries relational information from all pairings |
| `registry.summary` / `summary_vector` | Dropped | PoC is embedding-only |
| Uniform q in MetaBary formula | `accumulated_weight` compounds per level | Structural authority should grow with hierarchy depth |

---

## 15. What v0.5 Gains Over v0.4

- **Cleaner word vector.** Removing φ(W) and λ eliminates a spurious
  signal. "use" and "utilization" differ in their gloss embeddings
  already — no orthographic proxy needed.
- **Structurally weighted MetaBary vector.** Children with higher
  `accumulated_weight` pull the MB vector toward their semantic space.
  A bridge that has survived multiple levels of triadic selection
  dominates a freshly-formed L15 BE in the same triad.
- **Compounding authority.** `accumulated_weight` grows multiplicatively
  as the graph climbs. A MetaBary at L5 carries measurably more
  structural authority than one at L13, encoded directly in the scalar
  passed upward.
- **Two-field separation.** `connection_strength` stays in [0,1] for
  interpretability. `accumulated_weight` is the structural propagation
  signal and may exceed 1 — the two are never conflated.

## 16. What v0.5 Costs vs v0.4

- **`accumulated_weight` must be stored and fetched** for every child
  at MetaBary formation time. One extra field read per triad — negligible.
- **α is a new tuning parameter** (R6). Default 0.5 is reasonable but
  requires A/B validation in secondary eval.

---

## 17. Potential Issues

1. **L15 cosine matching at scale.** ~300K senses → 90B pairs brute-force. Use FAISS/hnswlib for ANN.
2. **v(type) embedding calls at L15.** ~100–150K calls. Batchable at 1K → ~150 batches.
3. **Orphan re-entry asymmetry.** Orphan sense paired with existing BE creates structurally asymmetric children. Algebraically fine, but needs a fresh v(type) embed call.
4. **L13 candidate search ambiguity.** Children must be near each other (cos > 0.9 mutual); bridge initiates search but doesn't constrain their mutual similarity.
5. **Sparsity above L12.** Each level roughly halves the node count with cos > 0.9 + unique-parent. Graph may top out before L1.
6. **`accumulated_weight` scale drift.** With α = 0.5, a chain of strong MBs from L13 to L1 could reach `accumulated_weight` ~ 1.5⁶ ≈ 11. Verify distribution in secondary eval; cap if needed.

---

## 18. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| mongot HNSW OOM | Medium | Baryedge subset first; fp16; Qdrant fallback |
| `_dis1` all-zero → poor disambiguation | Medium | Cosine fallback; track accuracy in eval |
| `antonyms[]` too sparse | **Known** | Synonyms only for primary eval |
| `bary_vec` averages to mush | **This is the test** | Tune q / v(type); falsify if no lift |
| Targets not in dump | Medium | Stub nodes; exclude from eval |
| Unique-parent too sparse | Medium | Track orphan rate; relax if >60% orphan |
| L15 ANN quality | Low | Verify recall vs brute-force on 10K sample |
| `accumulated_weight` scale drift | Low | Histogram per level in secondary eval; cap at ceiling if needed |
| α = 0.5 suboptimal | Low | A/B at α ∈ {0.0, 0.25, 0.5, 1.0}; R6 in §12 |

---

## 19. Open Questions

1. **Synset clustering** — agglomerative vs. Leiden on synonym BaryEdge graph
2. **Sparse L4–6** — collapse to L7 if <5% tag coverage
3. **Polysemy q floor** — start 0.40, tune after MetaBary formation
4. **Disambiguation threshold** — 0.72 default; tune via secondary eval
5. **MetaBary stopping criterion** — "no triads form" (natural) or hard cap?
6. **Stub promotion** — inline in stage 4 or separate pass?
7. **Antonym/synonym _dis1 filtering for v(type)** — revisit if L15 eval noisy
8. **`accumulated_weight` ceiling** — should a hard cap be applied to prevent scale drift above L7?
9. **α tuning** — 0.5 default; tune from secondary eval (R6)

---

## 20. Expansion Path

1. **Multi-language** — translations carry `sense` gloss + non-zero `_dis1`, enabling direct sense-level cross-language BaryEdges.
2. **Atlas migration** — identical schema; `mongodump` / `mongorestore`
3. **Live update loop** — `q` decay + incremental refresh
4. **RAG integration** — relationship structures as retrieval context

---

*BaryGraph Kaikki PoC v0.5 · CM Theory Project · April 2026*
