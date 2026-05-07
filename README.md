# BaryGraph

**A knowledge graph architecture where every relationship is a first-class object.**

BaryGraph borrows a principle from physics: in any system of two masses,
the barycenter — the shared center of mass — is as real as the masses
themselves. It has a position, it has properties, and it governs how the
two bodies relate.

Standard knowledge graphs treat relationships as thin edges: a label and
a weight connecting two nodes. BaryGraph promotes every relationship to a
**BaryEdge** — a full document with its own embedding vector, its own
semantic description, and its own position in a retrieval index. When you
search, you find not just similar nodes but the *connections between
nodes*, including connections between nodes that would never appear as
each other's nearest neighbours.

---

BaryGraph: A Semantic Description of the Database

  The Two Atoms

  Everything in this database is one of two things: a node or a
  baryedge. There are currently 6.66 million documents total.
  That's it — no separate edge collections, no join tables, no
  separate metadata stores. The choice to flatten both objects and
  relationships into one collection is not an accident; it is the
  entire point.

  Nodes are things. At the bottom of the graph (level 15) they are
  individual word senses — 1.74 million of them. Each sense is a
  single gloss from the Wiktionary-derived Kaikki corpus: "a device
   for turning a screw," or "a student at Oxford University." One
  level up (level 14) sit 1.44 million word nodes, each
  representing a (word, pos) pair like open/adj or run/verb. These
  two levels are the only node levels in use. Everything above them
   is composed entirely of edges.
  Relationships as First-Class Citizens

  A BaryEdge is a stored relationship that has its own identity,
  its own 768-dimensional embedding vector, and can itself be
  related to other BaryEdges. This is the central move: instead of
  annotating an edge with metadata, you promote the edge to a
  document that can be a child, a sibling, or a parent of other
  documents.

  At level 15 there are 1.11 million BaryEdges pairing sense nodes.   Two senses become a pair when their embedding vectors are close
  enough (cosine ≥ 0.72) and neither is already paired. The
  resulting BaryEdge vector is not the average of its two children
  — it is an algebraic blend weighted by q (a connection strength    between 0 and 1) and a type vector derived from the lexical        neighborhood of the paired words. The type vector encodes what
  kind of neighborhood this pair lives in: which antonyms and
  synonyms the parent words carry. This means two senses that are
  similar for different reasons can have different BaryEdge vectors
   even if their raw embedding distance is identical.

  At level 14 there are roughly 1.39 million BaryEdges pairing word
   nodes, and here the semantics become explicit. Each L14 edge
  carries an edge_type drawn from five named relations:

  - contradicts (293k) — antonyms: the two words push against each
  other
  - extends (890k) — derivational and relational links: one word
  grows out of the other
  - is_instance_of (52k) — hypernym/hyponym: one is a species of
  the other
  - applies_to (4.5k) — meronymy: one is a part of the other
  - same_phenomenon (151k) — synonyms and coordinate terms: both
  words reach toward the same thing

  The dramatic imbalance in these counts is real. extends dominates
   because Wiktionary's derived[] and related[] fields are densely
  populated; applies_to is rare because meronymy is rarely encoded.
   The graph reflects what lexicographers chose to record.

  The Forest Constraint

  Every node and every BaryEdge has at most one parent_edge_id.
  This unique-parent rule enforces a forest topology — a collection
   of trees with no shared ancestry. It means the graph cannot
  represent a sense that belongs equally to two different semantic
  neighborhoods. It must choose. This is a deliberate
  simplification: it makes traversal cheap and hierarchy
  unambiguous, at the cost of forcing a commitment that natural
  language often refuses to make.

  MetaBary: Edges of Edges

  Above level 14 the structure becomes recursive. A MetaBary at
  level 13 is a BaryEdge whose two children (cm1, cm2) are
  themselves L14 BaryEdges, held together by a third L14 BaryEdge
  acting as a bridge. There are 495k such triads. Each one groups
  two word-level relationships under a shared semantic roof.

  A live example from the database illustrates this clearly. One

  A live example from the database illustrates this clearly. One
  L13 MetaBary has:
  - child1: the edge between snow pentathlon and winter triathlon
  (similar winter multi-sport events)
  - child2: the edge between geschmozzle and snowboard cross (two
  snowboard racing formats)
  - bridge: the edge between pentathlon and snow pentathlon (a
  hyponym link anchoring the cluster)

  Together these four words and three relationships form a single
  document representing the concept "competitive winter
  multi-discipline sport." The MetaBary's own vector is an
  algebraic blend of its children's vectors, re-weighted by a
  rescaled connection strength q_MB that accounts for the
  compounding of three q values. A strong MetaBary has children
  This unique-parent rule enforces a forest topology — a collection
   of trees with no shared ancestry. It means the graph cannot
  represent a sense that belongs equally to two different semantic
  neighborhoods. It must choose. This is a deliberate
  simplification: it makes traversal cheap and hierarchy
  unambiguous, at the cost of forcing a commitment that natural
  language often refuses to make.

  MetaBary: Edges of Edges

  Above level 14 the structure becomes recursive. A MetaBary at
  level 13 is a BaryEdge whose two children (cm1, cm2) are
  themselves L14 BaryEdges, held together by a third L14 BaryEdge
  acting as a bridge. There are 495k such triads. Each one groups

  A live example from the database illustrates this clearly. One
  L13 MetaBary has:
  - child1: the edge between snow pentathlon and winter triathlon
  (similar winter multi-sport events)
  - child2: the edge between geschmozzle and snowboard cross (two
  snowboard racing formats)
  - bridge: the edge between pentathlon and snow pentathlon (a
  This unique-parent rule enforces a forest topology — a collection
   of trees with no shared ancestry. It means the graph cannot
  represent a sense that belongs equally to two different semantic
  neighborhoods. It must choose. This is a deliberate
  simplification: it makes traversal cheap and hierarchy
  unambiguous, at the cost of forcing a commitment that natural      language often refuses to make.

  MetaBary: Edges of Edges                                         
  Above level 14 the structure becomes recursive. A MetaBary at
  level 13 is a BaryEdge whose two children (cm1, cm2) are
  themselves L14 BaryEdges, held together by a third L14 BaryEdge
  acting as a bridge. There are 495k such triads. Each one groups    two word-level relationships under a shared semantic roof.
                                                                     A live example from the database illustrates this clearly. One
  L13 MetaBary has:                                                  - child1: the edge between snow pentathlon and winter triathlon
  (similar winter multi-sport events)
  - child2: the edge between geschmozzle and snowboard cross (two
  snowboard racing formats)
  - bridge: the edge between pentathlon and snow pentathlon (a
  hyponym link anchoring the cluster)

  Together these four words and three relationships form a single
  document representing the concept "competitive winter
  multi-discipline sport." The MetaBary's own vector is an           algebraic blend of its children's vectors, re-weighted by a
  rescaled connection strength q_MB that accounts for the
  compounding of three q values. A strong MetaBary has children      with high mutual cosine similarity; a weak one (like the
  jivanmukta/jism cluster at q=0.31) is a cosmological
  neighbor-of-last-resort pairing, not a semantic insight.         
  This recursion continues upward. L12 MetaBary edges pair L13       MetaBary edges (424k of them). L11 pairs L12 (35k). L10 pairs L11
   (35k). At each level the vocabulary covered by a single document   grows — the software-testing cluster at L12 encompasses
  Microspeak, bluelink, open beta, machete beta, alpha version, and   version under a single node — while the connection strength and
  semantic precision generally decrease.
                                                                     What the Vectors Encode                                          
  Every document carries a 768-dimensional vector from the
  nomic-embed-text-v1.5 model, but what each vector means differs
  by level:

  - L15 sense vectors: embed the gloss text plus up to two example   sentences. They encode what a specific usage of a word means in
  context.                                                           - L14 word vectors: no embedding call. Computed as the normalized
   sum of all BaryEdge vectors that hold that word's senses, plus    any unpaired sense vectors. A word's vector is literally the
  center of mass of its relationships.                               - L14/L15 BaryEdge vectors: algebraically derived from children
  and type. Never embedded directly.
  - L13–L10 MetaBary vectors: algebraically derived from their
  children. The embedding model is never called above L14.

  This means the model runs for L15 sense text and L14 type          sentences only. Every higher-level structure is pure algebra on    top of those anchors.

  The Practical Consequence                                                                                                             What you have is a six-million-document index where searching for
   "fast-moving water" doesn't just find nodes whose text mentions
  rapids or torrents — it can surface a BaryEdge that pairs rapids
  with cataract, whose MetaBary parent groups that pair with
  whitepool/maelstrom, under a grandparent spanning the whole
  domain of turbulent water movement. Retrieval can return not just
   words but the relationship between words, and that relationship
  is itself a searchable, rankable object with a vector position in
   the same 768-dimensional space as every sense and word in the
  graph.

---

## The Hypothesis

> Including BaryEdge documents in vector search retrieval outperforms
> flat nearest-neighbour search on the same corpus.

Concretely: given a query, a retrieval system that returns a mix of nodes
and BaryEdges should recover more relevant items than one returning nodes
alone — because BaryEdges act as bridges, pulling in both their parent
nodes as implied context.

This is falsifiable. We test it with held-out synonym links from a
dictionary corpus. If BaryGraph recall@20 does not beat flat recall@20,
the architecture does not justify its complexity.

---

## This Repository: Kaikki PoC

The proof-of-concept uses the English machine-readable dictionary from
[kaikki.org](https://kaikki.org) (~800K headwords, ~2.5M senses).

A dictionary is an ideal first testbed because:

- **Relations are pre-labeled.** Synonyms, antonyms, derived forms,
  etymology, and hypernyms are explicit in the data. BaryEdge types come
  from the corpus, not a classifier.
- **Ground truth is free.** Hold out 10% of synonym links before
  ingestion. Measure whether BaryGraph retrieval recovers them better than
  flat search. Zero human annotation required.
- **Polysemy is rich.** Words like *bank*, *crane*, and *bark* have
  senses so distant in meaning that no embedding will make them neighbours.
  Yet they are deeply related. BaryGraph should surface that structure
  through MetaBary triads — recursive relationships between relationships.
- **Any bilingual person is an oracle.** Unlike scientific papers or legal
  cases, evaluating whether two word senses are genuinely connected
  requires no domain expertise. The results are immediately human-readable.

---

## Core Concepts

### The Triad

Every connection is three entities, not two:

```
CM₁  →  BaryEdge  →  CM₂
```

`CM₁` and `CM₂` are knowledge nodes (senses, words, concepts). The
`BaryEdge` sits between them as a stored document with its own vector:

```
bary_vec = normalize( q·v(CM₁) + q·v(CM₂) + (1−q)·v(type) )
```

where `q` is connection quality (0–1) and `v(type)` is a level-dependent
embedding that captures the relational context of the pairing. A strong
connection (`q → 1`) produces a barycenter vector close to both parents
simultaneously — meaning a query near either parent also lands near the
BaryEdge, and vice versa.

### Forest Structure

BaryGraph organizes all nodes and BaryEdges into a forest with a
unique-parent constraint: every CM has at most one `parent_edge_id`.
This means:

- **Triadic recursion:** at higher levels, BaryEdges act as CMs for new
  connections. Two BaryEdges are bridged by a third, forming a MetaBary
  triad that climbs the hierarchy.
- **Single `$graphLookup`** walks from any node to root — no cycle
  handling, no special traversal logic.
- **Orphans are allowed** — a concept with no relationships simply has
  no upward path.

### v(type) — Level-Dependent Context

The third component of the bary_vec formula is not a fixed label. It
varies by level:

| Level | v(type) source | Why |
|---|---|---|
| L15 (senses) | Per-pair embed of both words' lexical neighborhoods (antonyms + synonyms) | Rich contextual signal per pairing |
| L14 (words) | Fixed TYPE_SENTENCES per edge type | Kaikki relations provide structure |
| L13+ (MetaBary) | Bridge BaryEdge vector directly | Already encodes relational info from below |

### The Registry

Each BaryEdge stores a `registry.summary`: a one-sentence natural-language
description of *what the connection means*, generated by a local LLM.
This summary is embedded separately as `summary_vector` — a second signal
independent of the structural `bary_vec`. The evaluation measures both
signals independently before deciding whether to merge them.

### Hierarchy

Nodes and BaryEdges are organized into 15 levels from individual sense
glosses (level 15) up to language family (level 1). BaryEdges connect
nodes at the same level. Cross-level hierarchy emerges from MetaBary
triads, where two BaryEdges at level L are bridged by a BaryEdge at
level L-1, forming a MetaBary at level L-2.

### Two Vectors Per BaryEdge

| field | what it encodes | how produced |
|---|---|---|
| `vector` | structural position — weighted mixture of both parent vectors + type context | algebraic formula, zero embedding calls |
| `summary_vector` | natural-language meaning of the connection | embed(`registry.summary`) |

---

## Stack

Everything runs locally. Zero cloud dependencies. Zero cost.

| component | role |
|---|---|
| MongoDB Community 8.x + mongot | storage, graph traversal, vector search |
| nomic-embed-text-v1.5 | 768-dim embeddings |
| Llama 4 Scout Q4 | selective registry.summary generation |
| llama.cpp / ollama | LLM + embedding runtime |

---

## How It Works

```
kaikki-en.jsonl
      │
      ▼
  1. Parse + Embed    extract senses, relations; embed sense glosses
  2. Insert nodes     sense (L15) + word (L14) nodes → MongoDB
  3. L15 BaryEdges    cosine-driven greedy matching of sense pairs
  4. Word vectors     BE-centroid + orphan senses (no embedding call)
  5. L14 BaryEdges    kaikki relations in fermion order (antonyms first)
  6. Orphan re-entry  unpaired CMs absorbed into existing BEs
  7. MetaBary         L13 triads + recursive L12→L1
  8. Summarize        selective LLM registry.summary generation (~3 days, async)
  9. Index            build mongot vector indexes
      │
      ▼
  queryable in ~7–12 hours
  fully enriched in ~4 days
```

The system is queryable after stage 9 without waiting for LLM summaries.
BaryEdges are retrievable via `bary_vec` immediately; `summary_vector`
enriches them asynchronously.

---

## Retrieval Difference

**Standard vector search** — query returns 20 nodes ranked by cosine
similarity. Finds things that *look like* the query.

**BaryGraph retrieval** — query returns a mix of nodes and BaryEdges.
Each BaryEdge result implies its two parent nodes as context. Effective
retrieved context is 2–3x the raw top-20. Finds things that *connect to*
what the query is about, not just things that resemble it.

**Cross-domain bridge query** — filter for `edge_type: 'same_phenomenon'`.
Returns only BaryEdges connecting nodes from different semantic fields.
This is the retrieval that standard vector search cannot do: finding the
structural bridge between two concepts that share no surface similarity.

---

## Evaluation

The primary test is simple and binary:

1. Hold out 10% of explicit synonym links before ingestion
2. Ingest the remaining 90%
3. For each held-out pair, query the held-out word's gloss
4. **Success:** the partner word appears in the CM lineage of any top-20 result
5. Compare BaryGraph recall@20 vs. flat recall@20

BaryGraph must beat flat retrieval to justify the architecture. If it
does not, the `summary_vector` signal is tested as an alternative primary
vector. If neither beats flat, the hypothesis is falsified.

---

## Expansion Path

If the PoC validates the hypothesis:

1. **Multi-language** — add French, German, Japanese kaikki dumps.
   Translation BaryEdges become cross-language bridges. MetaBary encodes
   the same metaphor pattern across languages: *"il pleut des cordes"*
   (French) <-> *"it's raining cats and dogs"* (English) — both encode
   rainfall intensity through culturally-specific impossibility. This is
   the original motivation for the architecture.

2. **Atlas migration** — identical schema; `mongodump` / `mongorestore`

3. **Other corpora** — the architecture is domain-agnostic. Immediate
   candidates: legal case law (precedent chains), vessel tracking
   (port-call anomaly detection), scientific papers (cross-domain
   anomaly clustering).

4. **RAG integration** — BaryGraph as a retrieval backend for an LLM
   that returns relationship structures, not just similar documents.

---

## Project Structure

```
barygraph-kaikki/
├── CLAUDE.md                  # development guide for AI-assisted coding
├── README.md                  # this file
├── BaryGraph_v1.1.md          # parent architecture spec (v1.2)
├── BaryGraph_Kaikki_PoC_v0.4.md  # full PoC spec (v0.4)
├── pyproject.toml
├── docker-compose.yml         # MongoDB (+ mongot) and ollama
├── Makefile
├── .env.example
├── data/
│   └── kaikki-en.jsonl        # download from kaikki.org (not in repo)
├── pipeline_state/            # resumability checkpoints (gitignored)
├── indexes/
│   └── vector_index.json
├── lib/
│   ├── config.py  log.py  checkpoint.py  db.py
│   ├── embed.py   llm.py
│   ├── bary_vec.py            # bary_vec / metabary / word_vector formulas
│   └── disambiguate.py        # _dis1 + cosine sense assignment
├── scripts/
│   ├── _base.py               # shared CLI bootstrap + `bary` dispatcher
│   ├── s01_parse.py … s10_index.py
│   ├── dev/make_fixture.py
│   └── eval/
│       ├── holdout.py  recall.py  ab_summary.py
└── tests/
    ├── fixtures/kaikki-sample.jsonl
    ├── unit/
    └── integration/
```

---

## Getting Started

```bash
# 1. Install (Python 3.11+)
cp .env.example .env
make install                    # pip install -e ".[dev]"

# 2. Start local services (project-owned Mongo on port 27117 — won't touch
#    any other Mongo you may have on 27017).
make up                         # MongoDB Community 8 + mongot (atlas-local)
make up-gpu                     # + ollama (requires NVIDIA GPU), optional
docker exec barygraph-ollama ollama pull nomic-embed-text:v1.5

# 3. Download kaikki English dump (idempotent + resumable)
make fetch-kaikki

# 4. Verify the environment before kicking off a multi-day ingest
make preflight                  # mongo, ollama, embed dim, dump size, disk, dirs

# 5. Run ingestion stages in order
python -m scripts.s01_parse
python -m scripts.s02_embed
python -m scripts.s03_insert_nodes
python -m scripts.s04_l15_edges
python -m scripts.s05_word_vectors
python -m scripts.s06_l14_edges
python -m scripts.s07_orphan_reentry
python -m scripts.s08_metabary
python -m scripts.s09_summarize    # async — system is queryable before this completes
python -m scripts.s10_index
# or: make pipeline

# 6. Run evaluation
python -m scripts.eval.holdout     # generate holdout set first
python -m scripts.eval.recall      # measure BaryGraph vs flat recall@20
```

### Development

```bash
make lint        # ruff + mypy
make test        # unit tests (no services)
make test-int    # integration tests (requires `make up`)
```

CI (GitHub Actions) runs lint + unit on every push, and an integration
smoke test against a MongoDB service container using fake embed/LLM
backends — no GPU required.

Hardware for full ingestion: 32 GB GPU VRAM (Llama Scout Q4), 32 GB+ RAM,
150 GB disk.

---

## Background

BaryGraph is a proof-of-concept for an architecture derived from
[CM Theory](https://github.com/), a unified physics framework in which
every physical phenomenon emerges from hierarchical triads of centers of
mass connected through barycenters. The insight that a barycenter is as
real as the masses it connects — that the *relationship* is a first-class
entity, not a thin pointer — transfers directly to knowledge representation.

The name "BaryEdge" comes from this origin: an edge that is itself a
barycenter, with position, weight, and semantic content of its own.

---

## Status

- [x] Architecture specification (BaryGraph v1.2)
- [x] Kaikki PoC specification (v0.4)
- [ ] Ingestion pipeline implementation
- [ ] Evaluation harness
- [ ] First results

---

*BaryGraph Kaikki PoC · CM Theory Project · April 2026*
