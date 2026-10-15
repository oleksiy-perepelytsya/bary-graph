# Tensegrity pipeline

A parallel ingestion pipeline to the Kaikki PoC (`scripts/s01..s10`). Where the
PoC ingests **dictionary entries** (L14 words / L15 senses), this pipeline
ingests **documents** and builds a BaryGraph-shaped graph out of them:

| Level | Meaning | Source |
| ----- | ------- | ------ |
| **L15** | document **sentences** | split from the input text; embedded (qwen0.6, 1024-d) |
| **L14** | section / paragraph **titles** | detected heuristically; **no native embedding** — vectors derived in t05 |
| **L13…L1** | **MetaBary** triads | built recursively from role-tagged BaryEdges |

Every BaryEdge carries a **role** — one of the kaikki relation types
(`contradicts`, `applies_to`, `is_instance_of`, `extends`, `same_phenomenon`) —
decided by the **tev1** decision model, and a modified barycentric vector whose
*sign* encodes that role (`contradicts` repels / "tension"; the other four
attract / "compression"). That signed role is what makes this a *tensegrity*
structure rather than a plain similarity graph.

Storage: a dedicated Mongo database **and** collection, both named
`tensegrity` (see `.env.tensegrity`). The MCP server never sources this profile;
it keeps serving the PoC from `.env`.

## Quick start

```bash
# 1. Put a text/markdown document somewhere, then run the whole pipeline:
make tensegrity DOC=docs/my_paper.md

# Or stage by stage (default profile is `tensegrity`):
python -m tensegrity_pipeline 01_parse --doc docs/my_paper.md
python -m tensegrity_pipeline 02_embed
python -m tensegrity_pipeline 03_insert_nodes
python -m tensegrity_pipeline 04_l15_edges
python -m tensegrity_pipeline 05_title_vectors
python -m tensegrity_pipeline 06_l14_edges
python -m tensegrity_pipeline 07_orphan_reentry
python -m tensegrity_pipeline 08_metabary
python -m tensegrity_pipeline 09_extend
python -m tensegrity_pipeline 10_index
```

Common flags (every stage): `--reset`, `--limit N`, `--dry-run`,
`--batch-size N`, `--force`, `--env NAME`.
Stage-specific: `--doc PATH` (t01, **repeatable** — pass multiple documents in
one run; each keeps its filename stem as `doc_id`), `--phase pair|absorb|both`
(t07), `--max-level N` (t08/t09).

Offline / CI: set `BARY_FAKE_EMBED=1` (deterministic vectors) and
`BARY_FAKE_DECISION=1` (deterministic tev1 roles). No ollama needed.

## Stages

| Stage | Does |
| ----- | ---- |
| `01_parse` | split the input document(s) into L15 sentences + L14 titles (`parse.py`; `--doc` repeatable, per-doc `doc_id` from filename stem) |
| `02_embed` | embed **sentence texts only**; titles get no embedding |
| `03_insert_nodes` | upsert L15 sentence nodes + L14 title nodes (`vector=None` placeholder) |
| `04_l15_edges` | cosine-match L15 sentences → BaryEdges, classify role with tev1 |
| `05_title_vectors` | derive L14 title vectors from their L15 BaryEdges (no embedding call) |
| `06_l14_edges` | cosine-match L14 titles → role-tagged BaryEdges |
| `07_orphan_reentry` | `pair` orphans among themselves, then `absorb` each remaining orphan into its nearest existing BaryEdge (tev1 classifies orphan vs the BE's two participants) |
| `08_metabary` | form L13 triads (children@L from bridges@L-1), classify each MB's bridge role |
| `09_extend` | relaxed rescue sweep at lower thresholds to absorb leftovers |
| `10_index` | standard indexes + mongot vector index (`tensegrity_vector`, dim = `EMBED_DIM`) |

Checkpoints live in `PIPELINE_STATE_DIR` (`pipeline_state_tensegrity/`); parsed
JSONL lives in `PARSED_DIR` (`data/parsed_tensegrity/`).

## The equations

PoC BaryEdge vector:

```
bary = normalize( q·v_cm1 + q·v_cm2 + (1 − q)·v_type )
```

Tensegrity keeps the PoC's shape but names the relation term a **role** — decided
by tev1 and given a **sign**. `v_role` is the embedding of a fixed per-relation
sentence (embedded once per run). Unlike the PoC's word-oriented sentences,
tensegrity's speak about **sentences**, because sentences are what its BaryEdges
pair; the relation *semantics* are the same kaikki types, so a role vector is
semantically aligned with — though not identical to — the PoC's `v_type`. `q` is
still the embedding cosine of the two participants:

```
any relation except `contradicts`:
    bary = normalize( q·v_cm1 + q·v_cm2 + (1 − q)·v_role )
`contradicts`:
    bary = normalize( q·v_cm1 + q·v_cm2 − (1 − q)·v_role )
```

The relation vocabulary is the PoC's kaikki-derived edge-type set, in fermion
priority order (rarer, more informative first):

| # | role | kaikki field | v_role sentence (`ROLE_SENTENCES`) |
| - | ---- | ------------ | ---------------------------------- |
| 1 | `contradicts` | `antonyms[]` | these two sentences contradict each other: one denies what the other asserts |
| 2 | `applies_to` | `meronyms[]`, `holonyms[]` | one sentence names a part, component or member of the whole named by the other sentence |
| 3 | `is_instance_of` | `hypernyms[]`, `hyponyms[]` | one sentence states a specific instance of the broader kind stated by the other sentence |
| 4 | `extends` | `derived[]`, `related[]` | one sentence is derived from, elaborates or continues the other sentence |
| 5 | `same_phenomenon` | `coordinate_terms[]`, `synonyms[]` | these two sentences describe the same phenomenon, differing only in wording or viewpoint |

There is no `complement` role: `same_phenomenon` plays the affirmative/default
part it used to, and it is the fallback vector for an unclassified edge.

MetaBary keeps the PoC "Born rule" for `q_mb_raw` but applies the role sign to
the bridge term:

```
dir      = normalize( w1·v1 + w2·v2 ± w3·v_bridge )
q_mb_raw = w3² / sqrt(w1⁴ + w2⁴ + w3⁴)
acc_w    = q_mb_raw · level_factor(L)          # level_factor(L) = 1 + α(14−L)/13
```

L14 title vectors are **derived**, mirroring the PoC word-vector stage:

```
v(title) = normalize( Σ v(BE_i) + Σ v(orphan_sentence_j) )
```

## Schema

Nodes (`docs.py`):

```jsonc
// L15 sentence
{ "doc_type": "node", "node_type": "sentence", "level": 15,
  "label": "doc:p0:s0", "vector": BinData(9, …), "parent_edge_id": null,
  "properties": { "sentence_id": "...", "text": "...", "doc_id": "...",
                  "paragraph_idx": 0, "sentence_idx": 0, "section_idx": 0,
                  "char_start": 0, "char_end": 42 } }

// L14 title (vector derived by t05)
{ "doc_type": "node", "node_type": "title", "level": 14,
  "label": "doc:t1", "vector": null | BinData(9, …), "parent_edge_id": null,
  "properties": { "title_id": "...", "text": "...", "doc_id": "...",
                  "section_idx": 1, "title_level": 2, "paragraph_idx": 1 } }
```

BaryEdges / MetaBarys:

```jsonc
{ "doc_type": "baryedge", "cm1_id": ObjectId, "cm2_id": ObjectId,
  "level": 15, "vector": BinData(9, …), "parent_edge_id": null,
  "connection_strength": 0.83,   // == q == embedding cosine
  "accumulated_weight": 0.83,
  "q": 0.83, "type_vector": BinData(9, …),
  "role": "contradicts" | "applies_to" | "is_instance_of" | "extends"
          | "same_phenomenon" | null,           // tev1 argmax
  "role_probability": 0.91,                     // WINNING probability only
  "role_source": "tev1" | "fake" | "unclassified",
  "source": "inferred" | "reentry", "confidence": 0.83 }
```

MetaBarys (`level ≤ 13`) use `metabary()`: same shape minus `q`/`type_vector`,
with `connection_strength = q_mb_raw` and `accumulated_weight = q_mb_raw · level_factor(L)`.

### Document provenance (`doc_bridges`)

A **sibling collection** to `tensegrity` (not a field on the graph docs),
adapted from the PoC's `doi_bridges` (`lib/doi_bridge.py`) with the paper DOI
swapped for the document name (`doc_id` = the t01 filename stem):

```jsonc
// tensegrity.doc_bridges
{ "_id": "2601.03812v1", "node_ids": [ObjectId, …], "created_at": …, "updated_at": … }
```

`_id` is the document name; `node_ids` lists **every node, BaryEdge and
MetaBary whose ancestry includes that document** — so a retrieved node can be
traced back to the paper(s) it came from, and a paper can be expanded to its
whole subtree. Wiring (`tensegrity_pipeline/doc_bridge.py`):

| stage | call | meaning |
|---|---|---|
| t03 | `register_nodes_for_docs` | root: each document ↔ the nodes it produced |
| t04/t06 | `propagate` (in `pairing.form_level_edges`) | new edge inherits both participants' documents |
| t07 | `propagate` (absorb phase) | new (orphan, BaryEdge) edge inherits both sides |
| t08/t09 | `propagate` (in `_form_level`) | MetaBary inherits both children + the bridge |

Read side: `docs_for_node(coll, node_id)` / `docs_for_nodes(coll, node_ids)`
(reverse lookup, multikey index on `node_ids`, ensured by t03 and t10).
`propagate_up_chain` covers the late-arrival case (an existing branch gains a
new document); t03's upsert key embeds `doc_id`, so the current stages never
trigger it, but it is kept for parity with the PoC and future re-ingest/merge.

Entries appear **as the pipeline runs** — t03 creates them, the edge/MetaBary
stages extend them. A graph built before this mechanism existed has no entries
until it is rebuilt (t03 refuses to re-insert nodes over existing BaryEdges
without `--force`, so that means a drop-and-rebuild).

### Role semantics

* `role` is the **argmax** of the tev1 probabilities over the kaikki relation
  set (`ROLES` in `tg_vec.py`). `role_probability` is the **winning probability
  only** (the losing probabilities are not stored).
* The question is a plain 5-way Jev choice with one criterion per relation
  (`ROLE_CRITERIA`), worded about **sentences** because that is what gets
  classified. On a one-example-per-relation probe (tev1:0.8b) the model picked
  the intended relation every time: `contradicts` 0.78, `applies_to` 0.83,
  `is_instance_of` 0.60, `extends` 0.47, `same_phenomenon` 0.84.
  `is_instance_of` and `extends` are the least confident, so check
  `role_probability` on real pairs rather than trusting the label alone; the
  probability is recorded as data and is not used to overrule the model.
* If tev1 is unreachable, `role=None`, `role_probability=0.0`,
  `role_source="unclassified"` — the pipeline records the gap, it never invents
  a relation. The vector then falls back to the `same_phenomenon` role term with
  the attractive (`+`) sign.

### Why there is no `not_related` role

"Not related" is a different axis from "which relation", so it is deliberately
**not** in the vocabulary. Two probes on the live model (one example per class)
argued against widening the argmax:

* a sixth `not_related` option in the same question flipped two of five correct
  reads — a contradiction (0.78 → `not_related` 0.61 vs `contradicts` 0.30) and
  a paraphrase (0.84 → 0.49 vs 0.44) — and a tightened criterion only brought the
  contradiction back to a coin flip;
* a second question in the same call (the Jev payload takes a *dict* of
  questions, so this would still be one round trip) was answered badly — `no`
  for plainly related pairs — **and** perturbed the role readout.

Instead, an absent relation is the **absence of an edge**: a pair whose cosine is
below `settings.q_min_l15` (default 0.72, from the PoC) never enters
`top_k_pairs` / `greedy_unique_match`, so nothing is written. Measured cosines
agree with that split — unrelated pairs 0.30 / 0.34 / 0.46 vs related pairs
0.72–0.91 — and a contradiction sits **high** (0.737: same topic, opposite
meaning), which is why the floor must not be raised blindly: it would start
discarding the contradiction band.

Residual case: a pair above the floor that stands in none of the five relations
keeps a forced label, usually a low-`role_probability` `same_phenomenon`. That is
deliberate — the role sets one sign (only `contradicts` is negative) and one term
in a vector the two participants dominate, so a mildly-off label on a genuinely
similar pair costs less than the measured degradation of real relations.

Caveats: these are hand-made pairs, not a corpus benchmark, and the floor is
inherited from kaikki *sense* pairs (`Q_MIN_L15`) yet applied at every level —
including the *derived* L14 title vectors. Re-calibrating it per level on a real
corpus is the open follow-up, independent of this decision.

## Title detection (t01)

Programmatic, no model call (`parse.py`):

* markdown ATX headings (`# …`, `## …`) and setext headings (`===` / `---`);
* numbered headings (`3.2 Methods`, `IV. Discussion`);
* ALL-CAPS short lines (< 100 chars, ≥ 80 % uppercase letters, no terminal
  punctuation);
* Title-Case short lines (no terminal punctuation, ≥ 60 % capitalised words).

Sentence splitting is a deterministic scanner with an abbreviation guard:
common English (`Dr.`, `e.g.`), initials/acronyms (`U.S.`, `J.`), and the
legal/citation domain measured on court opinions (`See id.`, `4th Cir.`,
no-space compounds such as `F.Supp.`, Roman section numerals `II.`).
Ellipsis runs (`. . .`, `...`) never end a sentence, and segments with no
letters at all (star-page `*173`, footnote numbers, bare punctuation) are
dropped instead of stored as sentences. `_TERMINAL_PUNCT` is a **tuple** on
purpose: passed as a plain string, `str.endswith` matches only its literal
suffix and the guard never fires — that bug is what let 2,820
punctuation-terminated lines through as titles. `section_idx` on each
sentence records which title it belongs to — membership is **metadata**, not
pairwise edges.

## tev1 decision model

Role classification goes through `POST /v1/systemone` on the ollama host
(`TEV1_URL`, model `TEV1_MODEL`), the same Jev TypeSafe wire format
`cognitive/not_sure.py` uses. `TEV1_URL` may point at the Kaggle GPU tunnel
(the job API proxies `/v1/systemone` and `/api/*` — see
`docs/kaggle_gpu_runner.md`); the proxy is bearer-gated, so export the printed
`KAGGLE_API_TOKEN` and `Tev1Client` attaches it automatically. A remote run
wants more transport patience than the local default — `TEV1_RETRIES` (default
2, ≈6 s of backoff) buys ~2 min at 6. Measured on the T4 tunnel: 0.55 s/call
warm vs 10.92 s local for the same sentence states (≈20×). Each call is short
and latency-dominated, so the GPU idles between them: `TEV1_WORKERS` (default
1) fans the per-pair decisions out to a thread pool while preserving pair
order — 8 workers measured **4.7×** (0.55 → 0.12 s/pair) on the tunnel. The
same helper (`classify_states`) drives t07's orphan-absorb loop, where the
decision count is one per orphan (~11k for the 100-opinion corpus).

Note: `--dry-run` skips the DB writes but **still asks tev1 for every pair /
orphan**, so a preview costs full GPU time. Set `BARY_FAKE_DECISION=1` as well
to preview pair counts for free.
`tev1.py` exposes:

* `Tev1Client` — transport client with retries;
* `FakeDecision` — deterministic offline stand-in;
* `classify_role(model, state)` — never raises; returns the role contract above.

## Configuration (`.env.tensegrity`)

```ini
MONGO_DB=tensegrity
MONGO_COLLECTION=tensegrity
MONGO_DOC_BRIDGES_COLLECTION=doc_bridges   # document-name -> node provenance
EMBED_MODEL=qwen3-embedding:0.6b
EMBED_DIM=1024
PARSED_DIR=data/parsed_tensegrity
PIPELINE_STATE_DIR=pipeline_state_tensegrity
Q_MIN_L15=0.72
META_BARY_COS_THRESHOLD=0.90
TEV1_URL=http://localhost:11435
TEV1_MODEL=tev1:0.8b
BARY_FAKE_EMBED=0
BARY_FAKE_DECISION=0
```

## Tests

* Unit: `tests/unit/test_tensegrity_{parse,tg_vec,tev1,docs}.py`
* Integration: `tests/integration/test_tensegrity_pipeline.py` — runs t01→t10
  end-to-end against an isolated `barygraph_test_*` DB with fakes enabled
  (requires a project Mongo; `make up`); asserts the `doc_bridges` provenance
  index covers every graph document.
* Integration: `tests/integration/test_tensegrity_doc_bridge.py` — the
  provenance module itself (register/propagate/up-chain/batched lookups).

## Design notes / differences from the PoC

* **No `lib/` edits** — the pipeline reuses `lib/{config,checkpoint,embed,vector,db,match,bary_vec}`
  unchanged and adds its own `pairing.py` / `tg_vec.py` / `tev1.py`.
* **In-memory loaders** in t08/t09 instead of s08's disk memmaps: document
  corpora are orders of magnitude smaller than kaikki's 12M-edge L15 pool.
  If this ever needs to scale to millions of edges, port `_load_unparented_bes`
  to s08's memmap strategy.
* **`source="structural"` SMBs are never consumed** by t08/t09 — the same rule
  the PoC uses so hand-admitted triads can be folded in on a later rerun.
* Roles are decided at edge-formation time (t04/t06) and MB-formation time
  (t08/t09). There is intentionally **no re-classification pass**: re-running a
  stage with `--reset` rebuilds both the edge and its role together.
