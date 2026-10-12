# Tensegrity pipeline

A parallel ingestion pipeline to the Kaikki PoC (`scripts/s01..s10`). Where the
PoC ingests **dictionary entries** (L14 words / L15 senses), this pipeline
ingests **documents** and builds a BaryGraph-shaped graph out of them:

| Level | Meaning | Source |
| ----- | ------- | ------ |
| **L15** | document **sentences** | split from the input text; embedded (qwen0.6, 1024-d) |
| **L14** | section / paragraph **titles** | detected heuristically; **no native embedding** — vectors derived in t05 |
| **L13…L1** | **MetaBary** triads | built recursively from role-tagged BaryEdges |

Every BaryEdge carries a **role** — `complement` (attraction / "compression") or
`contradict` (repulsion / "tension") — decided by the **tev1** decision model,
and a modified barycentric vector whose *sign* encodes that role. That signed
role is what makes this a *tensegrity* structure rather than a plain similarity
graph.

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
Stage-specific: `--doc PATH` (t01), `--phase pair|absorb|both` (t07),
`--max-level N` (t08/t09).

Offline / CI: set `BARY_FAKE_EMBED=1` (deterministic vectors) and
`BARY_FAKE_DECISION=1` (deterministic tev1 roles). No ollama needed.

## Stages

| Stage | Does |
| ----- | ---- |
| `01_parse` | split the input document into L15 sentences + L14 titles (`parse.py`) |
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

Tensegrity replaces the edge-type term with a **signed role term**. `v_role` is
the embedding of a fixed per-role sentence (embedded once per run), and `q` is
still the embedding cosine of the two participants:

```
complement:  bary = normalize( q·v_cm1 + q·v_cm2 + (1 − q)·v_role )
contradict:  bary = normalize( q·v_cm1 + q·v_cm2 − (1 − q)·v_role )
```

`ROLE_SENTENCES` (`tg_vec.py`):

```
complement: "these two statements are complementary: they are consistent with
             each other and reinforce the same meaning"
contradict: "these two statements are contradictory: they are mutually
             exclusive and cannot both be true"
```

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
  "role": "complement" | "contradict" | null,   // tev1 argmax
  "role_probability": 0.91,                     // WINNING probability only
  "role_source": "tev1" | "fake" | "unclassified",
  "source": "inferred" | "reentry", "confidence": 0.83 }
```

MetaBarys (`level ≤ 13`) use `metabary()`: same shape minus `q`/`type_vector`,
with `connection_strength = q_mb_raw` and `accumulated_weight = q_mb_raw · level_factor(L)`.

### Role semantics

* `role` is the **argmax** of the tev1 probabilities. `role_probability` is the
  **winning probability only** (the losing probability is not stored).
* If tev1 is unreachable, `role=None`, `role_probability=0.0`,
  `role_source="unclassified"` — the pipeline records the gap, it never invents
  a stance. The vector then falls back to the attractive (`+`) sign.

## Title detection (t01)

Programmatic, no model call (`parse.py`):

* markdown ATX headings (`# …`, `## …`) and setext headings (`===` / `---`);
* numbered headings (`3.2 Methods`, `IV. Discussion`);
* ALL-CAPS short lines (< 100 chars, ≥ 80 % uppercase letters, no terminal
  punctuation);
* Title-Case short lines (no terminal punctuation, ≥ 60 % capitalised words).

Sentence splitting is a deterministic scanner with a common-abbreviation guard
(`Dr.`, `e.g.`, `U.S.`, initials). `section_idx` on each sentence records which
title it belongs to — membership is **metadata**, not pairwise edges.

## tev1 decision model

Role classification goes through `POST /v1/systemone` on the ollama host
(`TEV1_URL`, model `TEV1_MODEL`), the same Jev TypeSafe wire format
`cognitive/not_sure.py` uses. `tev1.py` exposes:

* `Tev1Client` — transport client with retries;
* `FakeDecision` — deterministic offline stand-in;
* `classify_role(model, state)` — never raises; returns the role contract above.

## Configuration (`.env.tensegrity`)

```ini
MONGO_DB=tensegrity
MONGO_COLLECTION=tensegrity
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
  (requires a project Mongo; `make up`).

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
