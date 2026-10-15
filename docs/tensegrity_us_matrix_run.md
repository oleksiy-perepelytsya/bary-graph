# Tensegrity ingest — US matrix gold corpus (100 opinions)

First full end-to-end ingest of the 100-opinion corpus in
`tensegrity_pipeline/us_matrix_gold_100.zip` into the `tensegrity` database
(collection `tensegrity`). All ten stages completed; every stage checkpoint in
`pipeline_state_tensegrity/` is `done=true`.

Corpus: 100 opinion texts extracted to `data/us_matrix/cases/` (untracked),
`doc_id` = filename stem (e.g. `1038917`).

## Run environment

Embeddings and every tev1 decision ran on the Kaggle GPU through the
cloudflared tunnel (`docs/kaggle_gpu_runner.md`), not locally.

```bash
KAGGLE_API_TOKEN=<from notebook>            # bearer gate on the tunnel
TEV1_URL=https://<tunnel>.trycloudflare.com # tev1:0.8b on the T4
TEV1_RETRIES=6                              # remote patience
TEV1_WORKERS=8                              # concurrent decisions, order-preserving
ANN_THRESHOLD=2000                          # HNSW instead of brute force (see README)
MONGO_URI='mongodb://localhost:27117/?directConnection=true'
EMBED_BATCH_SIZE=128                        # tunnelled embed batch
```

Measured: 0.55 s/tev1 call warm serial, 0.12 s at 8 workers (4.7x); embedding
22,883 sentences took 9m49s (~38/s) versus ~1.06/s on this 4-core box.

## Stage results

| Stage | Output | Wall time |
| ----- | ------ | --------- |
| t01 parse | 22,883 L15 sentences + 2,390 L14 titles | — |
| t02 embed | 22,883 sentence vectors (1024-d, 0 missed) | 9m49s (GPU) |
| t03 insert nodes | 25,273 node docs | seconds |
| t04 L15 edges | 5,973 BaryEdges (5,973 tev1 calls) | 7.5 min (workers=8) |
| t05 title vectors | 2,008 L14 vectors (382 dead titles) | ~2 s |
| t06 L14 edges | 748 BaryEdges | ~1 min |
| t07 orphan re-entry | 205 re-pairs + 10,527 + 512 absorbed | 18.7 min |
| t08 metabary | 1,253 L13 MetaBarys | 2.5 min |
| t09 extend | 7 L13 MetaBarys (rescue sweep 0.90→0.76) | 16 s |
| t10 index | 9 standard indexes + `tensegrity_vector` (mongot) | <1 s |

tev1 decisions total: **19,225** (5,973 + 748 + 205 + 10,527 + 512 + 1,253 + 7).

## Final graph

```
total docs 44,498
  L15: 22,883 nodes (sentences)  16,705 BaryEdges
  L14:  2,390 nodes (titles)      1,260 BaryEdges
  L13:      0 nodes               1,260 MetaBarys
doc_bridges: 100 docs, 56,120 provenance refs
```

Every doc is role-tagged by tev1 (`role_source='tev1'`, zero `unclassified`):

| Level | same_phenomenon | extends | is_instance_of | applies_to | contradicts |
| ----- | --------------- | ------- | -------------- | ---------- | ----------- |
| L15 (16,705) | 46.3% | 27.7% | 16.9% | 6.2% | 2.9% |
| L14 (1,260) | 16.3% | 14.6% | 57.9% | 10.6% | 0.6% |
| L13 (1,260) | 71.0% | 2.5% | 4.0% | 22.1% | 0.5% |

Attachment:

* L15 sentences: **22,883 / 22,883 parented (100%)**.
* L14 titles: 2,008 / 2,390 (84%) — the 382 unparented titles are "dead"
  (no child sentences, so no derived vector) and can never pair.
* L15 BaryEdges used above: only 2,520 / 16,705 (15.1%).

## Structural finding: the L13 layer is bridge-capped

A MetaBary consumes one lower-level BaryEdge as its **bridge**, and each BE can
be a bridge at most once. So L13 is hard-capped by the L14 pool:

* t08 formed 1,253 of a possible 1,260 triads and dropped 3,196 matched pairs
  with "no unparented bridge in window".
* t09's rescue sweep added the remaining **7** — bringing L13 to exactly
  **1,260 = the number of L14 BaryEdges**.

Consequence: for document corpora the title level is ~10x sparser than the
sentence level, so 85% of L15 edges never reach the L13 structure. The graph
tops out at three levels (L15 → L14 → L13) and does not recurse deeper with the
default `--max-level 13`. This is a property of the corpus/pipeline shape, not
a failure; it is the main thing to consider before using this graph as a
cognitive-memory substrate.

Role quality: `role_probability` on L15 edges has median 0.455 (many near-ties),
and local vs GPU numerics flip some argmaxes — role distributions are
run-dependent at the margin.

## Verification

* `$vectorSearch` on `tensegrity_vector` returns neighbours correctly (index
  `READY`, `numDimensions=1024`); note it needs a moment in `INITIAL_SYNC`
  after creation before it is queryable.
* DB `created_at`/`updated_at` are UTC; stage log lines print local time
  (UTC+2 in this environment).
* Gold archive (`gold.jsonl`, `gold_matrices.csv`, `score.py`) validated
  100/100 with 0 FAILED, and its texts are sha256-identical to the input
  corpus — it stays available as an independent benchmark for this graph.
