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

## Benchmark: three extraction arms

Harness: `scripts/eval/us_matrix_common.py` +
`scripts/eval/run_eval_us_matrix.py`; scored with the archive's own `score.py`.

| arm | model? | what it sees |
| --- | --- | --- |
| `graph_regex` | no | `cases.csv` + the graph's L15 sentence index + regex harvesters |
| `llm_fulltext` | yes | `TASK.md` + `cases.csv` row + whole opinion text |
| `llm_graph` | yes | the same, plus a graph overlay (L14 outline, intra-opinion role-tagged relations, cross-opinion parallels); quotes snapped to graph sentences |

All arms get `cases.csv`, because the task hands it to testers.

### Arm A (`graph_regex`, all 100, ~3 s)

| metric | value |
| --- | --- |
| missing_fields | 0.000 |
| court_level | 0.980 |
| jurisdiction | 0.930 |
| date_decided | 0.950 |
| procedure_type / area_of_law / standard_of_review / disposition / winning_party | 0.000 |
| quotes_verbatim | 1.000 |
| cit_recall / cit_precision | 0.400 / 0.409 |
| stat_recall / stat_precision | 0.434 / 0.591 |

`quotes_verbatim = 1.0` is structural: the graph's L15 nodes *are* the opinion's
sentences, so evidence picked from the graph is verbatim by construction.

### Two findings that matter for interpreting any score

1. **`cases.csv` answers 3 of the 8 EXACT fields.** It ships with the task and
   carries `court_level`, `court`, `case_name`, `date_filed`, `docket_number`.
   Its `court_level` uses CourtListener's vocabulary (`fed_appellate`), not the
   template's (`federal_appellate`): copying it verbatim scores 43/100, but a
   one-line mapping scores **98/100**. With `date_filed` and a state-list
   derivation, `court_level`/`date_decided`/`jurisdiction` reach 0.98/0.95/0.93
   with no model at all. The discriminating fields are the five semantic ones
   (`procedure_type`, `area_of_law`, `standard_of_review`, `disposition`,
   `winning_party`).
2. **The graph is a lossy index of the corpus.** Comparing non-whitespace
   characters of the graph's L15 text against the source files: **97.59%
   corpus-wide**, with **35 of 100 documents below 98%** (worst 89%). Case
   `26133` lost 26 of its 28 reporter citations — verified absent from the
   database, not merely unmatched. Arm A inherits this; the LLM arms see the
   full file text instead, so the two are not reading quite the same document.

### GPU-side execution

The LLM arms never ran through the tunnel. A Cloudflare-fronted request to
`qwen3:14b` returned HTTP 524 at ~126 s on every real payload: prompt eval is
fast (555 tok/s), the model stays in VRAM (13.03 GB, ctx 24576) and eval runs at
~13 tok/s, but ~110 s of wall time was unaccounted for and arrived ~2.75 s per
streamed chunk (`scripts/kaggle_embed_server.py` forwards with `resp.read(1<<20)`,
i.e. up to 1 MB blocking reads, and strips `transfer-encoding`). Small prompts
passed; real ones did not. The arms are therefore packaged as a self-contained
bundle (cases + precomputed overlays + harness, no Mongo, no gold) and run
against `http://localhost:11434` on the GPU host — see the bundle's
`README_KAGGLE.md`.

Harness details worth keeping: `sentence_id` is a string (`26133:p100:s0`), so
document order requires parsing the integers out — sorting the ids directly
scrambles the text; and the overlay is trimmed, never the opinion, when a prompt
would exceed the context budget.

## Experiment (b): does graph *structure* improve semantic extraction?

The original `llm_graph` arm handed the model the opinion text **plus** an
overlay that, measured over all 100 documents, was mostly noise: 17% of
intra-opinion relation lines were exact self-duplicates, 20% had a side under 20
characters, 79% of cross-opinion "parallel passages" were under 20 characters
and 45% were empty, and the role labels were dominated by `same_phenomenon`
(`contradicts` was only ~3% of edges). So `llm_graph` was close to
`llm_fulltext` + a noisy appendix, and did not cleanly test the graph.

**Design.** `llm_graph_struct` (§ `build_overlay_struct`, commit `a5da782`)
keeps only the signals the raw text does not state: `section_idx` groups,
`contradicts` pairs, **de-duplicated** role-tagged sentence pairs, and
substantive non-paraphrase cross-opinion ties. A 30-case stratified pilot
(15 high-contradiction + 15 spread; `kaggle_bundle/cases_pilot30.txt`) ran both
arms with `qwen3:14b`, one ollama per GPU and the arm sharded across them.

**Result** (paired, n=30; metric deltas = `llm_graph_struct − llm_fulltext`):

| metric | fulltext | struct | delta |
| --- | --- | --- | --- |
| semantic mean (5 fields) | 0.713 | 0.740 | **+0.027** |
| `standard_of_review` | 0.533 | 0.667 | +0.133 |
| `procedure_type` | 0.667 | 0.700 | +0.033 |
| `area_of_law` | 0.833 | 0.800 | −0.033 |
| `disposition` / `winning_party` | 0.767 | 0.767 | 0.000 |
| metadata mean | 0.911 | 0.922 | +0.011 |
| citation / statute P-R | ~0.24–0.56 | ~0.24–0.56 | ±0.03 (noise) |
| **`quotes_verbatim`** | 0.573 | **0.967** | **+0.394** |

Per-case semantic mean: G>T 6, G<T 3, tie 21; **paired t-test p=0.255**. Per-field
paired flips are all non-significant (McNemar: `standard_of_review` 4–0,
p=0.125; every other field ≤1 flip).

**Conclusion.** The cleaned graph *structure* gives **no statistically
significant** semantic gain (+0.027, p≈0.26; even the one directional signal,
`standard_of_review`, is p=0.125 at n=30). The graph's one **large, real**
contribution is the verbatim-quote guarantee (0.573 → 0.967) — but that comes
from the L15 **sentence store** (quote snapping), not from the relation/role
structure. Metadata fields are unchanged (both arms read `cases.csv`), and
citations/statutes are unchanged.

This vindicates the original concern: in this design the LLM does the semantic
work from the opinion text, and the graph's *structure* contributes nothing
measurable. Caveats: n=30 on 5 binary fields is underpowered (ruling out only
large effects, ≳15–20%); a single model (`qwen3:14b`); the `quotes_verbatim`
delta conflates the sentence store with snap-on for both graph arms.

**Operational note.** `/no_think` (qwen3) cut per-case time ~4× (10.6 → ~2.5
min; ~14 tok/s, ~2,000 out tokens), and sharding one ollama per GPU (2×T4)
roughly doubled throughput. The 60-case pilot finished in ~80 min.
