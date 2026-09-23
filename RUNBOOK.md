# RUNBOOK — barygraph_all build operations

Operational runbook for the `barygraph_all` build. Config/env specifics live in
`AGENTS.md` ("Databases and .env files"); stage timing ledger lives in
`notes_on_stages.md`. This file holds run-specific launch plans and the
process/state map so nothing is lost on session compaction.

## Environment

- **barygraph_all** build env: source `.env.build-all` before any probe or launch
  (default `.env` targets `barygraph_poc` — mismatch caused a long rabbit hole).
- Pipeline state: `pipeline_state_all/*.json`; launch stages via
  `python3.11 -m scripts.<stage>` and keep logs under `/tmp/opencode/`.
- Stage order (guard in `scripts/_base.py`, refuses out-of-order runs):
  s01_parse → s02_embed → s03_insert_nodes → s04_l15_edges → s05_word_vectors →
  s06_l14_edges → s07_orphan_reentry → s07b_pair_orphans → s08_metabary →
  s09_extend → s10b_projection (deferred) → s10_index.

## Process map (as of 2026-09-16)

| PID | process | role |
|---|---|---|
| 87969 | `s05_word_vectors` | full s05 run on barygraph_all (log `/tmp/opencode/s05.log`) |
| 119884 | `cog_mb_loop --minutes 1440` | 24h cognitive loop, model `ollama/qwen3.6:27b-q4_K_M-ctx64k`, poll-timeout 600 (restarted to add author canonicalization + per-cycle session cleanup) |
| 64552 | `mcp_server --port 8000` | barygraph MCP (poc), keep-warm; behind cloudflared tunnel |
| 78171 | `mcp_int_cog` | local cog MCP (poc) |
| 3242 | cloudflared tunnel | https://mild-infrastructure-should-batch.trycloudflare.com/mcp — do not restart casually |
| 77601 | ollama serve | host ollama at `http://ollama:11434` |
| 77747 | llama-server :36783 | keep-warm llama.cpp instance (blob ≠ deepseek; left alone) |

## S06 LAUNCH PLAN (preserved 2026-09-16 — user will trigger manually)

Execution trigger: **user says "go"** (they will be present and will ask). Do
NOT auto-launch on s05 completion without being asked.

- **Trigger condition:** s05 checkpoint `done=true`
  (`pipeline_state_all/05_word_vectors.json`), expected ~03:10–05:50 UTC Sep 17
  (~10.9–13.5h from 16:23 UTC Sep 16; recent pace ~177 w/s, band 148–177).
- **Pre-flight memory inventory (required):**
  ```
  free -g
  ps aux --sort=-%mem | head -12
  ```
  Expect: s05's python has exited (frees its own GBs), cog-loop children
  ~1–2 GB each, ollama 27B lives in VRAM (not RAM), keep-warm/cloudflared are
  tens of MB. Need **≥ ~200 GB free RAM** for s06's 168 GB `V` matrix + dicts
  margin. If a surprise consumer is present, identify and resolve it first.
- **Launch s06 as-is** (no edits, no `--limit`, no `--force`), mirroring s05:
  ```
  set -a; source .env.build-all; set +a
  setsid nohup python3.11 -m scripts.s06_l14_edges > /tmp/opencode/s06.log 2>&1 < /dev/null &
  ```
- **Watch the load phase (first ~15–25 min):** this is when the 168 GB `V`
  allocation happens. Re-check `free -g` at ~1 min and ~10 min. Compute phase
  after is tiny (6 tiers × 10M dict-lookups, ~10–25 min CPU).
- **Swap floor:** host has only ~7 GB swap. If `free` shows swap growing /
  thrashing during the load, kill the stage and recover:
  ```
  pkill -f scripts.s06_l14_edges
  # then, once user is ready:
  set -a; source .env.build-all; set +a
  setsid nohup python3.11 -m scripts.s06_l14_edges --force > /tmp/opencode/s06.log 2>&1 < /dev/null &
  ```
  `--force` pre-loads already-paired words (skips them) and re-derives the
  rest — no corruption, just recompute cost. s06 checkpoints only at the end,
  so a mid-run crash / OOM costs most of the compute.
- **Keep the 24h cog loop running** during s06 — its RAM footprint is small
  and does not materially affect s06; do not pause it for s06.
- **s06 memory math (why the pre-flight matters):** `V` = 10,256,975 words ×
  4096-dim × 4 B ≈ **168 GB** float32, plus ids/words/dicts (~20–40 GB). Box:
  251 GB total, ~229 GB available at rest. Wall time expected ~1.5–2.5 h (load
  ~20 min, compute ~15 min, Mongo writes ~50–100 min for est. 4–7M L14 edges,
  each edge doc ~17–21 KB carrying the 4096-dim type_vector).
- **When done:** cp `done=true`, `cp.processed = n_edges`; proceed to s07.

## Log/state map

- `/tmp/opencode/s05.log` — s05 live log (PID 87969)
- `/tmp/opencode/s06.log` — s06 (to be created at launch)
- `/tmp/opencode/cog_mb_run24h_qwen36.log` — cognitive loop driver log
- `/tmp/opencode/cog_mb_cycleN.log` — per-cycle agent logs
- `/workspace/bary-vector/cognitive/batches/smb_candidates.jsonl` — SMB candidates (loop output)
- `/workspace/bary-vector/pipeline_state_all/` — all-build stage checkpoints

## Known watch items

- poc `$vectorSearch` degraded to ~25 s and `count_documents` can hang while
  the all-build writes — `context_search` (multi-stage) can exceed MCP
  timeouts; the cog-loop prompt bans it and caps vector-seeking calls.
- Keep-warm llama-server blob `3fcd3feb…` is unrelated to deepseek; do not
  delete models whose blobs it references (deepseek `6150cb38…` already removed
  safely).

# AUTHOR-LAYER EXPERIMENT — STRUCTURAL PROJECTION (design, folded 2026-09-17)

Status: **STORED, not scheduled.** Manual trigger (user's word). Do not start
automatically after s05/s06.

**Hypothesis.** Encoding a person's works as structural SMBs on top of the
dictionary graph yields a measurable associative signature ("way of thinking"
as structure). Feynman = the controllable test case; the same construction is
the intended mechanism for the user's own persona-projection layer (AGENTS.md
goal). It is a *projection/atlas*, not a recreation of a person.

**Prototype evidence already in poc.** The caterpillar/half-track SMB
`6aaaec6b516170692834ab15` encodes the user's own idiosyncratic bridge —
*erucism : skin :: half-track : tundra* (thin crust, extremely slow regrowth).
A testimony-association the dictionary alone never produces; confirms SMB form
carries individual associative data beyond the lexicon.

**enrich_request grounding already gathered (poc, 2026-09-16):**
- `path integral feynman` mean_cos 0.958 → L13 `path integral formalism ↔
  Faddeev–Popov ghost` bridge `quantum gravity`
- `diagram quantum electrodynamics` → L13 `Feynman diagram / spin network ↔
  interaction picture` bridge `diagram, drawing`
- `explanation analogy wonder` 0.929 → `analog/analogy` cluster; "wonder"
  retrieves NO coordinate → affect/tone registers are invisible to the graph
  (documented limit).

**Method (per author, works as timestamped slivers).**
1. Corpus: 3–4 short iconic Feynman texts ("The Value of Science" 1955,
   Messenger Lecture segments) + a control author's analogous essays.
2. Extract terms (existing `cog_store_paper_extraction` path) → ground each via
   `enrich_request` (mb_only, with_bridges, compactness) → structural SMBs via
   `_create_structure_meta_bary_body` (source='structural', manifest
   `smb_builds.jsonl`).
3. Grade via QC gates (distinctness cos(c1,c2)<0.92; mediation ≥0.50;
   coherence soft-flag <0.70; novelty vs the 6+ built SMBs).
4. Signature metrics: bridge-repertoire centroid, cross-region bridge density
   (abstract↔concrete/analogy clusters), edge-type mix; Feynman vs control →
   bridge-distance.
5. Extensions: diachronic slivers by decade → structural biography; same
   machinery on the user's own texts/notes → personal projection layer.

**Boundaries.** Structure ≠ process (thinking = the multi-probe walk +
bridge-collapse, the relational-compression axis). Lexicon-capped, affect-blind,
no negation/irony/un-said. Output = atlas of couplings, coordinates for
investigation only.

**Sibling interest (user's, drives a parallel exploration):** what the *models*
themselves produce — Qwen3.6's SMB bridges are also a projection (trained on
lexicons whose siblings fed its corpus). Map Qwen3.6's semantic landscape from
the candidate pool (see `cognitive/batches/smb_candidates.jsonl`,
~173 qwen3.6 records as of 2026-09-17).

**Artifacts when triggered:** `cognitive/author_feynman/` corpus+prompts,
`cog_author.py` driver, manifest + QC sheets appended to shared batches.

# ENRICH-REQUEST RELATIONAL COMPRESSION — FULL IMPLEMENTATION PLAN

Status: **CONFIRMED by user, NOT yet implemented / stashed for implementation.**
Preserved verbatim + design decisions so work can resume post-compaction without
re-deriving anything. Fold in with any bridge-estimation work before restarting
the MCP server. (Saved 16:5x UTC Sep-16 2026 — user resting; commit captures this.)

## Goal

`barygraph_enrich_request` payloads blow up on dense neighborhoods. New
`relational_compression: bool = False` mode collapses redundant vectors into
int8 cluster centroids while preserving rare/triad-bearing signals exactly.
Default path (`relational_compression=False`) must stay **byte-identical** to
today — safety guarantee for reverse-mode verification.

## Reference code (scripts/mcp_server.py)

- `enrich_request` tool — line 885; `_run_thr` call at 976-983 (passes
  query,top_k,max_probes,mb_only,with_bridges,precision,compactness)
- `_enrich_probe` — line 1067 (thread-safe, run via ThreadPoolExecutor at
  1156-1160; add param to the `ex.submit` call)
- `_enrich_request_body` — line 1127 (validate/clamp unchanged; thread flag
  into `_enrich_probe`)
- Helpers available: `_encode_vector` (1019), `_compactness` (1036), `_bridge_hit`
  (990), `unpack_vec` (lib.vector), `cosine`/`normalize as _norm_vec`
  (lib.bary_vec), `vector_search` (lib.db). Docs carry `cm1_id`/`cm2_id`/`bridge_id`
  (MBs) — confirmed in lib/docs.py metabary() lines 119-132.

## New helpers (in scripts/mcp_server.py)

### 1. `_cluster_vectors(vectors: np.ndarray, threshold: float = 0.85) -> tuple[list[list[int]], np.ndarray]`
- Greedy agglomerative clustering on cosine (reuses pair-cosine math of
  `_compactness`). threshold 0.85 → "same region". Returns member index groups
  + normalized-mean centroid per group. n ≤ 20 → trivially fast.
- Algorithm: start n singleton groups; repeat: stack centroid matrix, take
  upper-triangle pair cosines, highest ≥ threshold → merge those two groups
  (replace at lower index, pop higher); recompute centroids from groups each
  pass. Break when no pair ≥ threshold.
- Design decision (resolved): merge criterion is **centroid-to-centroid** cosine;
  single-pass greedy, not full agglomerative dendrogram.

### 2. `_protection_score(hit, cluster_size, total_hits, cos_to_centroid) -> dict`
- `rarity = 1 - cluster_size/total_hits` (singleton in a 10-hit probe ≈ 0.9)
- `distinctiveness = 1 - cos_to_centroid`
- `bridge`: level-scaled → L10=1.0, L11=0.8333, L12=0.6667, L13=0.5,
  L14/L15 BE (and unknown)=0.3; **+0.2 bonus only if level ≤ 13 AND cm1_id &
  cm2_id present** (triad-bearing MB).
- `protection = 0.35·rarity + 0.30·distinctiveness + 0.35·bridge`
- Returns {protection, rarity, distinctiveness, bridge} (rounded 4).
- Design decisions (resolved — plan numbers don't reach 0.60 for in-cluster
  triad MBs):
  - `cos_to_centroid` for a **singleton group** = cosine to the **nearest OTHER
    group's centroid** (max over others; 1.0 if none) — matches "isolated
    vector → high distinctiveness".
  - Triad-bearing MB (level ≤ 13, cm1_id & cm2_id present) is **force-preserved**
    regardless of score — "the explicit bridge is the point" (tool philosophy).
    This makes plan-test-3 pass.

### 3. `_encode_vector` — unchanged; reused for preserved nodes.
Preserved entries encode at **caller precision**, **promoted to "full"** when
`protection ≥ 0.90`. Cluster centroids **always int8** (+`centroid_scale`).

## `_enrich_probe` changes

- New param `relational_compression: bool`.
- Keep the existing doc loop unchanged (unpack → encode → hits), additionally
  collect `raw_vectors` (for compactness) and a `raw_meta` list
  `{raw, id, level, edge_type, cm1_id, cm2_id}` when the flag is on.
- After loop: if `relational_compression` → build block via new
  `_compressed_block(chunk, metas, precision, hits=None, raw_vectors=None)`:
  - cluster all raw vectors
  - per member compute protection (+ triad force-preserve)
  - **preserved** (protection ≥ 0.60 OR triad-MB): `{id, level, edge_type,
    vector, vector_scale?(int8 only), protection, rarity, distinctiveness,
    bridge}`, in rank order
  - **clusters** (non-preserved members, grouped): `{n, centroid(int8),
    centroid_scale, mean_cos, max_cos (member→centroid), residual_err =
    1-mean_cos, members: [{id, level, edge_type}]}`, ordered by first member rank
  - block = `{"chunk", "compression": {total_hits, preserved, clusters,
    vector_precision (=caller precision, "full" if any preserved promoted)}}`
  - `with_bridges` → block["hits"] unchanged; `compactness` → block["compactness"]
    unchanged (both can coexist)
- Flag off → exactly today's output shape.

## Tool API

- `enrich_request(..., relational_compression: bool = False)` on the enriched
  signature list; docstring section added. Confirmed: **new bool, not a
  precision value**.
- `_enrich_request_body` threads the flag; validation/clamps otherwise unchanged.

## Tests — tests/unit/test_enrich_relational.py (offline, no Mongo)

1. Dense 3-vector cluster (cos > 0.95) → one cluster, n=3, members listed, low
   protection → compressed (each member assimilated, none preserved).
2. Singleton isolated vector (as a member of a larger probe set) → preserved
   exact, high rarity + distinctiveness.
3. Triad-bearing MB (cm1_id+cm2_id, level ≤ 13) inside a dense cluster → still
   preserved despite being in a cluster (force-preserve rule).
4. int8 centroid reversibly dequantizes within expected error (< 1/127 scale;
   symmetric rounding ⇒ ≤ scale/254 actually).
5. Default path: `relational_compression=False` output == current format
   (guards byte-identical guarantee at the `_enrich_probe` block level).

## Verification

- `pytest` — expect the existing 108 passing + new file green (known
  `test_sense_node_schema` failure unrelated).
- Live smoke (do NOT do this while user is resting / without coordination —
  needs `mcp_server_ctl.sh restart` which also restarts the remote MCP that the
  24h cognitive loop lives on; a bad mid-cycle restart drops at most one cycle):
  `enrich_request(probe e.g. "autapse dendritic", relational_compression=true,
  with_bridges=true, compactness=true, precision="int8")` and compare payload
  size vs `compactness=true` baseline via the tunnel.
- Rebase onto whatever is current; fold in with any bridge-estimation work
  before the single restart.

## Stash note

The full plan + design decisions live HERE, not just in conversation. Resume
from this section.

# S08 WRITE-PATH PATCH — PLAN (deferred, implement "a bit later")

Status: **APPROVED for drafting, NOT yet implemented.** Decided 2026-09-18:
let s07b roll as-is (A+B load parallelization rejected — 2–3h gain not worth
new-bug risk). These s08 changes are queued; apply them as part of s08
launch-readiness (before the full s08 run, i.e. after s07b band-exhaust +
s07 sweep). s07 needs NO port — it already carries s07b's write tuning.

## What to change in `scripts/s08_metabary.py`

1. **Unacknowledged MB parent stamps** (the real win — same rationale as
   s07b's `_flush`):
   - Currently `:347` does `coll.bulk_write(ups, ordered=False)` acknowledged.
   - Switch to `coll.with_options(write_concern=WriteConcern(w=0))`, default ON,
     env switch `S08_STAMPS_UNACK=0` restores acked (mirror s07b).
   - MB `insert_many` (`:340`) STAYS acknowledged — the MB docs are the durable
     payload; stamps are idempotent bookkeeping.
   - Safety: s08 does **3 stamps per triad** (2 children + bridge), so the
     s07b-measured gain (~7s→0.2s per batch) scales ~3× here. Lost stamps are
     harmless — a BE left unparented sits at a level no later pass consumes
     (bridges selected at exact level, children only at that pass's child
     level), so it cannot be re-consumed or double-parented. The `:379`
     level≤13 guard still blocks bad re-runs.
2. **Write `BATCH` 1000→2900** (`:324`) for all level passes — cuts
   insert round-trips ~2.9×; keep the per-batch
   `posix_fadvise(POSIX_FADV_DONTNEED)` (`:350-353`) so write-phase RSS stays
   bounded (same as s07b's batch 2900).
3. **Per-batch write progress log** — s08 logs nothing during the long L13
   write; add a `log.info("  written %d triads (L%d)")` per batch like s07b's
   "inserted %d BEs this round" so the multi-hour write can be monitored.

## Launch-readiness sequence for s08 full run

- s07b band exhausted (watcher stops; `07b_rounds.json` `resume` ≥ block_hi).
- Run s07 sweep: `S07_ORPHAN_LIMIT=7000000` (absorb ALL remaining orphans,
  no floor) — after an optional post-fix `--limit 4000 --dry-run` smoke that
  shows `winning partners` in the hundreds (the pre-fix smoke printed `=1` =
  the knn_query label bug's signature; check4 already proved the fix).
- Apply this patch to s08, py_compile, then launch full
  `python3.11 -m scripts.s08_metabary` (no --force; structurally-guarded,
  no 08 checkpoint). Expect ≈6–7h+; L15 child pass = 12,051,296 BEs, load
  rate (~360/s under contention, should improve alone on the box) paces it.

## SMB candidates file (cognitive agent results)

- Path: `/workspace/bary-vector/cognitive/batches/smb_candidates.jsonl`
  (driver output of `cog_mb_loop.py`; 335 records as of 2026-09-18).
- Copy placed at workspace root: `/workspace/bary-vector/smb_candidates.jsonl`
  (identical content, per user request 2026-09-18).
- Author mix: 311 `qwen3.6:27b-q4_K_M-ctx64k@opencode`, 24
  `big-pickle@opencode-0.5`.

# S10 SERVING INDEX — 1024-DIM SCALAR-QUANTIZED (FEASIBILITY + DECISION)

Status: **AGREED DIRECTION (2026-09-18), deferred-pending.** Do NOT run
`s10_index` against `barygraph_all` until this is executed. Decided with user:
full-corpus quick search must be 1024-dim scalar-quantized; the deferred
decision was to keep building 7b/7/8 first (they run on host RAM, mongot-
independent) and execute this at the clean break before s10.

## Why the 4096 spec cannot serve

- Final corpus ≈ 42–45M docs, each carrying a 4096-dim `vector` field.
- Payload = 44M × 16 KB ≈ **700 GB** of float32 vector data. The POC anchor
  (6.78M × 768-dim = 20.8 GB) is what forced `10g→20g`; scaling ~34× wants a
  many-hundreds-of-GB container. Host: 251 GB total / ~230 GB available. Not
  feasible — payload alone exceeds the box, before WT cache (9.5 GiB), the
  HNSW graph, or build peak.
- NOTE: mongot `scalar` quantization alone (without dim reduction, i.e. on the
  existing 4096 field) still indexes 4096-d vectors → the ~700 GB graph/
  traversal problem stays. The dim reduction must live in the stored field.

## Scale anchors for the 1024 route

| serving pack | payload | verdict |
|---|---|---|
| 4096 float32 (current spec) | ~700 GB | impossible |
| 1024 float32 | ~176 GB | needs near-host-size container, no margin |
| **1024 scalar (int8-style)** | **~45 GB** | **feasible** in 64–96 g container |
| 512 scalar | ~22 GB | trivially feasible, more precision loss |

Quality note: the graph was ALREADY built at 1024 — every pairing/bridge in
s04/s07b/s08 chose parents via `lib/match._project_match` (seeded Gaussian,
4096→1024, L2-normalized). Serving at 1024 is *more consistent* with the
graph's own decision space than 4096.

## mongot capability smoke test — PASSED (2026-09-18)

On a scratch `barygraph_test_vector_smoke` collection (dropped after): local
mongot 8.3.4 accepts 1024-dim vector indexes, both plain float32 and with
`"quantization": "scalar"`. Both built to READY and answered `$vectorSearch`
(note: option token is `scalar`, NOT `int8`). No dimension-support blocker.

## Execution steps (at the clean break — post-s08, pre-s10)

1. **Backfill stage (new, e.g. `s10b_projection`):** band-scan all docs by
   pure `_id` windows (no fat scans; `count_documents` stalls at 35M+ docs).
   Per doc: read `vector` (4096 f32) → `_project_match` → L2-normalize →
   scalar-quantize to int8 with per-doc `vector_scale` (same convention as the
   int8 payloads in `lib`/enrichment) → store as new field `vector_m`.
   Writes: batch 2900, unacked-stamp style like s07b, resume-safe via
   `pipeline_state_all/10b_projection.json` checkpoint.
2. **Index spec edit:** `indexes/vector_index.json` → path `vector_m`,
   `numDimensions: 1024`, `"quantization": "scalar"`, keep filter paths
   (doc_type, level, edge_type, node_type).
3. **Container memory:** raise docker-compose `mem_limit` 20g → **96 g** and
   restart mongod — ONLY at the clean break, deliberately, never mid-s07b.
4. **Run `s10_index`** — build on ~45 GB int8 payload; expect build peak
   ~20–50 GB, fits 96 g.
5. **Quality gate BEFORE wiring to MCP:** compare top-k neighbor agreement
   (1024-scalar index vs a persisted offline 4096 HNSW over a query sample,
   `MATCH_DIM=1024` cosine); MUST confirm rank overlap first.
6. **MCP/`cog` query path:** embeds qwen3 4096 → project via the same seeded
   projector → `$vectorSearch` on `vector_m`. Currently built for poc 768; the
   all-build serving path is a config switch (EMBED_DIM + projection keyed off
   env), not code surgery.
7. **This section stays authoritative** for the s10 re-run; revisit before
   executing if any step is stale.