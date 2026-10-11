# cognitive/ — the native SMB loop (toolkit + data)

This directory is the **working surface of the native cognitive loop**. The loop
discovers *productive adjacencies* in BaryGraph, mints them as **Structural
MetaBarys (SMBs)**, and records everything durably. The long-term aim is an
agent cognitive-memory / persona layer built from those SMBs — so every artifact
here is meant to be reused, not thrown away.

> Standing context (corrected requirements, sampler policy, blockers) lives in
> `COMPACTION_NOTES.md`. Read that first when resuming. This README is the
> *tool and data reference*.

---

## 1. What the loop does

> **The loop is a thinking loop, not a build loop.** A retrieved coordinate is
> a **kindle**: I think over it, self-aware, and record what it fires in **my
> own abstract BaryGraph** (`agent_memory/`, the real workspace) in
> `batches/instigations.jsonl`. Building an SMB in the substrate is an optional,
> downstream *share* — not the cycle's goal. A cycle with a strong instigation
> and **zero builds is a good cycle**. See `agent_memory/README.md`.

**Productive signal = a pivot word shared by two DISTANT docs** ("one word, two
lives"). A low `child_cosine` between two clusters that share a word is the
target — it means the word is doing real double duty across domains. **Caveat
(Oct-11):** the metric also ranks *dead homonymy* (shock=lapdog, strain=deer
track) as low as real two-lives; low cosine is a prompt to think, not an answer.

**Level arithmetic** for a structural SMB:

```
children   = L      (cm1, cm2 — one node/cluster from each the two lives)
bridge     = L - 1  (a doc that itself carries the pivot word)
SMB        = L - 2
```

**Cycle protocol** (analysis method: `prompts/native_loop.md`)

1. **Seed** — draw a random MetaBary/BaryEdge (L12/L13) as a starting point.
2. **Probe** — vector-search its neighbourhood; keep neighbours that **share a
   leaf word** with the seed, ranked by ascending cosine. (`loop_pivot.py`)
3. **Cosine check** — confirm the two clusters are actually distant
   (target ≈ 0.5–0.65).
4. **Thin by judgment, not by filter** — read the hits: same-domain blob?
   component-word coincidence? genuine two-lives? Skip blobs *as blobs*
   (recorded, not committed); carry ≤1–2 threads forward. The oracle
   (`not_sure.py`) may help at a genuine fork — never as a validator.
5. **Bridge hunt** — search the L-1 neighbourhoods of *both* clusters for a doc
   carrying the pivot word. (`find_bridge.py`)
6. **Analyze** — one-pass analysis rubric (`cognitive/build_rubric.py`):
   distance / distinctness / mechanism / grounding evidence + scores, recorded
   in the build's `analysis` block. It is a decision aid, **never a gate**.
7. **Build** — only if a real pivot-word bridge exists (see the admission
   rule). **Every insert passes the strict schema gate**
   (`cognitive/build_gate.py` — same invariants the MCP builder
   enforces: levels L/L-1/L-2, references resolve, vectors present,
   ids distinct). GATE FAIL ⇒ do not insert, record the outcome.
   Then append to the manifest via `ledger.record_build(...)`.
8. **Verify + record** — `smb_audit.py` after insert; append the cycle to
   `loop_cycles.jsonl` (with `probes_run` + `rationale` — see schema below) and
   update `loop_state.json`.

### Admission rule (user decision — no coercion)

- **Value judgment is the sole admission criterion.** There is **no `q` rule**
  for SMBs. `child_cosine` is a *diagnostic*, recorded on every candidate — never
  a denial. Flag `< 0.20` for a later recheck, but still build.
- **Build only when a meaningful pivot-word bridge exists** (an L-1 doc literally
  carrying the pivot word). If cm2 doesn't literally carry the word but a
  pivot-word bridge does, build and note it. If no pivot-word bridge exists,
  **HOLD** — do not force one.
- **No automated retrieval filtering.** If a retrieval is unproductive, skip it
  by judgment and re-query; do not filter mechanically.
- Built SMBs live in `barygraph_poc` with `source='structural'`, an explicit
  `bridge_id`, and `author` = the model/person signature.

### Commit policy

Commit the loop every **10–16 cycles** (one batch per commit), never per-cycle.
Use **random seeds**; at each commit, self-check (continue the thread vs. take a
fresh random seed) to avoid fixation. Re-consult `smb_candidates.jsonl` on new
searches so near-misses are not lost.

---

## 2. Tools

### `not_sure.py` — uncertainty oracle

A **decision aid for the agent's own choices** (Jev API, `POST /v1/systemone`)
that returns a choice + probabilities for a given state. Use it where the **agent**
is genuinely unsure — which seed/thread/bridge to pursue, continue vs re-query,
which of several fine options to take. It is **not** a validator of SMBs: an SMB
is a product of analysis admitted by value judgment, not something to check
deterministically (asking it "is this SMB productive?" was measured near-constant,
2% yes under a term-list framing, 27% under prose — see `oracle_read.py`).

```bash
# single question
python3 cognitive/not_sure.py --state "..." --ask "Is X genuine?" \
        --options yes no --criteria "yes=...,no=..."
# full control
python3 cognitive/not_sure.py --state-file cand.json --questions-json '{...}'
python3 cognitive/not_sure.py --short   ...    # one line per question
python3 cognitive/not_sure.py --raw     ...    # untouched API response
python3 cognitive/not_sure.py --check          # is the model served?
```

Importable for the loop:

```python
from cognitive.not_sure import judge, ask
judge(state, "Is this a productive adjacency?", ["yes", "no"],
      criteria={"yes": "...", "no": "..."})
# -> {"type": "choice", "choice": "yes",
#     "probabilities": {...}, "confidence": 0.07}
```

Model `tev1:0.8b` (override `--model`/`$JEV_MODEL`), host `http://127.0.0.1:11435`
(override `--host`/`$JEV_HOST`).

> **Measured behaviour (Oct-11) — read before trusting it.** It is
> **deterministic** (identical input → identical numbers), but **highly
> suggestible**: the same candidate flips yes/no with the *phrasing* of the state
> and the *wording* of the criteria (leading phrasing → 0.80; terse → 0.15).
> `confidence` is **not** reliability. Always use a **fixed neutral template**,
> treat "no" as "re-examine", and never let it override value judgment.

### `loop_pivot.py` — pivot scan

Randomly draw seeds, vector-search each neighbourhood, and print the neighbours
that share a leaf word (ascending cosine = most distant first).

```bash
python3 cognitive/loop_pivot.py [N=15] [LEVELS="13,12"] [--seed N] [--top N]
# JSON lines: {"cos","shared","seed":{id,lvl,w},"nb":{id,lvl,w}}
# --seed makes a draw reproducible for replay/diff.
# --top N = print the N most distant pairs (default 24); use a large N to see
#           every shared-word pair (needed to file per-seed outcomes).
# Every drawn seed is also emitted to stderr as {"drawn": true, ...} so a
# batch is fully recordable even when a seed yields no shared-word neighbour.
```

### `word_pivot.py` — word-anchored scan ("one word, two lives")

Alternative to `loop_pivot.py`. Where that scan looks at docs *near* a random
seed (so the shared words are the cluster's own texture, cosine ≥ 0.84), this
scan anchors on a **word** and enumerates the word's own lives directly:

1. take a word (a seed leaf-word, a candidate, or any list — **primary mode**),
2. load every L15 sense node of the word (`properties.word` is indexed),
3. cluster senses by cosine and find the most distant pair of **substantive**
   senses (drops obsolete/archaic/rare/slang, names, inflected forms),
4. rank by that distance (low = a real two-lives spread).

The word's L14 word node is reported as `bridge`: if two sense-lives are
distant, that node literally carries the word and sits between them — the same
shape as the accepted `lawn` SMB (L15 sense-lives → L14 word node → L13 SMB).

```bash
python3 cognitive/word_pivot.py WORD [WORD ...]     # primary: explicit words
python3 cognitive/word_pivot.py --draw 12 [--seed N] # random words (noisy)
# JSON lines: {word, senses, substantive, clusters, min_inter, A, B, bridge_id,
#              A_sense_id, B_sense_id}
# --thr F (cluster cosine, default 0.80), --max-skew N (cap senses, default 60)
```

> **Calibration (Oct-11):** known builds land low — lawn 0.205 (grass/linen),
> crane 0.235 (bird/machine), tanker 0.239 (crew/refuel), part 0.16, grommet
> 0.19 — so the tool does rank real two-lives. A scan hit is a **candidate**,
> not a build: still judge substantiveness, hunt the two lives as graph
> clusters, then `build_rubric` → `build_gate`. Random `--draw` is noisy on this
> dump (rare-word dominated); prefer explicit seed/candidate words.

### `find_bridge.py` — bridge hunt

Given a pivot word and the two child clusters' representative ids, take the union
of their L-1 neighbourhoods and print every doc whose leaf words carry the pivot.

```bash
python3 cognitive/find_bridge.py PIVOT SEED_ID [SEED_ID ...] \
        [--keys a,b,c] [--level 12] [--limit 70] [--num-candidates 800]
# prints MATCH <doc_id> [leaf words...]
```

### `build_rubric.py` — one-pass pre-build analysis rubric

A light analysis-quality check run *before* a build is decided. Computes
mechanical evidence from the graph (child cosine, cos(bridge↔each child),
shared leaf words, pivot presence) and scores four axes 0–5 —
**distance / distinctness / mechanism / grounding**. It is a **decision aid
for value judgment, explicitly NOT a gate and NOT an oracle**: it never
blocks, it has no q threshold, and its scores land in the build's
`analysis` block so near-misses stay auditable.

```bash
python3 cognitive/build_rubric.py CM1 CM2 BRIDGE [--pivot WORD] \
        [--cm1-terms a,b,c] [--bridge-terms x,y] [--cm2-terms p,q] [--json]
```

Importable: `rubric(coll, cm1_id, cm2_id, bridge_id, pivot, terms=...)`
→ `{"scores": {...0-5}, "evidence": {...}, "mechanism_test": "..."}`.

> **Calibration (Oct-11):** lawn (built) → distance 4 / distinctness 5 /
> mechanism 5 — the two-lives-with-bridge signal; bandicoot (HOLD) → mechanism
> 2 (pivot absent from the bridge, bridge lopsided toward one life — exactly
> why it was held); coven blob → distance 1 (cos 0.85, a seam not a triad —
> read mechanism "5" there as the bridge being *part of* the cluster).

### `ledger.py` — durable record writers

Formalized append-only writers for the loop records, so every entry carries
**analysis, not just a verdict**:

```python
from cognitive.ledger import record_cycle, record_build

record_cycle(412, seed_id, 13, "similarity-cluster",
             probes_run=["loop_pivot: 14 seeds '13,12'", "leaf: X<->Y same-domain"],
             rationale="All neighbours share the cluster's own texture; one "
                       "cluster with a seam, not two lives.",
             note="<short label>")
record_build(name="...", smb_id="...", cm1_id="...", cm2_id="...", bridge_id="...",
             cm1_terms=[...], bridge_terms=[...], cm2_terms=[...],
             child_cosine=0.42, levels={"children":15,"bridge":14,"smb":13},
             cycle=406, seed_mb_id="...", rationale="...",
             analysis={"distance":4,"distinctness":5,"mechanism":5,
                       "grounding":3, "mechanism_note":"..."})
```

Appends only (never rewrites/dedupes); state rolling stays with the caller /
`loop_state.json`.

### `oracle_read.py` — calibration experiment (negative result)

Runs `not_sure.py` over the SMB manifest with a **fixed** template + question and
appends every read to `cognitive/batches/oracle_reads.jsonl`.

```bash
python3 cognitive/oracle_read.py --all --template both      # all 55, both framings
python3 cognitive/oracle_read.py --sample 25 --seed 20261011 --template term
```

Records: `at, model, template, question, criteria, name, smb_id, author,
child_cosine, choice, p_yes, probabilities, confidence`.

> **DEPRECATED use — kept as the negative-result ledger.** Asking tev1 to
> validate SMBs is a category error (SMBs are analysis products, not oracle
> subjects). The run proved the point: term-list framing → yes 1/55 (2%), prose
> → yes 15/55 (27%) on the same candidates — framing-dominated and near-constant,
> so it must never gate admission. The oracle's real role is agent decision
> support (see `not_sure.py` above). See `COMPACTION_NOTES.md`.

### `build_gate.py` — strict pre-creation schema gate

Run before EVERY SMB insert (the loop builds with direct lib calls, not
the MCP server, so the check must not rely on it). Mirrors the invariants
of `scripts/mcp_server._create_structure_meta_bary_body`:
**errors block the build** (bad ids / missing docs / no vector / level
mismatch), **warnings never block** (`child_cosine < 0.20` flag, pivot not
found in bridge leaves — value judgment is the sole admission criterion).

```bash
python3 cognitive/build_gate.py --check CM1 CM2 BRIDGE --pivot WORD   # dry run
python3 cognitive/build_gate.py CM1 CM2 BRIDGE --pivot WORD           # check + insert
```

Importable: `gate(...)` / `create_structural(...)`.

### `smb_audit.py` — structural consistency audit

Verifies the *mechanics* of built SMBs (no content judgment): leveling rule
(children same level L, bridge L-1, SMB L-2), referenced ids resolve to real
docs, ids distinct, explicit `bridge_id`, `source='structural'`. Also reports
manifest-vs-DB drift (rows claiming SMBs that are not in the DB).

```bash
python3 cognitive/smb_audit.py --sample 10 --seed 20261011   # random 10
python3 cognitive/smb_audit.py --all                          # every structural row
```

> **Result (Oct-11):** 47/58 manifest rows exist in the DB and pass 100% of
> checks; 11 rows are phantom claims (no doc by direct `_id` lookup) — see
> `COMPACTION_NOTES.md` for the pending decision. Doc-type nuance: SMBs built at
> L13 have *node*-doc children (L15 senses / L14 words), which is valid.

### `agent_memory/` — my abstract BaryGraph + `ledger.record_instigation`

The layer the loop was missing. A retrieved coordinate is a **kindle**; the
primary act is introspection. `record_instigation(...)` appends one row to
`batches/instigations.jsonl`: the kindle, an honest `fire`
(none/low/medium/high), the thoughts it instigated, the internal nodes/edges it
activated in **my** graph, any `genuinely_new` connection, and an optional
`share` block (how it grounds into the real schema). See
`agent_memory/README.md` for the schema and the node/edge → SMB mapping.

```python
from cognitive.ledger import record_instigation
record_instigation(cycle=415, kindle={"seed_id": "...", "words": [...]},
                   fire="high", instigated=["...", "..."],
                   internal_nodes=[{"id": "agent:...", "label": "..."}],
                   internal_edges=[{"src": "agent:...", "dst": "agent:...",
                                    "rel": "...", "novelty": 0.6}],
                   genuinely_new="...", share=None, note="...")
```

`fire: none/low` is a first-class result — noticing which kindles are inert is
the self-awareness the loop is for.

### `scripts/mcp_server_ctl.sh` — serve/restart the MCP (repo root)

The streamable-http MCP server sits behind a cloudflared quick tunnel. The tunnel
URL is **precious** — `restart` never kills it (`tunnel restart` is the only
command that mints a fresh URL).

```bash
scripts/mcp_server_ctl.sh status | start | stop | restart
scripts/mcp_server_ctl.sh tunnel status | url | start | restart
```

Interpreter is auto-detected (first of `python3.11/python3` that can `import
mcp`); override with `MCP_PYTHON`. The server loads `.env` **once at startup**, so
an `OLLAMA_URL` change requires a restart.

---

## 3. Data files (`cognitive/batches/`)

| file | lines | what |
|---|---|---|
| `smb_builds.jsonl` | 62 | **build manifest** — every SMB minted (see schema below) |
| `loop_cycles.jsonl` | 435 | per-cycle ledger (seed + outcome + probes + rationale) |
| `instigations.jsonl` | 4 | **what a kindle fired in my graph** (see `agent_memory/README.md`) |
| `loop_state.json` | — | cycle counter, commit watermark, self-check |
| `smb_candidates.jsonl` | 369 | candidate notes (rationale + probes), re-consult on search |
| `oracle_reads.jsonl` | — | oracle reads over the manifest (append-only; see `oracle_read.py`) |
| `smb_proposals.jsonl` | 23 | DOI-anchored proposals (older pipeline) |
| `smb_validation.jsonl` | 9 | proposal grounding checks |
| `terms_batch.jsonl` | 21 | paper → terms extraction |
| `papers_batch.jsonl` | 21 | arXiv paper pool |
| `s08_growth_runs.jsonl` | 9 | s08 triad-growth run log |
| `smb_candidates_baseline_big-pickle.jsonl` | 23 | baseline candidate set |

### Schemas

**`smb_builds.jsonl`** — *heterogeneous*: 56 structural entries + 6 older
DOI/proposal records. Structural shape:

```jsonc
{
  "name": "cat/feline-and-nautical-catfall",
  "author": "big-pickle@opencode-0.5",
  "cm1_terms": ["biochipped", ...],     // child A leaf words
  "bridge_terms": ["cat o' nine tails", ...],
  "cm2_terms": ["cat-block", ...],      // child B leaf words
  "cm1_id": "6ac6...", "bridge_id": "6ac6...", "cm2_id": "6ac6...",
  "smb_id": "6acb...",                  // the doc in barygraph_poc
  "level": 11, "q_mb_raw": 0.250115, "accumulated_weight": ...,
  "child_cosine": 0.5563,               // DIAGNOSTIC ONLY (never a denial)
  "low_q_flag": false,                  // true iff child_cosine < 0.20
  "source": "structural", "built_at": "...", "seed_mb_id": "...",
  "cycle": 355, "note": "feline ↔ nautical catfall",
  "levels": {"children": 15, "bridge": 14, "smb": 13},  // newer rows
  "analysis": {                          // rubric scores that informed the build
    "distance": 4, "distinctness": 5, "mechanism": 5, "grounding": 3,
    "mechanism_note": "..."              // 1-sentence mechanism test
  },
  "rationale": "..."                     // 1-2 sentences: why value admitted it
}
```

The 6 legacy entries carry `doi/verdict/status/proposal_index/reason` and no
name/terms — filter on `name` + `cm1_terms` before treating the file as uniform.

**`loop_cycles.jsonl`**:
`{cycle, seed_mb_id, seed_level, outcome, probes_run, rationale, note, at}`.
`probes_run` is the list of probes actually fired for that draw; `rationale`
is the 1–2 sentence analysis — **required even for skips** (a skip without a
reason is noise). `outcome` ∈ `similarity-cluster`, `no-adjacency`, `built`,
`pivot-half`, `candidate`, `bridge`, `held`, `rejected`,
`proper-noun-cluster`, `morphological-cluster`, … (observed distribution ≈
45% similarity-cluster, 19% no-adjacency, 11% built).

**`loop_state.json`**:
`{cycles_done, last_commit_cycle, next_commit_cycle, loop, self_check, notes}`.

**`smb_candidates.jsonl`**:
`{source_mb_id, source_level, rationale, cm1_terms, bridge_terms, cm2_terms,
relation_summary, grounding_probes, author, cycle}`.

**`smb_proposals.jsonl`** adds `doi, cm1_edge, cm2_edge, bridge_edge, smb_id,
child_cosine, grounding_queries` to the candidate shape.

### SMB document in Mongo (`barygraph_poc`, collection `barygraph`)

`_id`, `doc_type='baryedge'`, `cm1_id`, `cm2_id`, `bridge_id`, `level`, `vector`
(binary), `parent_edge_id`, `connection_strength` (= q), `accumulated_weight`,
`source`, `author`, `created_at`/`updated_at`. **There is no `child_cosine` field
on the doc** — child_cosine lives only in the manifest.

---

## 4. Environment

- **`barygraph_poc`** — the live index the MCP serves. Embeddings
  `qwen3-embedding:0.6b`, dim 1024. (`barygraph_all` = the all-languages build;
  `pipeline_state_all/`; not served.)
- **ollama `:11435`** — host binary `/tmp/opencode/ollama_040`, models
  `/tmp/opencode/jev_models`; serves `qwen3-embedding:0.6b` (poc) and
  `tev1:0.8b` (oracle). The old `ollama:11434` container was retired Oct-11
  (all embeddings now come from `:11435`; `.env` `OLLAMA_URL` points there).
- **Prefer cheap direct lib calls** (`lib.config`, `lib.db`, `lib.embed`) for the
  loop's own lookups; MCP tools are for interactive investigation.
- `.env` is **not tracked by git** — config changes live only on the host.
  `opencode.json` is tracked-but-dirty and must **not** be committed.
