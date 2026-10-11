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

**Productive signal = a pivot word shared by two DISTANT docs** ("one word, two
lives"). A low `child_cosine` between two clusters that share a word is the
target — it means the word is doing real double duty across domains.

**Level arithmetic** for a structural SMB:

```
children   = L      (cm1, cm2 — one node/cluster from each the two lives)
bridge     = L - 1  (a doc that itself carries the pivot word)
SMB        = L - 2
```

**Cycle protocol**

1. **Seed** — draw a random MetaBary/BaryEdge (L12/L13) as a starting point.
2. **Probe** — vector-search its neighbourhood; keep neighbours that **share a
   leaf word** with the seed, ranked by ascending cosine. (`loop_pivot.py`)
3. **Cosine check** — confirm the two clusters are actually distant
   (target ≈ 0.5–0.65).
4. **Bridge hunt** — search the L-1 neighbourhoods of *both* clusters for a doc
   carrying the pivot word. (`find_bridge.py`)
5. **Build** — only if a real pivot-word bridge exists (see the admission rule).
   Build via `scripts/mcp_server._create_structure_meta_bary_body`, then append
   to the manifest.
6. **Record** — append the cycle outcome to `loop_cycles.jsonl`, update
   `loop_state.json`.

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

A **decision model** (Jev API, `POST /v1/systemone`) that returns a choice +
probabilities for a given state. Use it when a judgment is uncertain — a *second
estimation*, never a coercion or a fact.

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
python3 cognitive/loop_pivot.py [N=15] [LEVELS="13,12"] [--seed N]
# JSON lines: {"cos","shared","seed":{id,lvl,w},"nb":{id,lvl,w}}
# --seed makes a draw reproducible for replay/diff.
```

### `find_bridge.py` — bridge hunt

Given a pivot word and the two child clusters' representative ids, take the union
of their L-1 neighbourhoods and print every doc whose leaf words carry the pivot.

```bash
python3 cognitive/find_bridge.py PIVOT SEED_ID [SEED_ID ...] \
        [--keys a,b,c] [--level 12] [--limit 70] [--num-candidates 800]
# prints MATCH <doc_id> [leaf words...]
```

### `oracle_read.py` — formalized oracle reads over the manifest

Runs `not_sure.py` over the SMB manifest with a **fixed** template + question (so
runs are comparable) and appends every read to a durable ledger
`cognitive/batches/oracle_reads.jsonl`.

```bash
python3 cognitive/oracle_read.py --all --template both      # all 55, both framings
python3 cognitive/oracle_read.py --sample 25 --seed 20261011 --template term
```

Records: `at, model, template, question, criteria, name, smb_id, author,
child_cosine, choice, p_yes, probabilities, confidence`.

> **Result (Oct-11, full 55):** term-list framing → **yes 1/55 (2%)**, mean
> p(yes) 0.32; prose framing on the same 25 → **yes 7/25 (28%)**. The oracle is
> **framing-dominated and near-constant**, so absolute yes/no is meaningless and
> it must never gate admission. See `COMPACTION_NOTES.md`.

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
| `smb_builds.jsonl` | 61 | **build manifest** — every SMB minted (see schema below) |
| `loop_cycles.jsonl` | 357 | per-cycle ledger (seed + outcome) |
| `loop_state.json` | — | cycle counter, commit watermark, self-check |
| `smb_candidates.jsonl` | 366 | candidate notes (rationale + probes), re-consult on search |
| `oracle_reads.jsonl` | — | oracle reads over the manifest (append-only; see `oracle_read.py`) |
| `smb_proposals.jsonl` | 23 | DOI-anchored proposals (older pipeline) |
| `smb_validation.jsonl` | 9 | proposal grounding checks |
| `terms_batch.jsonl` | 21 | paper → terms extraction |
| `papers_batch.jsonl` | 21 | arXiv paper pool |
| `s08_growth_runs.jsonl` | 9 | s08 triad-growth run log |
| `smb_candidates_baseline_big-pickle.jsonl` | 23 | baseline candidate set |

### Schemas

**`smb_builds.jsonl`** — *heterogeneous*: 55 structural entries + 6 older
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
  "cycle": 355, "note": "feline ↔ nautical catfall"
}
```

The 6 legacy entries carry `doi/verdict/status/proposal_index/reason` and no
name/terms — filter on `name` + `cm1_terms` before treating the file as uniform.

**`loop_cycles.jsonl`**: `{cycle, seed_mb_id, seed_level, outcome, note, at}`.
`outcome` ∈ `similarity-cluster`, `no-adjacency`, `built`, `pivot-half`,
`candidate`, `bridge`, `held`, `rejected`, `proper-noun-cluster`,
`morphological-cluster`, … (observed distribution ≈ 45% similarity-cluster,
19% no-adjacency, 11% built).

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
