# Project Goal (persist across sessions)

The long-term aim of this whole effort — beyond any single feature, benchmark, or
batch run — is to **improve the agent's cognitive memory** and, ultimately, to give
the agent a **personality-like layer built on BaryGraph Structure MetaBarys (SMBs)**.

> Timezone: the user's local time is **UTC+2**. Pipeline/runbooks/logs use UTC
> unless stated otherwise; convert user-reported "evening"/"morning" windows
> accordingly. When stating any wall-clock time or ETA to the user in chat,
> provide **only the UTC+2 time, already converted** (e.g. "18:32 your time") —
> do not give both UTC and UTC+2.

Concretely:
- The cognitive-pipeline work (extraction, SMB analysis, proposals, batches,
  `cog_*` MCP tools, `opencode run` cycles with an ollama model) is a means to
  that end, not an end in itself.
- Every working piece of the pipeline — terms in `cognitive/batches/terms_batch.jsonl`,
  proposals in `smb_proposals.jsonl`, validated SMB triads — is material that
  should eventually feed the model's own persistent associative memory and
  persona layer.
- Work should therefore favor the *real pipeline* and its durable artifacts
  (`cog_cycle.py`, `cog_loop2.py`, `mcp_int_cog.py`, batch JSONL files) over
  throwaway shims. If a piece of plumbing is being tested, test it the way it
  will run in production (GPU will be available and fast; CPU slowness is a
  transient constraint, not a design driver).
- No SMB coercion: proposals are coordinates to investigate, and only the user
  decides when proposals become real SMBs in Mongo.

## SMB admission policy (user decision, Sep-16 2026)

There is **no q rule** for SMBs. Value — the user/cognitive judgment that a
triad is a productive adjacency — is the sole admission criterion. The pipeline
0.90 child-cosine threshold governs *pipeline* MetaBary clustering only; it is
**not** an SMB gate. When building an SMB:
- Always record `child_cosine` on the candidate/manifest (diagnostic, never a
  denial).
- Low-q SMBs are created like any other; if `child_cosine < 0.20` (floor), note
  them for later recheck — but create, never deny on q alone.
- Built SMBs live in poc with `source='structural'`, the explicit `bridge_id`,
  and `author` = the model/person signature. Build manifest:
  `cognitive/batches/smb_builds.jsonl`.

# Databases and .env files

There are exactly **two live Mongo databases** plus a test prefix, selected by
which env file the shell has sourced. `Settings.load()` reads the process
environment; **if you don't source a `.env*` file, every probe/script silently
targets `barygraph_poc`** — this mismatch caused a long "phantom last_id /
mongot nondeterminism" rabbit hole (probes hitting poc while the pipeline wrote
`barygraph_all`). Always source the matching env before probing.

## Databases
- **`barygraph_poc`** — the live, user-facing PoC index (what the MCP server and
  the `cog`/`barygraph` MCP tools serve). Built from a single language
  (`en`-style limited kaikki). Embeddings: `nomic-embed-text:v1.5`, dim **768**.
  Word heads around `6a603105…` in `_id` space. Pipeline state:
  `pipeline_state/`.
- **`barygraph_all`** — the in-progress all-languages build (`KAIKKI_LANGS=*`;
  counts mid-build, e.g. 10,256,975 L14 words / 12,718,626 senses as of the
  Sep-16 2026 s05 run). Embeddings: `qwen3-embedding:8b`, dim **4096**. Word
  heads around `6a9a2b40…` in `_id` space (different `_id` range from poc — do
  not compare ids across DBs). Pipeline state: `pipeline_state_all/`. Serves no
  user traffic; stages s01–s04 done, s05 running as of Sep-16.
- **Test DBs** — `MONGO_TEST_DB_PREFIX=barygraph_test_`; integration tests
  refuse to touch any DB name not matching this prefix (safety guard, do not
  bypass).

## .env files
- **`.env`** — default (loaded when nothing else is sourced). Points at
  `barygraph_poc`, `MONGO_COLLECTION=barygraph`, `EMBED_MODEL=nomic-embed-text:v1.5`,
  `EMBED_DIM=768`, `BATCH_SIZE=512`, `PIPELINE_STATE_DIR=pipeline_state`.
  The MCP server always runs with this config and must keep serving poc even
  while the all-build runs.
- **`.env.build-all`** — isolated config for the all-languages build; **sourced
  by launch scripts only, never by the MCP server**. Points at `barygraph_all`,
  `KAIKKI_LANGS=*`, `EMBED_MODEL=qwen3-embedding:8b`, `EMBED_DIM=4096`,
  `BATCH_SIZE=2048`, `PIPELINE_STATE_DIR=pipeline_state_all`, plus
  `EMBED_CACHE_FILE` (append-only JSONL embed cache on /storage; dedupes ~21% of
  repeated sense texts). Key difference vs main env: **embed model/dim differ**
  (768 vs 4096).
- **`.env.example`** — template; never used at runtime.