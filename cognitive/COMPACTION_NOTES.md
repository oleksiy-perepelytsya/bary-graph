# Compaction notes — native cognitive loop (STANDING)

Carried into the next session/compaction so the loop's correct context survives.
This mirrors (and corrects) the conversation checkpoint. Read this first when
resuming the loop.

## Schema — authoritative (old `cog` schema is LEGACY)

- The **old `cog` schema is legacy** (the `cog_cycle.py` / `cog_mb_loop` era and
  its `cognitive_*` doc shapes / older proposal records). Do **not** treat it as
  the reference.
- **What we use now is the right one:** structural SMBs minted via
  `scripts.mcp_server._create_structure_meta_bary_body`, stored with
  `source='structural'`, an explicit `bridge_id` (+ `cm1_id`/`cm2_id` at the
  child level), `author='big-pickle@opencode-0.5'`, `doc_type='baryedge'`.
  In the DB only `accumulated_weight` is stored; `q_mb_raw` / `child_cosine`
  are build-response-only diagnostics. Manifest:
  `cognitive/batches/smb_builds.jsonl`.
- Architecture reference: `BaryGraph_Architecture_v1_0.md` (v1.0 supersedes the
  Kaikki PoC v0.6; the "MetaBary triad" is *one registered composition
  operator*, not a universal rule; the v0.6 formulas are not normative).

## Requirements (corrected)

- Commit only every ~10 cycles (never per-cycle). Choose seeds at random;
  self-check at each commit (continue the thread vs. take a new random seed) to
  avoid fixation.
- No SMB coercion: value judgment is the sole admission criterion;
  `child_cosine` always recorded (diagnostic; low-q is never a denial; flag
  `<0.20` for recheck). `source='structural'`, explicit `bridge_id`, author
  `big-pickle@opencode-0.5`; manifest `smb_builds.jsonl`.
- Prefer cheap direct lib calls (non-MCP) for the loop's own lookups.
- Report wall-clock times/ETAs in **UTC+2 only**.
- `opencode.json` stays uncommitted (`cog`/`mnemo` are VPS-only); the public
  tunnel serves read-only (no `create_*`).
- _(The former "do not alter s08/s09 thresholds" clause is dropped — those are
  all-build pipeline stages, not part of the native loop.)_

## Blocked

- (none) — the native loop runs on this box; it does not depend on the
  VPS-bound pipeline runner.
- _(The former note about `cog_cycle.py` being unusable was removed. `cog_cycle` /
  `cog_mb_loop` are simply the LEGACY schema path; not a blocker.)_

## Loop operation

- **Sampler**: plain random draw (no automated cluster filter by default). If a
  drawn retrieval is unproductive, **skip it by judgment and re-query** rather
  than baking filter rules into the sampler. `loop_batch3.py` keeps the old
  filter behind an explicit `filter` argv flag only.
- **Keep candidates in context**: `cognitive/batches/smb_candidates.jsonl` is
  retained and should be re-consulted on new searches — e.g. noticing that a
  previously near-miss candidate now has its missing bridge ("aha" catch).
- **Productive signal (KEY, batch V).** Random MetaBary seeds are homogeneous:
  a seed's *own* child_cosine (cosine of its `cm1`/`cm2` vectors) runs min ~0.79,
  median ~0.92 — i.e. a random seed almost never straddles two distant branches,
  which is why random draws yield so little. The productive unit is a **pivot
  word shared by two docs that are DISTANT in vector space** (low child_cosine):
  `yarn` thread/tale (0.514), `tanker` military crewman / liquid-cargo vessel
  (0.617). Find it while enriching a seed: look for neighbours that share a word
  with the seed but sit far from it. Children at the same level L, bridge at
  L-1, SMB at L-2. Well-formed SMBs sit at child_cosine ~0.24-0.62; >0.9 = the
  two children are co-located (flag for recheck, still buildable).
- **Sampler fix.** `_random_metabary_docs(level, n, independent=True)` now draws
  n *separate* random offsets (n index-only probes). The default window walk
  returns `n` consecutive index entries — one contiguous content cluster — so
  repeated draws / seed batches collapse onto near-duplicate seeds.
  `loop_batch3.py` now passes `independent=True`.
- **Pivot-build pipeline (batches V/W/X).** To turn a pivot into an SMB:
  1. **Scan** (`cognitive/loop_pivot.py`): draw independent seeds, vector-search
     each, keep neighbours that SHARE a leaf word but sit far away.
  2. **Probe** (`loop_probe_word.py`, one-off): embed the pivot word (or a
     sense description) to locate the FAR cluster of the other life.
  3. **Cosine** (`childcos.py` / inline): confirm the two clusters are distant
     (target ~0.5-0.65; the good ones so far: yarn 0.514, tanker 0.617,
     hurdy-gurdy 0.610, crab 0.627).
  4. **Bridge hunt** (`cognitive/find_bridge.py`): the bridge is an L12 doc
     containing the pivot word, or a shared-concept doc *equidistant* from both
     clusters. Take the union of the two clusters' L12 neighbourhoods and keep
     docs whose leaf words contain the pivot word.
  Then build via `_create_structure_meta_bary_body` and record.
- **Environment caveat (Oct-11).** While the all-build runs, the host is CPU
  saturated (load ~2-7 on 4 cores) and the poc embedder answers in ~20 s per
  call (the ollama runner + mongot dominate). So: (a) never embed per-word in a
  loop — batch or avoid; (b) everything that uses a STORED vector
  (`$vectorSearch` with an existing doc vector, DB reads, `cm_leaf_words`) is
  fast (~0.1-0.3 s) and is the right toolkit for discovery; (c) **all embeddings
  now come from the new ollama at :11435** (`/tmp/opencode/ollama_040`, models
  in `/tmp/opencode/jev_models`). Both `qwen3-embedding:0.6b` (1024) and `:8b`
  (4096) were copied out of the old container and verified (cos ≈ 0.9997 vs the
  old vectors — same model, tiny version/CPU numerics). `.env` `OLLAMA_URL` is
  now `http://localhost:11435`; the old `barygraph-ollama` container was
  **stopped** (freed ~6.5 GiB; host avail 3.4→9.8 GiB). The running barygraph
  MCP server loads `.env` once at startup, so it must be **restarted** to use
  :11435. `.env.build-all` is unchanged (`OLLAMA_URL=http://ollama:11434` is part
  of a docker-network bundle that also uses `mongodb:` service names) — the
  all-build is not running; revisit before the next all-build stage.

## Jev oracle — `cognitive/not_sure.py`

- **ROLE — CORRECTED by user (Oct-11).** The oracle is a **decision aid for the
  agent's own choices**, never a validator of SMBs. An SMB is a *product of
  analysis*, admitted by value judgment — the sole admission criterion; it is
  not something to check deterministically, and asking a small decision model
  "is this SMB productive?" is a category error (measured near-constant, below).
  **Use it where the AGENT is genuinely unsure**: which seed/thread/bridge to
  pursue, continue this line vs re-query, which of several fine options to take.
  Advisory and suggestible — a second estimation, never a veto, never a gate.
- **Coin-flip principle (endorsed by user, Oct-11):** the oracle is a *mirror*,
  not a decision-maker. Flip it, see the answer, and read **your own reaction**
  to that possible decision: if the oracle's pick is one you'd be happy with,
  go there; if it makes you uneasy, go the other way (or re-think) — the point
  is a cheap second point of view on a decision you were already about to make.
  Prefer it over exhaustive analysis of *every* direction: quick test first,
  dive deep only into the one thread it surfaces (still not forced to follow it).
- Tool: `cognitive/not_sure.py`. Call it at a decision point where the agent is
  torn. It speaks the **Jev API** (`POST /v1/systemone`): pass `--state` (the text to
  judge) and either a quick `--ask`/`--options` or a full `--questions-json`
  with typed `choice` questions + `criteria`; it returns `{choice, probabilities}`
  per question. Importable for loop use: `from cognitive.not_sure import judge,
  ask`. Extras: `--check` (verify the model is served), `--short` (one line per
  question), `--retries`.
- Model: `--model`, else `$JEV_MODEL`, else `tev1:0.8b` (Together AI's tiny
  decision model — the chosen Jev stand-in, small enough for this link).
  `nimble` (Bespoke Labs 9B) is a drop-in when bandwidth allows.
  Host: `$JEV_HOST` or `http://127.0.0.1:11435`.
- **Status (Oct-11): WORKING end-to-end.** `tev1:0.8b` is pulled and served on
  `:11435` (ollama 0.40.2 host binary, `/tmp/opencode/ollama_040`, models
  `/tmp/opencode/jev_models`). All three input shapes verified: `--raw`,
  single-question, multi-question — each returns `{choice, probabilities,
  confidence}`. `--check` confirms the model is present.
- **How it behaves (measured Oct-11, must inform any use):**
  * **Deterministic** — the same state+questions returns the same numbers
    (token scoring, no sampling): repeated crab call gave p(yes)=0.440 ×3.
  * **Highly framing-sensitive / suggestible.** Same crab candidate: terse
    term-list → no (p(yes) 0.145); plain prose → yes (0.652); rich context →
    no (0.476); leading/flattering phrasing → yes (0.803). The oracle *agrees
    with how you talk to it*.
  * **Criteria wording shifts the answer too**: same state, "distinct senses"
    → no (0.440) vs "worth recording as a structural SMB" → yes (0.595).
  * **`confidence` is NOT reliability** — it ranged 0.000–0.435 in these runs
    and did not track agreement with my admissions.
  * Calibration (full history, 55 structural SMBs): term-list framing →
    **yes 1/55 (2%)**; prose framing → **yes 15/55 (27%)**; corr(child_cosine,
    p(yes)) ≈ 0. **Near-constant** ⇒ it cannot discriminate SMB quality at all.
    My earlier "no to 10/11 of my builds" was this same misuse — it said no to
    *everything*, so it never confirmed the user's critique either way.
- **Rules:** when consulted at a genuine decision point, feed a *fixed, neutral*
  template (the oracle is suggestible — phrasing moved the same candidates
  2%↔27% yes), treat "no" as "re-examine / re-ask with crisper options", and
  never let it override value judgment or replace analysis. If a consult
  influenced a decision, note it in the loop ledger.
- **Artifacts (negative result):** the Oct-11 calibration reads live in
  `cognitive/batches/oracle_reads.jsonl` (110 reads: term+prose × 55), tool
  `cognitive/oracle_read.py` (marked as a calibration experiment). They prove
  the tiny model must not gate SMBs; they are not a validation channel. (Note:
  `smb_builds.jsonl` is heterogeneous — 55 structural entries + 6 legacy
  DOI/proposal records; filter on `name`+`cm1_terms`.)

## Bridges structural audit (Oct-11) — tool: `cognitive/smb_audit.py`

Full audit of every manifest structural row vs `barygraph_poc`:
- **All 47 SMB docs that exist in the DB are structurally consistent**: children
  same level L, bridge L-1, SMB L-2, explicit `bridge_id`, distinct ids,
  references resolve, vectors present, manifest ids match doc ids. 100% pass.
- **11 of 58 manifest rows are PHANTOM claims — no doc in the DB (direct _id
  lookup NOT FOUND).** Three clusters: (1) 3 early qwen3.6:27b rows
  (caterpillar/half-track, flag/anthem, GH/TNF, Oct-01 era id space
  `6aaaec6b…`); (2) 3 legacy DOI-schema rows (no name, source=None,
  `6aab0db7/8…`); (3) 5 rows with sequential ids `6aab426fb6c3fc4deb254b82–86`
  (scale-ladders, embryo-segment/phonology, nucleation-and-growth/rumour,
  topology/surveying, value-appraisal) — smells like a batch insert that never
  landed in poc (possible wrong-env insert; cf. the phantom-id rabbit hole in
  AGENTS.md). **Decision pending (user): rebuild the 11 into poc if the analysis
  still stands, or strike/flag them in the manifest — do NOT silently edit.**
- Doc-type nuance learned: SMBs built at L13 (e.g. the two `biternary` SMBs)
  have L15 sense / L14 word *node* docs as children — `doc_type` must accept
  `baryedge` | `node`; the audit now does.
- **Build gate (Oct-11):** `cognitive/build_gate.py` — strict pre-insert schema
  check mirroring the MCP builder (children same level L, bridge L-1, SMB L-2,
  refs resolve, vectors present, ids distinct). Errors BLOCK the insert;
  warnings (child_cosine < 0.20, pivot missing from bridge leaves) never do.
  The loop inserts an SMB only through `create_structural(...)`; GATE FAIL ⇒
  record the outcome, do not force. `smb_audit.py --all` afterwards must agree.
