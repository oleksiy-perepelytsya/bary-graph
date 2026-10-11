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

- Tool: `cognitive/not_sure.py`. Call it when a retrieval/judgment is uncertain.
  It speaks the **Jev API** (`POST /v1/systemone`): pass `--state` (the text to
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
  * With one *neutral* template it said **no to 10 of my 11 session SMBs**
    (crab/crane/gin/ram/goose/dog/horse/spud/cat/hurdy-gurdy; only tanker was
    a marginal yes). Treat that as a second estimation, not a veto — but note
    it is consistent with the user's Oct-11 critique that recent builds drifted
    into trivial animal↔machine analogies.
- **Rules for loop use:** always feed a *fixed, neutral* state template (never
  leading) so runs are comparable; treat "no" as a prompt to re-examine, not a
  denial; never let it override value judgment (no coercion).
- **Full-history read (Oct-11; 55 structural SMBs — random 25, then all 55):**
  with the fixed **term-list** template it said **yes to 1/55 (2%)**, mean p(yes)
  0.32, corr(child_cosine, p(yes)) ≈ 0. So it does **not** single out the recent
  animal↔machine builds — it (near-constantly) says no to essentially everything,
  the earlier "sophisticated" builds included (possession-law-demonology,
  laughter-fracture, renormalization, omen-augury, …). **Prose framing on the
  full 55 → yes 15/55 (27%), mean p(yes) 0.443** — same candidates, 13× the
  yes-rate, purely from phrasing. p(yes) barely moves with content ⇒
  **tev1:0.8b is framing-dominated and not a usable discriminator on this
  question**, and the earlier "no to 10/11" was *not* evidence about those
  specific SMBs. Ledger: `cognitive/batches/oracle_reads.jsonl` (110 reads),
  tool `cognitive/oracle_read.py`. Next: a forced-choice/comparative question to
  see if *any* framing separates; then consider `nimble`. (Note:
  `smb_builds.jsonl` is heterogeneous — 55 structural entries + 6 legacy
  DOI/proposal records; filter on `name`+`cm1_terms`.)
