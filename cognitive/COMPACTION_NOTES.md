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

## Jev oracle — `cognitive/not_sure.py`

- Tool: `cognitive/not_sure.py`. Call it when a retrieval/judgment is uncertain.
  It speaks the **Jev API** (`POST /v1/systemone`): pass `--state` (the text to
  judge) and either a quick `--ask`/`--options` or a full `--questions-json`
  with typed `choice` questions + `criteria`; it returns `{choice, probabilities}`
  per question. Override `--system` to impose a stricter contract (e.g. rule-set
  labeling of SMB candidates) without editing the script.
- Model: `--model`, else `$JEV_MODEL`, else `tev1:0.8b` (Together AI's tiny
  decision model — the chosen Jev stand-in, small enough for this link).
  `nimble` (Bespoke Labs 9B) is a drop-in when bandwidth allows.
  Host: `$JEV_HOST` or `http://127.0.0.1:11435`.
- **Blocker / status:** decision models need ollama **≥0.35**; the embedding
  server on `:11434` is the old container (0.21.0) and is left untouched. A
  *separate* newer ollama runs on `:11435` (install `/tmp/opencode/ollama_040`,
  models `/tmp/opencode/jev_models`). Deployment in progress: egress is
  ~150 KB/s and flaky, so the 1.44 GB ollama binary is a slow **resumable**
  background download (`/tmp/opencode/ollama.tar.zst`); a watcher
  (`/tmp/opencode/setup_jev2.sh`) then extracts, serves on :11435 and pulls
  `tev1:0.8b`. Plumbing already verified end-to-end (clean model-404 path).
- Intended loop use: fast-check a seed/retrieval, and label SMB candidates
  against a rule set as a second estimation for candidate review / schema
  inclusion / revision iterations. Never a coercion — another estimation.
