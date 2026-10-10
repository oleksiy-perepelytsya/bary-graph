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

## Planned: Jev fast-check + candidate labeling (NOT yet wired)

- Integrate a call to the **Jev model** (`jev-latest`; hosted
  `api.typesafe.ai/v1/systemone` or self-hosted) at the seed/candidate stage to:
  - fast-check a retrieval/seed, and
  - label SMB candidates against a specific **rule set**, giving a second
    estimation used for candidate review, inclusion into the schema, or
    revision with further iterations.
- No endpoint/credentials present in env or `opencode.json` yet. Wire it when
  the endpoint + rule set are supplied.
