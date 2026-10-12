# S001 — GovKM continuity test: run plan (tensegrity pipeline)

Status: **PLANNED — not run.** Fixture verified; live prerequisites verified;
code untouched beyond the pipeline as built. Decision to proceed: user (later).

## What the test is

GovKM supplied `BaryGraph-S001-Fixture-v1.0.zip` (committed at repo root;
SHA256s verified against `MANIFEST-SHA256.txt`):
- Three synthetic source documents about a supplier-approval state:
  - **P01** — Procurement Director (Morgan Hale) approves **Supplier A** (2026-03-01).
  - **P02** — Analyst (Riley Chen) recommends **Supplier B**, no authority (2026-07-05).
  - **P03** — Procurement Director approves **Supplier C** and explicitly
    supersedes P01 (2026-07-07).
- No precomputed CONTRADICTS / SUPERSEDES edges are supplied; evidence is in
  the documents themselves.
- Deliverables at freeze: version/commit, config, ingestion order, errors /
  warnings / retries / interventions, graph export or structural description,
  **UTC freeze timestamp**. No challenge questions until after freeze.

## Verified live prerequisites (2026-10-12)

- Mongo 8.2.6 on 27117 (Community — mongot `$vectorSearch` unavailable; t10
  degrades gracefully; irrelevant for a freeze since retrieval is out of scope).
- Ollama on 11435 has `qwen3-embedding:0.6b` (1024-d) and `tev1:0.8b`
  (`capabilities: ["decision"]`).
- tev1 classifies the real S001 sentences (live probe below).

## Requirement coverage of the as-built pipeline

| Requirement | Status | Note |
|---|---|---|
| P01/P02/P03 distinct source events | **gap** | t01 is single-doc; needs multi-doc support |
| Preserve history; never delete/overwrite P01/P02 | ✅ | nodes keyed by doc_id; older docs untouched |
| Provenance + event-time vs observation-time | **gap** | text preserved, distinction not structured |
| Actor identity + stated authority | **gap** | preserved in sentence text only |
| Derived vs source-supplied relations | ✅ | `source` (inferred/reentry) + `role_source` |
| No answer-optimization | ✅ | no query stage exists |
| Header/front-matter of the P0X files | **gap** | current parser turns header lines into 4 spurious L14 titles and shifts section indices |

## tev1 relational finding (live probe — architectural boundary)

Binary role model (complement/contradict) only — no SUPERSEDES/CONTRADICTS
taxonomy. Observed on this corpus:

| Pair | role | p | reading |
|---|---|---|---|
| P01 ↔ P03 (supersede) | complement | 0.972 | supersession = consistent history, not negation |
| P01 ↔ P02 (approve A vs recommend B) | contradict | 0.750 | statement-level tension (authority ignored) |
| P03 ↔ its own justification (control) | complement | 0.871 | sanity |

Both non-trivial roles are *derived* (`role_source=tev1`) and distinguishable
from supplied edges (none supplied).

## Proposed changes (small, committed behind the existing design)

1. **t01 multi-doc** — repeatable `--doc P01… --doc P02… --doc P03…` with
   distinct `doc_id`s into one parsed set; t02–t10 already operate corpus-wide
   (cross-doc pairing is native and desirable here). ~15 lines + test.
2. **t01 front-matter** — parse `Document type:/Document ID:/Date:/Author:/
   Subject:` keys into structured node properties (`source_event_id`,
   `document_type`, `event_time_utc`, `actor_id/name`, `authority_scope`);
   keep banner + synthetic-note lines as content. Fixes the header mangling and
   gives structured provenance / event-vs-observation distinction. ~60 lines + tests.
3. **Freeze tooling** — `python -m tensegrity_pipeline.freeze --tag s001`:
   JSON+MD bundle with commit SHA, profile config (redacted secrets), ingestion
   order (P01→P02→P03), stage warnings/retries, graph counts (nodes by type,
   edges by level/role/role_source), and the UTC freeze timestamp. ~80 lines.

## Run procedure (when approved)

1. Fresh `tensegrity` DB (drop `tensegrity` DB; verify empty).
2. `python -m tensegrity_pipeline 01_parse --doc <P01> --doc <P02> --doc <P03>` (+ `--reset`)
3. `02_embed` → `03_insert_nodes` → `04_l15_edges` → `05_title_vectors` →
   `06_l14_edges` → `07_orphan_reentry` → `08_metabary` → `09_extend` → `10_index`
   (live ollama embed + tev1; tiny corpus — minutes).
4. `freeze --tag s001` → review bundle → declare frozen with UTC timestamp,
   then notify GovKM. Do not modify the graph afterward.

## Boundary notes to record in the freeze

- No operative-state resolution engine (out of scope; graph preserves the
  evidence, resolution is a query-era concern).
- Two-class role surface; supersession lands as complement on this corpus.
- Estimated effort for the three changes: ~2–4 h including tests; live run minutes.