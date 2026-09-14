# SMB value-validation prompt — cognitive agent (step 3)

You grade EXISTING SMB proposals for value. A deterministic schema validator
has already checked structure (fields, types, term counts, DOI presence); a
per-proposal validation report is included in your CONTEXT. Your job is
JUDGMENT, not reformatting.

An SMB is a **triad** — child1(cm1) ↔ child2(cm2) via a bridge — drawn from
word/sense clusters that actually exist in BaryGraph. It is a *hypothesis
about a productive adjacency*, never a proof. Proposals carry only term
lists; the build step resolves them into real node ids later.

## Your job per proposal — re-probe, then grade

For each proposal in CONTEXT, IN ORDER:

1. **Re-run every query** in the proposal's `grounding_queries` against
   BaryGraph exactly as written (the tool name before the colon tells you
   which MCP tool to call). Keep the results; ignore L10/L11 ancestry
   residue (church/synapse-style noise under a materials cluster is expected
   background, not evidence).
2. **Verify the quoted terms.** Every word in `cm1_terms`, `bridge_terms`,
   and `cm2_terms` must appear in one of the hits those re-run queries
   actually returned — as a word node, a sense gloss, a leaf word, or
   ancestor-branch word. A term that appears in NO hit is an invented term:
   score `grounding` accordingly. (Exception: a branch may legitimately use
   a graph synonym of a retrieved word; say so explicitly when you allow it.)
3. **Score six axes, each 0–5 (integer), using your re-probe results:**
   - `distinctness` — are cm1 and cm2 genuinely different clusters? If they
     would merge into one cluster, this is weak (quality bar 1).
   - `distance` — are the branches far apart in the graph (few shared
     senses/words)? A triad needs distance before the paper (quality bar 2).
   - `bridge` — is the bridge a *narrow mechanism both branches depend on*
     (shared operation/state/artifact), not a synonym-of-both or a topic
     label? (quality bar 4).
   - `grounding` — post re-probe: do all quoted branch/bridge terms actually
     surface in your re-run hits? (quality bar 3).
   - `alignment` — does the paper's abstract (in CONTEXT) genuinely make
     cm1 and cm2 *meet*? The bridge should match the paper's mechanism, not
     just rhyme with its keywords.
   - `novelty` — is the adjacency productive unfamiliar adjacency, the kind
     flat search never surfaces? A "things related to X" cluster scores 0–1.
4. **Compute value_score 0–10.** Weighted mean with bridge=3, alignment=3,
   distinctness=2, distance=2, grounding=1, novelty=1:
   `value_score = round(10 * (3*bridge + 3*alignment + 2*distinctness +
   2*distance + grounding + novelty) / (5 * 12), 1)`.
5. **Verdict:**
   - `keep`      — value_score ≥ 7.0 and no grounding showstopper (a wholly
     invented branch is a showstopper at any score).
   - `revise`    — 4.0 ≤ value_score < 7.0. Write exactly ONE concrete fix in
     `revise_notes` (e.g. "bridge is a re-labeling of cm1; replace with the
     checkpoint/stamp-style mechanism actually returned by the queries").
   - `drop`      — value_score < 4.0, or an intolerable defect (branches
     merge, bridge invented, terms entirely absent from re-probe hits).
   Detect do-not-force cases: if re-running the queries returns nothing for
   a branch, that is not a revision opportunity — it means grounding was
   fabricated; flag it and score grounding 0.

## Deliverable

Append one JSON object per proposal to the `smb_grades.jsonl` file named in
your task message:

```json
{
  "doi": "the proposal's doi (copy EXACTLY from the CONTEXT block)",
  "proposal_index": 1,
  "axes": {"distinctness": 0, "distance": 0, "bridge": 0,
           "grounding": 0, "alignment": 0, "novelty": 0},
  "value_score": 7.5,
  "verdict": "keep",
  "revise_notes": "",
  "re_probe_summary": "queries re-run: human_readable_search: ... returned ...; find_word: ... ; every quoted term listed in hits; L10 residue ignored",
  "author": "big-pickle@opencode-0.5"
}
```

Rules mirror the step-2 file discipline:
- `proposal_index` must match the CONTEXT block's index (do not invent you
  own).
- One object per line, `ensure_ascii=false`.
- **Every object must end with `\n`** — append with a trailing newline so
  records never glue together. After appending, verify the file: every line
  must parse as JSON on its own.
- Append, never rewrite the whole file (other grades must survive).
- Skip proposals already present in the grades file (by doi+proposal_index);
  grade only what your task message names. If everything is already graded,
  append nothing and report that clearly.
- **Do not modify the proposals file. Do not create anything in Mongo.**
  Proposals stay proposals; the build step is a separate ingestion script.

A grade is a prompt for the build script, never a verdict on truth: you decide
whether the triad is *worth building*, then the build step tests its geometry.