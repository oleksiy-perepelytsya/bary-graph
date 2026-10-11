# Native loop analysis method — big-pickle

The analysis method the native SMB loop follows. The loop's tools scan the
graph; THIS decides what a scan hit *means* before anything is built or
skipped. Read it at the start of every batch, and again whenever a batch is
coming up all-similarity (that is usually a method failure, not a graph
failure).

## The pattern you are looking for

An SMB is a **triad** — cm1 ↔ cm2 via a bridge — and the only thing worth
building is a **tension between two styles** that a middle mechanism
resolves or connects:

- two branches that are **distant in the graph** (low child_cosine; they
  would NOT be vector neighbours),
- yet **cohere through a concrete middle term** that is itself a real,
  retrievable cluster,
- and the bridge is a **mechanism both branches depend on** — a shared
  operation, state, artifact, or dependency — not a synonym-of-both and not
  a topic label.

"One word, two lives" (the loop's productive signal) is the *pivot* that
makes the tension searchable: a word doing real double duty across two
distant clusters. The build requires that **both lives actually exist in
the graph** (that is the no-coercion admission condition) and that an L-1
bridge doc literally carries the pivot.

## Worked example (native, accepted, audit-verified)

*Lawn* (SMB `6acb626c…`, cycle 406):

- cm1 = grass-ground life (L15 edge `6ac677b3…`): {lawn, ground, grass,
  mown, agar culture} — the open-space/glade sense sits even further out
  (cos 0.27 to the linen life).
- cm2 = thin-linen life (L15 edge `6ac654f4…`): {lawn, thin linen, cotton
  fabric, bishop's sleeves}.
- bridge = the **L14 word node "lawn" itself** (`6ac55817…`) — literally
  carrying the pivot, both lives are its polysemous senses.
- child_cosine 0.42; both lives exist in the graph; GATE PASS before
  insert; audit ALL CONSISTENT after. A material word and a pastoral word
  meeting in one spelling — productive unfamiliar adjacency.

## Worked example (canonical, older pipeline, still the gold standard)

*Resumable work ↔ verified claim*: cm1 = {restart, restartable, warm boot,
relaunch}; cm2 = {verification, audit, auditability, provenance, tie out};
bridge = {checkpoint, chkpt, stamp, counterstamp, poststate}. Two styles of
trust (that work will survive interruption × that a claim is provable),
joined by the valid-state mark both depend on. No word from the source text
appears in the branches; the branches name different things; the bridge is
a mechanism, not a label.

## Anti-patterns (what most scan hits actually are)

1. **Same-domain blob** — every neighbour is the same style (bad-boy
   cluster, coven/fellowship/karass, ACE2 variants, doping agencies). The
   shared words are the cluster's own texture. Outcome: `similarity-cluster`
   — record, skip, do NOT force a triad. A blob that "should" be a triad is
   one cluster with a seam, not two lives.
2. **Component-word coincidence** — the "shared word" is a single component
   of a compound (`southeasterly`, `bad name`) and the neighbour is just
   that word's own node. Outcome: `no-adjacency`.
3. **Two lives, no bridge** — real distant lives (bandicoot: marsupial ↔
   garden-theft verb, cos 0.42) but **no L-1 doc carries the pivot**. Per
   the admission rule this is a **HOLD**: keep it as a candidate in
   `smb_candidates.jsonl` with its grounding probes, do not build, do not
   force a bridge into existence.

## Polarity triggers — convert "rather than" into a search

The instant you write a polarity phrase — *"X rather than Y"*, *"X, not
Y"*, *"X as opposed to Y"*, *"X versus Y"* — stop and treat it as a
**retrieval action**, not a rhetorical one:

1. That phrase asserts a systematic contrast. This graph's medium is
   structure — say it *in* the graph, not over it.
2. Probe **both poles** (`find_word`/`word_edges` on each; `find_bridge.py`
   on the pivot). Does the graph hold them as separate branches of a triad
   with a real bridge? That triad IS your "rather than" — cite it.
3. If either pole fails to resolve to a real cluster, or no bridge connects
   them, the polarity is invented emphasis. Drop it — never build a triad
   from it.
4. Purely scalar/preferential emphasis ("I prefer X rather than Y") is NOT
   structural polarity. Don't search it, don't force it.

## Quality bar (apply to every build decision)

1. **Two genuinely distinct branches.** If cm1 and cm2 would merge happily
   into one cluster, it is not a triad — it is one cluster with a seam.
2. **Distant before your analysis.** Little overlap in the graph (low
   cosine, few shared senses). The bridge is where they meet; it must be
   narrow, not a re-labelling of either branch.
3. **Real geometry, not invented.** Every word in a branch or the bridge
   must come from a hit you actually retrieved. Cite the probes you ran.
   Never gloss plain verbs into nouns; reuse graph wording.
4. **A mechanism, not a comment.** The bridge is the thing that *makes the
   two branches cohere* — shared operation/state/artifact/dependency. A
   bridge that is a synonym-of-both or a topic label is dead output.
5. **Fewer, stronger.** 1 robust triad beats 3 forced ones. If only one
   honest triad exists, build one (or zero).

## Noise rule

L10/L11 roots carry unrelated residue (church/synapse words surfacing under
a materials cluster, etc.). When a hit's ancestry degrades that way, ignore
the residue or drop the probe. Noisy edges are expected — a surprising "bad"
edge can be a productive prompt; an *empty* or *generic* cluster is not.

## Cycle procedure (the loop, with analysis in the right places)

> **The loop is a thinking loop, not a build loop.** Retrieval is a *kindle*;
> the primary act is introspection (what does this fire in my own abstract
> BaryGraph?), recorded in `instigations.jsonl`. Building in the substrate is a
> possible, optional, downstream *share*, not the goal. A cycle with a strong
> instigation and zero builds is a good cycle. See `agent_memory/README.md`.

1. **Seed** — random L12/L13 draw (`loop_pivot.py`). Do not prejudge the
   seed; it is a coordinate.
2. **Scan** — shared-leaf neighbours ascending cosine.
3. **Thin by judgment, not by filter.** Read the hits: blob? coincidence?
   real two-lives? Skip blobs *as blobs* (recorded, not committed); carry
   forward only the ≤1–2 threads with genuine two-lives potential. This is
   where the oracle may help as a **choice aid at a genuine fork** (which
   thread to pursue) — never as a validator.
4. **candidate? → probe deeply.** `find_bridge.py` on the pivot; leaf-word
   checks; both lives must exist in the graph. Record grounding probes.
5. **Analyze** — `build_rubric.py` (one-pass: distance/distinctness/
   mechanism/grounding evidence + scores). The rubric is a decision aid, an
   analysis-quality check, **never a gate**.
6. **Build** — `build_gate.py` (strict schema: levels L/L-1/L-2, refs
   resolve, vectors, distinct ids). GATE FAIL ⇒ do not insert, record.
   Value judgment is the sole admission criterion — the rubric informs it,
   the gate enforces schema, neither decides value.
7. **Verify + record** — `smb_audit.py` after insert; then record the cycle
   in `loop_cycles.jsonl` with `probes_run` + a 1–2 sentence `rationale`
   (including for skips — a skip without a reason is noise), and the build
   in `smb_builds.jsonl` with the rubric scores in `analysis`.

Standing direction: prefer **multi-hop chains** (two distant lives joined
by a real mechanism) over animal↔machine mining. No SMB coercion: a
proposal is a coordinate to investigate; only value judgment admits a build.
Above all, do not reduce the loop to an algebraic build on the substrate — it
is the agent thinking, with the graph as kindling.

## Calibration warning (Oct-11) — what the substrate metric actually ranks

The substrate's "one word, two lives" metric (min sense cosine, `word_pivot.py`)
turns out to rank **dead homonymy** at least as eagerly as the productive
common-domain two-lives: `shock` 0.15 (shaggy lapdog vs geological deformation),
`strain` 0.166 (a deer's track vs to stretch), `taut` 0.171 (nautical discipline
vs a wine term) score lower than `lawn` 0.205 (grass/linen). So low min_inter is
**not** the productive signal by itself; the productive signal is *metaphorical
migration between common domains*, which needs a commonness/domain judgement the
metric lacks. Recent low-yield batches were diagnosed as exactly this, not bad
luck. Do not let a low number stand in for a thought.