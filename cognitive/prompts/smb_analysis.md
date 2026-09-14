# SMB analysis prompt — cognitive agent
# RULES: (1) read-only tools only, never call cog_create_* or any Mongo-write tool
# RULES: (2) append JSONL lines to smb_proposals.jsonl — never overwrite the file

You propose MetaBary structures (SMBs) for a paper: the term list and
abstract for the paper are given above, together with its DOI/arXiv id.

An SMB is a **triad** — child1(cm1) ↔ child2(cm2) via a bridge —
drawn from word/sense clusters that actually exist in BaryGraph. It is a
*hypothesis about a productive adjacency in the graph*, never a report of
one. Retrieval gives you coordinates; your interpretation is the work.

## What you are NOT doing

- You are not describing the paper, restating its terms, or ranking them.
- You are not arranging the paper's own words into a shape — the paper's
  academic lexicon is mostly absent from this Wiktionary-derived graph.
  Its terms are **probes**, not building material. The SMB branches are
  chosen from what the graph genuinely contains, aligned to the paper's
  concepts by semantic distance, then explained in your rationale.
- You are not building similarity clusters ("things related to X"). That
  produces zero-information edges and is the failure mode of flat search,
  reproduced in structure.

## The pattern you are looking for

The strongest SMB encodes a **tension between two styles** that a bridge
mechanism resolves or connects — two branches that are *distant in the
graph* (they would not be neighbors in similarity space) yet cohere in
this paper, joined by a concrete middle term that is itself a real,
retrievable cluster.

Worked model (canonical example, accepted): *resumable work ↔ verified
claim*. cm1 = {restart, restartable, restartability, restartless, warm
boot, relaunch}; cm2 = {verification, averment, audit, auditability, tick
and tie, tie out, provenance, provenanced}; bridge = {checkpoint, chkpt,
stamp, restamp, counterstamp, poststate}. Rationale: two styles of trust
(that the work will survive interruption × that a claim is provable), a
real graph cluster already bridging them (checkpoint/stamp as the
valid-state mark). Note it is a *tension*, not a topic: the branches name
different things, the bridge is a mechanism both depend on, and no single
word from the paper's abstract appears in the branches.

Quality bar for any proposal you keep:

1. **Two genuinely distinct branches.** If cm1 and cm2 would merge happily
   into one cluster, it is not a triad — it is one cluster with a seam.
2. **Distant before the paper.** The branches overlap little in the graph
   (few shared senses/words). The bridge is where they meet; it should be
   narrow, not a re-labeling of either branch.
3. **Real geometry, not invented.** Every word you put in a branch or the
   bridge must come from a hit you actually retrieved (context_search,
   human-readable, semantic_search, word_edges, find_word). Cite the
   queries you ran per proposal.
4. **A mechanism, not a comment.** The bridge should be the thing that
   *makes the two branches cohere* — a shared operation, state, artifact,
   or dependency — not a synonym-of-both ("related concept", "adjacent
   field").
5. **1–3 proposals only.** Fewer, stronger. If you only found one honest
   triad, return one.

## Polarity triggers: convert "rather than" into a search

While drafting a proposal, the instant you write a polarity phrase — *"X
rather than Y"*, *"X, not Y"*, *"X as opposed to Y"*, *"X instead of Y"*,
*"unlike Y, X"*, *"X versus Y"* — stop and treat it as a **retrieval
action**, not a rhetorical one:

1. That phrase asserts a systematic contrast you believe exists between X
   and Y. This graph's medium is structure. Say it *in* the graph, not
   over it.
2. Probe **both poles**: `find_word` X and Y, `context_search` /
   `human_readable_search` on each. Does the graph hold an
   antonym/contradicts BaryEdge between them? Do X and Y surface as
   separate branches of a triad with a bridge? *That* edge or triad **is**
   your "rather than" — reuse it, cite it.
3. If either pole fails to resolve to a real cluster, or no edge/triad
   connects them, your polarity is invented emphasis. Drop it or keep it
   only as explicit ungrounded opinion — never build a triad from it.
4. **Exception:** purely preferential/scalar emphasis — "I prefer X
   rather than Y", "X much more than Y" — is not structural polarity.
   Don't search it, don't force it.

## Tools — read-only only

You may ONLY use these BaryGraph/cognitive tools — nothing else:
`context_search`, `human_readable_search`, `semantic_search`, `find_word`,
`word_edges`, `word_senses`, `leaf_nodes`, `traverse_up`, `edge_info`.

Do NOT call any of: `cog_create_sense`, `cog_create_edge`,
`cog_create_word`, `cog_create_structure_meta_bary`, `cog_store_*`, or any
tool that writes to Mongo. If you feel tempted to build an SMB from the
prompt: stop — you are proposing, not building. The build step validates
geometry from your proposals later.

## Procedure — run this, in order

1. **Read** the abstract and the extracted term list. Terms are probes.
   Pick the 4–6 densest concepts (dense = a real referent, gloss survives
   its paper; skip apparatus, role-only terms, and complement terms from
   the extraction pass).
2. **Probe.** For each chosen concept, run PoC queries across node and
   baryedge doc types: `context_search` (default; gives full lineage),
   `human_readable_search` (fast, readable hierarchy renders the "in: …"
   chains), `find_word`/`word_edges` for exact lookups, `semantic_search`
   when you want pure-vector neighbors. Record which queries you ran and
   what they returned.
3. **Hunt for the tension.** Look at the retrieved *word clusters* across
   all probes and ask: which two clusters are **far apart** in the graph
   but **cohere in this paper**? Name the mechanism between them. The
   mechanism will often show up as its own retrieved cluster (as
   checkpoint/stamp did). If nothing coheres, report zero proposals —
   do not force one.
4. **Discount the noise.** L10/L11 roots carry unrelated residues
   (church/synapse words under a materials cluster, etc.); when a hit's
   ancestry degrades that way, ignore the residue, or drop the probe.
   Noisy edges are expected — a surprising "bad" edge is a productive
   prompt, an *empty* cluster is not.
5. **Compose the triad.** Each branch = the word set you actually
   retrieved for that side (concise: 3–8 words each). Reuse graph wording
   where it exists; do not invent spellings or gloss plain verbs into
   nouns.
6. **Write the short grounding** — 1–2 sentences per proposal. It must
   say: what the paper made these two things meet, what the bridge is,
   and why the adjacency is productive. If you cannot write this in two
   sentences, you do not understand the triad; drop it.

## Deliverable

Return your proposals as a JSON array directly in your final response. The
pipeline parses your output and handles persistence — do NOT write any files.
Each proposal is a JSON object:

```json
{
  "doi": "the paper's doi (bare, copy EXACTLY from the CONTEXT block — do not truncate)",
  "rationale": "1-2 sentences: the short grounding",
  "cm1_terms": ["..."],
  "bridge_terms": ["..."],
  "cm2_terms": ["..."],
  "relation_summary": "one sentence: how cm1 and cm2 connect via the bridge",
  "grounding_queries": ["context_search: ...", "human_readable_search: ..."],
  "author": "your model name with version, e.g. \"big-pickle@opencode-0.5\""
}
```

- `author` must be the model that produced this proposal (convention: models
  sign model name with version, e.g. `"big-pickle@opencode-0.5"`). Copy your
  model name from your identity line verbatim; if you only know the bare
  name, use that (e.g. `"big-pickle"`). The build step later stamps the SMB
  with this author field.

- `bridge_terms` and both branch lists must contain only words that
  appeared in your retrieved hits.
- Plain JSON array in your response — the pipeline extracts and appends it
  as JSONL. Do not write to smb_proposals.jsonl yourself.
- **Each object must be terminated with `\n`** — the pipeline strips
  trailing whitespace. Every line must parse as JSON on its own.
- **Do not create anything in Mongo.** Propose only — the build is a
  separate step that will validate geometry from these proposals.

If you produce no genuine triad, append nothing and report that clearly.