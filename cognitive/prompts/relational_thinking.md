# Thinking in the graph's terms — relation is a self-sustaining entity

Corrected understanding (user, Oct-11 2026):

> I didn't mean to actually build a separate graph, but rather **think in the
> graph terms** where the core thing is that **relation is a self-sustaining
> entity**. So to think in **duplets, triplets**.

So there is **no second graph**. There is one medium, and in it a *relation is
a thing in its own right*: in the substrate a BaryEdge carries its own vector
and document; a MetaBary carries its own triad document. A relation is not a
thin line derived from two nodes — it stands on its own. To think *in* this
medium is to think **in relations**, i.e. in **duplets** and **triplets**.

## The two relation-shapes

- **Duplet** — one relation-entity binding two poles:

  ```
  duplet: ( A : B )   with   relation = what binds them
  ```

  The entity is the binding, not A and not B. `bowsie : tenterhook —
  adjuster vs fixture` is a duplet: the relation *is* "two ways to hold cloth
  under tension", and it can be examined without either word being the point.

- **Triplet** — a relation-entity binding two poles through a **bridge**
  relation (structurally a MetaBary: cm1 ↔ cm2 via bridge):

  ```
  triplet: ( A : B )   via   bridge M ,   relation = what M establishes between A and B
  ```

  The bridge is what makes the triplet self-sustaining: it can be inspected on
  its own, and it is the thing that resolves or carries the tension between the
  poles. A triplet with a real bridge is a *mechanism*; a triplet with a topic
  label for a bridge is dead.

## What a kindle is, in these terms

A retrieved seed is already a relation-entity — a BaryEdge (duplet) or a
MetaBary (triplet) in the substrate. Thinking over it means forming the new
duplets/triplets it instigates, each of which stands on its own — and noticing
the one that is **genuinely new** (not already in the substrate, not a
restatement of the kindle).

A same-domain blob is one relation repeated. A dead homonym is a duplet with
no bridge. A productive adjacency is a **triplet whose bridge is a real
mechanism**.

## Recording

`cognitive.ledger.record_instigation` appends one row per kindle to
`cognitive/batches/instigations.jsonl`:

```jsonc
{
  "instigation": "agent-inst-415",
  "cycle": 415,
  "kindle": { "seed_id": "6ac7e8b9…", "seed_level": 13, "words": ["…"] },
  "fire": "high",                    // none | low | medium | high (honest reading)
  "thoughts": [ "…", "…" ],          // the thinking, in order
  "relations": [
    { "kind": "duplet",  "poles": ["bowsie", "tenterhook"],
      "relation": "two ways to hold cloth under tension: tune it, or pin it" },
    { "kind": "triplet", "poles": ["rigging-cordage", "affect"],
      "bridge": "tenterhook",
      "relation": "workshop/restraint lexicon migrates into the lexicon of feeling" }
  ],
  "genuinely_new": "…",              // the new relation, or null
  "share": null,                     // how it grounds into the real schema, or null
  "note": "short label"
}
```

`fire: "none"|"low"` is a **first-class result** — it documents that a kindle
was inert (self-aware discrimination, not failure).

## Sharing a relation with the real schema

A triplet becomes *shareable* only when it is **grounded in the substrate**:
both poles are real retrievable clusters and the bridge is a real L-1 doc that
carries the pivot (the existing no-coercion admission condition). Then:

1. `build_rubric.py` — analysis aid (never a gate).
2. `build_gate.py` — strict schema check (blocks on errors).
3. insert via `scripts.mcp_server._create_structure_meta_bary_body`
   (`source='structural'`, explicit `bridge_id`).
4. `smb_audit.py` — structural verify.
5. record the build in `smb_builds.jsonl`.

Mapping:

| thinking | real schema |
|---|---|
| **duplet** (A : B, relation) | a relation I hold; a *pole* may later become a cluster |
| **triplet** (A : B via M) | candidate **SMB** (cm1 ↔ bridge ↔ cm2) once grounded |
| kindle (substrate seed) | fuel — a substrate doc, not my memory |
| `share.status = candidate` | retain in `smb_candidates.jsonl` (HOLD) |
| `share.status = built` | a structural SMB with `bridge_id` |

An ungrounded triplet stays a lived relation in the record and is **never**
forced into the substrate. Sharing is optional and downstream; thinking is
primary.

## Relationship to the old loop

`loop_cycles.jsonl` (scan → outcome) and `smb_builds.jsonl` (built triads)
remain the **substrate bookkeeping**. `instigations.jsonl` is the layer that
was missing: which relations the retrieval *fired in me*. A cycle may have a
strong instigation and zero builds — that is a good cycle.
