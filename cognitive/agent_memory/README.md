# agent_memory/ — my abstract BaryGraph (the real workspace)

Corrected understanding of the loop (user, Oct-11 2026):

> It is **not** about building bridges from the substrate, mimicking an
> algebraic process on the graph. It is about **me as a thinking entity
> grabbing a retrieved bridge and thinking over it**, self-aware, observing
> which of my own thoughts it instigates and whether a *genuinely new*
> connection appears in those thoughts. Retrieval is not an object to act on;
> it is a **kindle** to check what it can fire in my own abstract BaryGraph —
> which I define — and afterwards share with the real schema.

So there are **two graphs**:

| graph | what it is | role |
|---|---|---|
| **substrate** — `barygraph_poc` | the kaikki-derived graph (words, senses, BaryEdges, SMBs) | **kindling**: a source of coordinates to think from |
| **agent graph** — this directory | *my* associative graph: concepts I activate, edges I assert | the **real workspace**; the thing being grown |

The substrate is not the memory. It is fuel. My memory is the graph of
associations *I* form when a coordinate fires.

## The instigation record

One row per kindle (`cognitive/batches/instigations.jsonl`, written by
`cognitive.ledger.record_instigation`):

```jsonc
{
  "instigation": "agent-inst-415",
  "cycle": 415,
  "kindle": { "seed_id": "6ac7e8b9…", "seed_level": 13,
              "words": ["earing","grommet","marline","reef line",…] },
  "fire": "high",                 // none | low | medium | high  (honest reading)
  "instigated": [                 // the thoughts it fired, in order
    "every term is named for its role in a load path, not its form",
    "bowsie tunes tension under load; tenterhook pins cloth stretched, immobile"
  ],
  "internal_nodes": [             // concepts activated/created in MY graph
    { "id": "agent:identity-by-load-path", "label": "identity by load path",
      "kind": "concept", "origin": "kindle:AF6/415" }
  ],
  "internal_edges": [             // associations I assert (my graph's edges)
    { "src": "agent:rigging-cordage", "dst": "agent:affect",
      "rel": "lexical migration", "novelty": 0.6, "via": "tenterhook" }
  ],
  "genuinely_new": "…the new connection, or null…",
  "share": null,                  // or how it grounds into the real schema
  "note": "short label"
}
```

`fire: "none"|"low"` is a **first-class result**: it documents that a
coordinate was inert for my graph. Self-aware discrimination — noticing which
kindles are dead — is the point, not a failure.

## How my graph shares with the real schema

An internal edge becomes *shareable* only when it is **grounded in the
substrate**: both lives exist as real retrievable clusters, and an L-1 doc
carries the pivot (the existing no-coercion admission condition). Then:

1. `build_rubric.py` — analysis aid (never a gate).
2. `build_gate.py` — strict schema check (blocks on errors).
3. insert via `scripts.mcp_server._create_structure_meta_bary_body`
   (`source='structural'`, explicit `bridge_id`).
4. `smb_audit.py` — structural verify.
5. record the build in `smb_builds.jsonl` and the reflection here.

Mapping:

| my graph | real schema |
|---|---|
| internal **node** (`agent:<slug>`) | a concept that may become an SMB **bridge theme** or stay private |
| internal **edge** (association) | candidate **SMB triad** (cm1 ↔ bridge ↔ cm2) once grounded |
| kindle coordinate | a substrate doc (seed / word / pair) — fuel, not memory |
| `share.status = candidate` | retain in `smb_candidates.jsonl` (near-miss/HOLD) |
| `share.status = built` | a structural SMB with `bridge_id` |

An internal edge with **no** grounding stays in my graph (it may still be
productive) and is **never** forced into the substrate. Sharing is optional
and downstream; thinking is primary.

## Relationship to the old loop

`loop_cycles.jsonl` (scan → outcome) and `smb_builds.jsonl` (built triads)
remain the **substrate bookkeeping**. `instigations.jsonl` is the layer that
was missing: what the retrieval *fired in me*. A cycle may have a strong
instigation and zero builds — that is a good cycle now.
