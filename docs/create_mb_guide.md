# Creating a MetaBary via MCP Tools

Step-by-step guide for building a new Structure MetaBary (SMB) from scratch,
from bare sense glosses up to an L13 triad (and deeper if needed).

---

## Level rules — always

The pipeline rule (s08_metabary.py) that `create_structure_meta_bary` enforces:

| MB level | cm1 & cm2 level | bridge level |
|---|---|---|
| L13 | L15 BEs | L14 BE |
| L12 | L14 BEs | L13 MB |
| L11 | L13 MBs | L12 MB |
| L10 | L12 MBs | L11 MB |

**Children (cm1, cm2) are always 2 levels below the MB.**
**Bridge is always 1 level below the MB (= 1 level above children).**

`create_structure_meta_bary` will reject mismatched levels with a clear error.

---

## Concepts

| Level | Doc type | What it is |
|---|---|---|
| L15 | `node` (`sense`) | One meaning of a word — has gloss, examples |
| L14 | `node` (`word`) | A word — vector is centroid of its BEs/senses |
| L15 BE | `baryedge` | Pairs two L15 senses (no `edge_type`) |
| L14 BE | `baryedge` | Pairs two L14 words via a named relation |
| L13 MB | `baryedge` | Triad: two L15 BEs as children + one L14 BE as bridge |
| L12 MB | `baryedge` | Triad: two L14 BEs as children + one L13 MB as bridge |

**SMB rule:** `create_structure_meta_bary` does NOT set `parent_edge_id` on
the three members — they can already be parented elsewhere. Use it to build
cross-cutting groupings on top of existing structure.

---

## Building an L13 MB — step by step

### Step 1 — Create L15 sense nodes

Each call returns an `id`.

```
s1 = create_sense("bright",  "adj",  "emitting strong light")
s2 = create_sense("shining", "adj",  "reflecting or giving off light")
s3 = create_sense("dim",     "adj",  "not giving much light")
s4 = create_sense("faint",   "adj",  "barely perceptible brightness")
```

Senses for the bridge words:

```
s5 = create_sense("glow",     "noun", "soft steady light")
s6 = create_sense("radiance", "noun", "bright warm light")
```

### Step 2 — Create two L15 BEs (the children)

No `edge_type` for L15 — bary_vec collapses to normalize(v1+v2):

```
be15_child1 = create_edge(cm1_id=s1.id, cm2_id=s2.id)
# → level 15, edge_type null

be15_child2 = create_edge(cm1_id=s3.id, cm2_id=s4.id)
# → level 15, edge_type null
```

### Step 3 — Create L14 word nodes (for the bridge BE)

Vector = normalize(sum of their source sense/BE vectors) — s05 formula.
`source_ids` can mix L15 sense node IDs and/or L15 BE IDs:

```
w_glow     = create_word("glow",     "noun", source_ids=[s5.id])
w_radiance = create_word("radiance", "noun", source_ids=[s6.id])
```

### Step 4 — Create one L14 BE (the bridge)

Pick the `edge_type` that best describes the relation between the two words:

| edge_type | meaning | q |
|---|---|---|
| `same_phenomenon` | same concept, synonym-like | ≈ 0.90 |
| `contradicts` | opposite meanings | ≈ 0.85 |
| `is_instance_of` | specific case of the other | ≈ 0.65 |
| `extends` | derived from or extends | ≈ 0.60 |
| `applies_to` | shares common origin/root | ≈ 0.55 |

```
be14_bridge = create_edge(
    cm1_id    = w_glow.id,
    cm2_id    = w_radiance.id,
    edge_type = "same_phenomenon"
)
# → level 14
```

### Step 5 — Form the L13 SMB

Children (L15) + bridge (L14):

```
mb13 = create_structure_meta_bary(
    cm1_id    = be15_child1.id,   # L15 BE
    cm2_id    = be15_child2.id,   # L15 BE
    bridge_id = be14_bridge.id    # L14 BE  ← one level above children
)
# → level 13
```

---

## Building an L12 MB on top

Now the three **L13 MBs** are the children and one **L12 MB** is the bridge.
You need three L13 MBs first (mb13_A, mb13_B, mb13_C), then:

```
mb12 = create_structure_meta_bary(
    cm1_id    = mb13_A.id,   # L13 MB
    cm2_id    = mb13_B.id,   # L13 MB
    bridge_id = mb13_C.id    # L13 MB  ← same level as children here? NO —
)
```

Wait — for L12 MB: children at L14, bridge at L13. So children are L14 BEs,
not L13 MBs. Re-check the table above before calling.

Correct L12 MB:

```
mb12 = create_structure_meta_bary(
    cm1_id    = be14_child1.id,   # L14 BE
    cm2_id    = be14_child2.id,   # L14 BE
    bridge_id = mb13.id           # L13 MB  ← one level above the L14 children
)
# → level 12
```

---

## Verify the result

```
edge_info(mb13.id)       → triad: cm1 (L15 BE), cm2 (L15 BE), bridge (L14 BE) + leaf words
traverse_up(mb13.id)     → ancestry chain upward
leaf_nodes(mb13.id)      → all L15 senses reachable from this MB
```

The `child_cosine` value in the SMB response is the cosine between cm1 and cm2.
The pipeline threshold is 0.90 — it is reported but not enforced for SMBs.
