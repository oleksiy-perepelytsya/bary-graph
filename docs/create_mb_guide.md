# Creating a MetaBary via MCP Tools

Step-by-step guide for building a new Structure MetaBary (SMB) from scratch,
from bare sense glosses up to an L11 triad.

---

## Concepts to keep in mind

| Level | Doc type | What it is |
|---|---|---|
| L15 | `node` (`sense`) | One meaning of a word — has gloss, examples |
| L14 | `node` (`word`) | A word across all its senses — vector is centroid of its BEs/senses |
| L15 BE | `baryedge` | Pairs two L15 senses (no `edge_type`) |
| L14 BE | `baryedge` | Pairs two L14 words via a named relation |
| L13 MB | `baryedge` | Triad: two L14 BEs as children + one L14 BE as bridge |
| L12 MB | `baryedge` | Triad: two L13 MBs as children + one L13 MB as bridge |
| … | | Same pattern up |

**SMB rule:** `create_structure_meta_bary` does NOT set `parent_edge_id` on the
three members — they can already be parented. Use it to build cross-cutting
groupings on top of existing structure.

---

## Step 1 — Find or create L15 sense nodes

If the senses already exist, find them:

```
find_word("luminous")          → returns word node id + sense ids
word_senses("luminous")        → lists gloss for each sense
```

If you need a new sense that is not in the graph:

```
create_sense(
  word     = "luminous",
  pos      = "adj",
  gloss    = "emitting or reflecting light brightly",
  examples = ["a luminous full moon"],   # optional, up to 2
  tags     = ["usually"],                # optional
  topics   = ["physics"]                 # optional
)
→ { "id": "<sense_oid>" }
```

Create as many senses as you need. Each call returns an `id`.

---

## Step 2 — Create L15 BaryEdges (sense pairs)

Pair two sense nodes. No `edge_type` for L15:

```
create_edge(cm1_id="<sense_A_id>", cm2_id="<sense_B_id>")
→ { "id": "<be_L15_id>", "level": 15, "edge_type": null }
```

You need **at least three L14 BEs** to form one L13 MB (two children + one bridge).
Each L14 BE needs two L14 word nodes, and each word node needs at least one L15 BE.
So the minimum graph for one MB is:

```
sense_A ──BE1── sense_B   ← bridge BE's senses
sense_C ──BE2── sense_D   ← child1 BE's senses
sense_E ──BE3── sense_F   ← child2 BE's senses
```

---

## Step 3 — Create L14 word nodes

One word node per word, using the L15 sense/BE ids you just created:

```
create_word(
  word       = "luminous",
  pos        = "adj",
  source_ids = ["<sense_oid>", "<be_L15_id>"],  # sense nodes and/or L15 BEs for this word
  ipa        = "ˈluːmɪnəs",    # optional
  etymology  = "Latin luminosus"  # optional
)
→ { "id": "<word_oid>" }
```

`source_ids` can mix L15 sense node IDs and L15 BE IDs — the word vector is
`normalize(sum of their vectors)`, same formula as `s05_word_vectors.py`.

---

## Step 4 — Create L14 BaryEdges (word pairs)

Pick an `edge_type` that fits the relationship:

| edge_type | meaning |
|---|---|
| `same_phenomenon` | same concept, synonym-like (q ≈ 0.90) |
| `contradicts` | opposite meanings (q ≈ 0.85) |
| `is_instance_of` | one is a specific case of the other (q ≈ 0.65) |
| `extends` | derived from or extends (q ≈ 0.60) |
| `applies_to` | shares common origin/root (q ≈ 0.55) |

```
create_edge(
  cm1_id    = "<word_X_id>",
  cm2_id    = "<word_Y_id>",
  edge_type = "same_phenomenon"
)
→ { "id": "<be_L14_id>", "level": 14, "q": 0.9 }
```

Build three such BEs: one for each role (child1, child2, bridge).

---

## Step 5 — Form the Structure MetaBary

```
create_structure_meta_bary(
  cm1_id    = "<be_L14_child1_id>",
  cm2_id    = "<be_L14_child2_id>",
  bridge_id = "<be_L14_bridge_id>"
)
→ { "id": "<mb_L13_id>", "level": 13 }
```

All three IDs must be distinct and at the same level (L14).
The MB vector is computed via the Born-rule formula:
`q_mb = q_bridge² / sqrt(q1⁴ + q2⁴ + q_bridge⁴)`
`mb_vec = normalize(q_mb·v_child1 + q_mb·v_child2 + (1−q_mb)·v_bridge)`

---

## Step 6 — Go deeper (optional)

To build an L12 MB on top, form three L13 MBs first (this step or from existing
graph), then call `create_structure_meta_bary` again with the three L13 MB ids.
The level validation ensures CMs are always at the same level.

---

## Verify the result

```
edge_info("<mb_L13_id>")      → triad: cm1, cm2, bridge + leaf words for each
traverse_up("<mb_L13_id>")    → full ancestry chain upward
leaf_nodes("<mb_L13_id>")     → all L15 senses reachable from this MB
```

---

## Minimal example — one L13 MB

```
# Six senses
s1 = create_sense("bright",   "adj", "emitting strong light")
s2 = create_sense("shining",  "adj", "reflecting or giving off light")
s3 = create_sense("dim",      "adj", "not giving much light")
s4 = create_sense("faint",    "adj", "barely perceptible brightness")
s5 = create_sense("glow",     "noun", "soft steady light")
s6 = create_sense("radiance", "noun", "bright warm light")

# Three L15 BEs
be15_bridge = create_edge(s1.id, s2.id)          # bridge senses
be15_child1 = create_edge(s3.id, s4.id)          # child1 senses
be15_child2 = create_edge(s5.id, s6.id)          # child2 senses

# Three L14 word nodes
w_bright   = create_word("bright",   "adj",  [s1.id, be15_bridge.id])
w_shining  = create_word("shining",  "adj",  [s2.id, be15_bridge.id])
w_dim      = create_word("dim",      "adj",  [s3.id, be15_child1.id])
w_faint    = create_word("faint",    "adj",  [s4.id, be15_child1.id])
w_glow     = create_word("glow",     "noun", [s5.id, be15_child2.id])
w_radiance = create_word("radiance", "noun", [s6.id, be15_child2.id])

# Three L14 BEs
be14_bridge = create_edge(w_bright.id,  w_shining.id, edge_type="same_phenomenon")
be14_child1 = create_edge(w_dim.id,    w_faint.id,   edge_type="same_phenomenon")
be14_child2 = create_edge(w_glow.id,   w_radiance.id, edge_type="same_phenomenon")

# L13 SMB
mb = create_structure_meta_bary(be14_child1.id, be14_child2.id, be14_bridge.id)

# Inspect
edge_info(mb.id)
```
