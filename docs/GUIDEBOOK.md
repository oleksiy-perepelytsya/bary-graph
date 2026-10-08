# BaryGraph GuideBook

Conceptual design decisions and empirical observations for the bary-vector
effort. Entries are dated; observations cite the numbers as measured at the
time and where they live. Live pipeline state is *not* archived here — see the
pointers below.

**Live status pointers**
- Growth-round ledger (append-only, one record per round):
  `cognitive/batches/s08_growth_runs.jsonl`
- Chain driver journal (per-boundary log): `/tmp/chain_growth.md`
- Stage checkpoints: `pipeline_state/*.json` (gitignored, runtime state)
- Analysis probe scripts (ephemeral, in /tmp): `survey_l4l5.py`,
  `neigh_search.py`, `name_share.py`, `collect_round.py`, `chain_growth.sh`

---

## 1. Design decisions

### 1.1 Names & geonames as language substrate — 2026-10-08

**Context.** The s08b growth round extended the PoC graph to new depth (L5, L4).
Surveys showed the upper band of the graph is dominated by name clusters
(surnames and placenames — both `pos='name'` in kaikki-en): L9 83%, L7 100%,
L5 100%, L4 ~74% of MBs by sampled leaf census; the name world is largely
closed (80% of name-MBs attach upward into pure-name parents); and ~1/3 of
non-name MBs' upward attachments ride on name bridges. Retrieval probes over
thousands of calls surface name clusters only rarely. This raised the question
of pruning, tagging, or otherwise managing names.

**Position (decided).** Names are not noise and not a category to manage.
They are *language substrate* — the sedimented ground of the corpus — and we
accept them ontologically as such, exactly as one accepts soil: full of
everything from past states of matter — living cells, pure elements, molecules
and their wreckages. You do not prune soil, label it, or weigh it; you build
on it, and the graph grows from it.

Names are rigid designators: tokens that fix unique referents without
description ("defining unnamable stuff") — the instance axis of language,
complementary to the type axis of common nouns. Their meaning is
culture-localized ("specific to this region or branch of humankind"), and the
embedding geometry already encodes the language and culture rules of that
localization without any annotation — surname clusters separate by culture
(Germanic, Italic, Iberic…), school-suffix families (-ianism, -ian), and
grapheme families (amphi-, amino-, Ga-/Go-) all form spontaneously.

**Consequences.**
- No tagging pass, no `name_share` metadata, no culture classifier, no
  retrieval weighting or filtering. Nothing to mark, nothing to manage.
- The structural function stands as it is: names scaffold ~1/3 of the
  non-name world's upward movement and carry the deep cascade; that is
  substrate doing substrate's work.
- Retrieval surfacing names rarely is the substrate staying in the ground —
  by nature, not by defect. Queries shaped like proper-noun context will
  reach the substrate; generic prose will not; no mechanism needed.
- SMB admission remains purely value-based (no category-based rules). Name-
  rooted triads (Hegel↔Heideggerianism, Leibniz↔Kummerian, Schrödinger↔
  "Schrodinger equation") remain coordinates because they exist in the
  substrate, not because they are marked.
- Pipeline: s09 rescue sweep and s10 index proceed unchanged on the graph as
  it lies.

---

## 2. Observations

### 2.1 s08b growth round — 2026-10-08

Second, non-destructive s08 run (`--force`, existing structure, residue-only
children). Result: **+24,195 triads** — per level:

| L13 | L12 | L11 | L10 | L9 | L8 | L7 | L6 | L5 | L4 | L3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 13,499 | 512 | 5,146 | 32 | 2,556 | 1,313 | 570 | 474 | 47 | 46 | 0 |

- **L5 and L4 are brand-new levels** (run #1 stalled at L5 = 0). Graph depth
  now reaches L4. L3 pass hit 0 naturally = clean round termination.
- Residue accounting checks out exactly (e.g. L15 pool shrank by
  13,499 × 2 children = 26,998; new L13 MBs fed the L11 pass etc.).
- Record stored in the ledger with `delta_matches_log: true` (log milestone
  count == Mongo count delta per level).

### 2.2 L4/L5 triad structure survey — 2026-10-08

Top-5 by accumulated weight per level (47 L5 MBs, 46 L4 MBs), resolving
children (L+2), true bridge (L+1, via `parent_edge_id`), and leaf words
(`cm_leaf_words` BFS to node `properties.word`).

- **Two structurally different levels.** Every L5 *child* is a surname cluster
  and every L5 *bridge* is a run-1 L6 — orthographic/prefix clusters
  (amphi-/aph- taxonomy, ship-maritime, h-/ho- forms, amino-/benzo- chemistry,
  Ga-/Go-). L5 triads are coarse homology: two name clusters fused through a
  non-name cluster. At L4 the whole triangle turns surname-based (children AND
  bridge), often same-theme (German × German, Italian × Italian).
- **`connection_strength` (q) is bridge-weight dominance, not triad
  coherence.** q = Born rule `w3²/√(w1⁴+w2⁴+w3⁴)`; heavy L6 bridges
  (w up to 1.3077) vs light L7 children (w ≈ 0.13–0.19) force q ≈ 1.0.
  The right quality read is child-pair cosine: **0.97–0.997 everywhere**.
- **Famine floor persists, deeper and thinner.** L5 median q 0.255, L4 median
  q 0.151, L4 min q **0.008** — the deep tail is mostly near-zero-weight
  triads riding the surname hubs.
- **Hub effect extrapolates.** Guardado/Kort anchor multiple L4 triads;
  Echeverri/Manchu repeat inside L5 examples; Compton repeats ~14× as a
  sibling at the L15 BE level (star artifact, not structure).
- **Notable outlier:** L4 entry with *eye-dialect ‑in' forms*
  (crackah/crakin'/howlin' × blazin'/glowin'/screechin') bridged through a
  Manchu-carrying surname MB — the only non-name thematic triad at the top;
  candidate coordinate for the cognitive pipeline.

### 2.3 Surname neighbourhoods: Hegel, Leibniz, Schrödinger — 2026-10-08

Structural walk-up (word node → BE → MB lineage via `cm1_id`/`cm2_id` reverse
lookups; indexed). Note: vector-similarity search was unavailable (mongot
index is s10's job) — this is lineage adjacency, not vector neighbourhood.

- **Hegel** — two nodes. The *philosopher* L15 node's BE is orphaned (never
  consumed). The *Hegelism* L14 node climbs a real **philosophy-schools web**:
  L12 siblings Heideggerian(ism), bridge Althusserianism/Quineanism/
  Schopenhauerianism → L10 (q=0.92) siblings Derridean, Heideggerian,
  existentialized, philosopheme, bridge contemplate/studious.
- **Leibniz** — two parallel worlds. (a) *Heritage chain* via L15:
  Grimm/Grimmian → Schopenhauerian/Wertherism (L11 bridge) → Herbartian/
  Herderian/Rahnerian (L9 bridge) → run-1 famine floor → pathology bridge at
  L7 (actinopathy/myofibrosis/polymyositis…). (b) *Mathematics chain* via L14
  Leibnizian: Ludolphian/Eulerian (L12) → **Fuchs, Fuchsian, Kummer,
  Kummerian** (L10, q=0.997) — a genuine mathematicians cluster.
- **Schrödinger** — the graph recovers the physics: anglicized node
  cohabits a BE with Schroedinger → L13 MB whose siblings are literally
  "Schrodinger equation" and "Schrodinger's". The umlaut surname sense pairs
  with **Wien** (child-pair cos = 1.0000). Four placename senses: a ~14-BE
  star all echoing sibling "Compton"; plus a Robinson/Saint-Onge branch.
- **None reach L4–L6** — famous names top out at L7–L13, and their
  neighbourhoods are thematic (philosophy, maths, physics) rather than
  surname-hub noise. Famous = coherent; obscure = hub.
- Pattern: the **-ian/-ianism school-of-thought suffix family recurs four
  times** across these chains — the graph captures the orthographic+semantic
  family of European intellectual heritage.

### 2.4 Name/geo share census & scaffolding — 2026-10-08

`pos='name'` is the node-level pool for both surnames and geonames (geo is a
gloss-distinguishable subset — placename glosses). ~**15% of corpus nodes**:
276,469 of 1,785,126 at L15; 196,783 of 1,473,954 at L14.

MB census by sampled leaf type (cap 50 leaves):

| level | n MBs | pure-name | ≥50% name | no-name |
|---|---|---|---|---|
| L13 | 311k | 12.9% | 15.8% | 82.1% |
| L12 | 107k | 9.3% | 15.0% | 79.3% |
| L11 | 38k | 45.7% | 47.8% | 47.9% |
| L10 | 18k | 8.6% | 13.6% | 80.0% |
| L9 | 5.7k | 80.0% | 82.9% | 15.7% |
| L8 | 3.8k | 10.7% | 10.7% | 78.6% |
| L7 | 778 | 97.1% | 100% | 0% |
| L6 | 681 | 29.3% | 30.0% | 51.4% |
| L5 | 47 | 97.9% | 100% | 0% |
| L4 | 46 | 71.7% | ~74% | 23.9% |

Scaffolding tests (what consumed what, upward):
- Name MBs → parents: **80% pure-name** (244/306) — the name world is largely
  closed, it chains through itself.
- Non-name MBs (L12–L6, n=487) → attach upward: **64% no-name, 32% name,
  4% mixed** — ~1/3 of the non-name world's upward edges ride on name bridges.

Readout: the graph is two worlds — the belly levels (L13/L12/L10/L8, 78–82%
no-name, ~440k MBs) barely touch names; the low-volume upper band
(L9/L7/L5/L4) is name-monopolized and is where the name chains scaffold the
deep cascade. This observation is the empirical basis for decision 1.1.

### 2.5 Operations notes — 2026-10-08

- **Autonomous chain driver** (`/tmp/chain_growth.sh`): local bash driver
  that finishes the current growth round → collects+stores each round to the
  ledger → relaunches further rounds while yield ≥ 1,000 (max 8) → s09 → s10.
  Fully local (Mongo on localhost:27117), no network.
- **Bug found & fixed live**: `printf "s08%c" 99` yields "s089" (bash `%c`
  takes the first char of the arg, not the ASCII code). Module call was hard-
  coded correctly, so data was never at risk; log/pre files renamed to the
  proper `s08c` names (`mv` on an open fd is seamless) and the driver
  relaunched in resume mode with a proper round-name array.
- **Collector** (`/tmp/collect_round.py`): derives new-triads-per-level from
  the stage log, reads live Mongo counts, diffs against the pre-run snapshot,
  appends ONE JSONL record per round to `cognitive/batches/s08_growth_runs.jsonl`.
- **State at writing**: s08c in flight — L13 5,505 / L12 59 / L11 3,567 /
  L10 1 (residues at L12/L10 effectively dead, mirroring s08b); driver will
  auto-continue s08d → s09 → s10. Disk ~49G free, RAM ~8G available.

---

## 3. Open threads

- s09 rescue sweep and s10 mongot index pending on the current graph version.
- Once mongot is live: re-test the "names rarely surface in retrieval" claim
  quantitatively (name-shaped queries vs generic prose), and re-probe the
  Hegel/Leibniz/Schrödinger chains via actual vector neighbourhood.
- The eye-dialect L4 triad and the philosophy/maths name chains are candidate
  "coordinates to investigate" for the cognitive pipeline (value-based
  admission only, per policy).