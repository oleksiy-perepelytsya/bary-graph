# cement_binders anomaly — why the graph arms went negative there

Pooled, the schema effect (graph_vec over flat baselines) is +13–14 pp on hit-precision
@k=20 and strongly significant (n=30: Wilcoxon p<0.005, McNemar 13:1 p=0.002).
`cement_binders` is the only problem where the pooled graph arms land at or below
BM25 (graph 38% vs raw 44% @20, Δ = −6 pp). This note dissects that case.

## Per-query strict hitP @k=20 (from `results/*.jsonl`)

| cement query | raw_bm25 | nodes_bm25 | nodes_vec | edges_vec | graph_vec |
|---|---|---|---|---|---|
| statement — "Reduce CO2 from clinker in structural concrete." | 20% | 35% | 55% | **80%** | 65% |
| rephrase_2 — "Research on: …" | 25% | 35% | 50% | **80%** | 70% |
| objective — "alternative binder chemistries at TRL 4–8 …" | 55% | 25% | 15% | 25% | 30% |
| rephrase_1 — objective rephrase | 45% | 15% | 10% | 20% | 15% |
| must_terms — "geopolymer, calcined clay, LC3, alkali-activated" | **75%** | 45% | 15% | 10% | 10% |

Two distinct regimes, not one flat failure.

## Mechanism 1 — lexical queries hand BM25 a free win on a screened pool

The `must_terms` and `objective` queries are jargon strings ("LC3", "TRL",
"calcined clay", "geopolymer") that appear verbatim in the abstracts, and the
cement positive pool was built by **keyword screening** with exactly those terms.
Raw BM25 therefore aligns with the label definition by construction: must_terms →
75% hitP, 19/20 labeled hits at k=20 (15 positive).

The embedded arms do the opposite on that query: 22–34 *unlabeled* DOIs occupy the
k=20 slots — conceptually cement-adjacent papers rather than the string-matched
positives. A comma-joined term list embeds as a centroid over four disjoint
concepts; its nearest edge/sense vectors match each term only weakly, so the
surface is generic cement work. Semantic lists are the weak side of embeddings;
lexical conjunctions are exactly BM25's strength.

## Mechanism 2 — cement carries the largest uncharted zone

~323 of 690 cement rows are `triage_label: None` (~47%, highest of the six
problems). Vec-space ranks these cement-adjacent, unlabeled papers naturally, and
every one counts as a *strict* miss. Raw BM25 is largely insulated because it only
surfaces strings the screening already tagged. The `·shown` variant (excluding
unlabeled) recovers much of the graph arms' apparent loss.

## The opposite regime — thematic queries go the other way

On the two statement-style queries, edges_vec = 80%/80% vs raw_bm25 = 20%/25% —
the single largest schema win in the entire suite (+55–60 pp). Embedding a
*sentence* lands near the alternative-binder concept manifold (geopolymer/LC3
senses), surfacing positives that the string matcher misses because the abstract
wording differs (e.g. "limestone calcined clay cement" vs "reducing clinker's
embodied carbon"). nDCG for cement remains **+0.071** in the graph's favor
(graded metric credits `pass` and is insensitive to strict-miss None rows).

Verdict: the −6 pp is a net of a labeling/query-register skew (−) against strong
thematic retrieval (+), not a schema failure. Same space scores 80% and 10%
depending purely on whether the query is a sentence or a term list.

## Overall efficiency estimate (pooled, n=30)

- **Schema = precision multiplier on thematic retrieval.** graph_vec hitP 72/81%
  @k=20/5 vs raw_bm25 58/63% and nodes_bm25 46/57% → roughly halves the error rate
  at top-20. Wilcoxon p<0.005; McNemar 13:1 (p=0.002) on top-1 relevance.
- **nDCG:** graph_vec .923 vs raw_bm25 .859 vs nodes_bm25 .716 (+7 pp over raw,
  +21 pp over flat nodes).
- **Rank stability:** edges_vec holds 81/78/75% across k=5/20/50 — flat baselines
  decay (63→57, 57→44). The merged graph trades trivial precision for robustness.
- **Cost = recall breadth:** graph_vec surfaces 468 distinct positives @k=50
  vs nodes_vec 591 (raw_bm25 410). Node-vector space is the recall play; graph is
  the per-slot-efficiency play.
- **Per-domain rule of thumb:** biggest gains on topic-built pools with technical
  vocab (ner +37, carbon_capture +26, tech_forecasting +17); ~null-to-negative where
  the positives were keyword-screened with the query strings themselves
  (cement −6, solar_leo +3) — an eval-construction bias, not a schema weakness.

## Open options (not run)

1. Split the 30-query suite into *thematic* (statement/rephrase_2) vs *lexical*
   (objective/must_terms/rephrase_1) families and report the schema effect per
   family across all six problems — directly tests the register hypothesis.
2. Add a sentence-form must-terms variant to the suite and re-run cement.
3. Sensitivity band: treat `None`-labeled cement DOIs as `pass` (not strict miss)
   to bracket worst/best case per problem.