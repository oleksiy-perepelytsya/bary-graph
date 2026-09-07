# Academic-dataset retrieval evaluation — methodology

Protocol for measuring whether the BaryGraph schema boosts retrieval quality over
flat text search on the academic term corpus. Scope: `barygraph_poc` ONLY (the
batch terms were ingested there via `scripts/ingest_batch.py` + re-run stages s04–s08;
this evaluation is independent of the multilingual `barygraph_all` build).

## Ground truth

`/workspace/papers_combined.parquet` — 6 use cases (`use_case_key`), each paper row
has `doi`, `triage_label` (positive | pass | negative | None), `problem_statement`,
`objective`, `terms_must_include`, `terms_exclude`, `title`, `abstract`.

- Labels are joined as `(doi, use_case_key)`. Only 6 DOIs appear in >1 use case.
- Relevant pool per problem = deduped DOIs with `triage_label == 'positive'`.
- `None` labels and DOIs with no row for the *queried* problem are "unlabeled":
  excluded from `·shown` denominators, counted as misses in `·strict`.

## Query set (`evaluation/queries.json`)

5 deterministic queries per problem (30 total): `problem_statement`, `objective`,
`terms_must_include` joined, and two fixed template rephrasings. `terms_exclude`
is not injected into queries; matching hits are counted in the `excl` column only.

## Retrieval arms (matched k ∈ 5, 10, 20, 50)

| arm | space | method |
|---|---|---|
| raw_bm25 | paper `title + abstract` (2692 unique DOIs) | BM25 (k1=1.2, b=0.75) |
| nodes_bm25 | word/sense nodes (gloss text) | BM25 (client-side, single gloss scan) |
| nodes_vec | node space | `$vectorSearch` (nomic-embed-text:v1.5, 768d) |
| edges_vec | BaryEdge + MetaBary space | `$vectorSearch` |
| graph_vec | nodes + edges, merged by score, top-k | `$vectorSearch` |

Query embeddings: one batched Ollama call for all 30 unique texts (same embedder
as the corpus). Published derivatives: PMID-style measure at 30 queries.

DOIs per hit resolve via the `doi_bridges` reverse index (`node_id -> DOIs`),
the same provenance the `include_dois` MCP path uses.

## Metrics

Per (problem, query, arm, k): hit-level P@k (strict and ·shown), DOI-level P@k,
R@k vs the problem pool, nDCG@k graded (positive=2, pass=1, negative=0),
distinct-DOI count, exclusion violations. Pooled per-query means; per-problem
5-query means; global positive-DOI coverage at k=50.

Decomposition: method effect = `nodes_vec − nodes_bm25` (space fixed);
schema effect = `graph_vec − nodes_vec` and `edges_vec − nodes_vec` (method fixed).

Tests over the 30 queries at k=20: McNemar (top-1 hit relevance, exact binomial)
and Wilcoxon signed-rank (hit-P@20), plus a null consideration for recall ceilings.

## Aggregation quality (`aggregation_quality.csv`)

Over `barygraph_poc` BaryEdge/MetaBary docs registered in `doi_bridges`
(the academic subgraph), per level (15→10):

- mean distinct-DOI richness per node — expect ↑ toward the root
- trivial-cluster share (single-DOI nodes) — expect ↓ toward the root
- z of observed richness vs a slot-permutation shuffle (per-doc DOI slot counts
  preserved, DOI identities reassigned at random within the level)
- mean pairwise cosine among {node, cm1, cm2} vectors (coherence)

Stratified by modal use-case where n ≥ 20.

## Caveats

- Provenance circularity: senses were extracted from these papers, so absolute
  positive-rates are not the claim; only the *relative* arm deltas and the
  method/schema decomposition are interpretable.
- Schema arms are recall-capped by per-problem coverage (76–97%, see summary);
  raw_bm25 is not.
- Edge/MB hits carry propagated DOIs (one hub DOI can appear in many hits) —
  hit-level metrics are the primary read; DOI-level is reported separately.

## Running

```
python -m scripts.eval.build_queries       # → evaluation/queries.json
python -m scripts.eval.run_eval_academic   # → evaluation/results/*.jsonl  (reads barygraph_poc)
python -m scripts.eval.score_eval_academic # → evaluation/results/academic_eval_metrics.csv + academic_summary.md
python -m scripts.eval.agg_quality         # → evaluation/results/aggregation_quality.csv
```

All entrypoints refuse to run unless `MONGO_DB == barygraph_poc`.