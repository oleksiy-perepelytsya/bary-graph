# Academic-dataset retrieval evaluation — barygraph_poc

30 queries (6 problems x 5: statement, objective, must-terms join, 2 template rephrasings). k = 5, 10, 20, 50.

## Schema reach per problem (recall ceiling, from doi_bridges)

| problem | positive pool | with graph terms | coverage |
|---|---|---|---|
| carbon_capture | 143 | 109 | 76% |
| cement_binders | 166 | 144 | 87% |
| ner | 218 | 212 | 97% |
| soil_microbiome | 94 | 85 | 90% |
| solar_leo | 255 | 244 | 96% |
| tech_forecasting | 136 | 121 | 89% |

## Pooled per-query mean (30 queries), k in {5,10,20,50}

| arm | k | hitP | hitP·shown | doiP | doiP·shown | R@k | nDCG | doi# | excl |
|---|---|---|---|---|---|---|---|---|---|
| raw_bm25 | 5 | 63% | 70% | 63% | 70% | 1.98% | 0.897 | 5 | 0.0 |
| raw_bm25 | 10 | 60% | 70% | 60% | 70% | 3.72% | 0.875 | 10 | 0.0 |
| raw_bm25 | 20 | 58% | 67% | 58% | 67% | 7.04% | 0.859 | 20 | 0.0 |
| raw_bm25 | 50 | 57% | 65% | 57% | 65% | 16.83% | 0.853 | 50 | 0.0 |
| nodes_bm25 | 5 | 57% | 68% | 59% | 68% | 3.67% | 0.704 | 10 | 0.0 |
| nodes_bm25 | 10 | 50% | 62% | 51% | 60% | 11.37% | 0.706 | 37 | 0.0 |
| nodes_bm25 | 20 | 46% | 60% | 51% | 61% | 15.04% | 0.716 | 49 | 0.0 |
| nodes_bm25 | 50 | 44% | 62% | 53% | 63% | 22.67% | 0.742 | 71 | 0.0 |
| nodes_vec | 5 | 60% | 72% | 61% | 68% | 2.01% | 0.864 | 6 | 0.0 |
| nodes_vec | 10 | 60% | 73% | 59% | 68% | 5.52% | 0.853 | 16 | 0.0 |
| nodes_vec | 20 | 60% | 72% | 57% | 66% | 18.75% | 0.841 | 58 | 0.0 |
| nodes_vec | 50 | 56% | 68% | 54% | 64% | 33.70% | 0.840 | 109 | 0.0 |
| edges_vec | 5 | 81% | 88% | 60% | 72% | 3.27% | 0.939 | 9 | 0.0 |
| edges_vec | 10 | 78% | 84% | 59% | 68% | 6.40% | 0.929 | 19 | 0.0 |
| edges_vec | 20 | 78% | 84% | 56% | 66% | 12.76% | 0.921 | 40 | 0.0 |
| edges_vec | 50 | 75% | 83% | 50% | 63% | 23.17% | 0.918 | 77 | 0.0 |
| graph_vec | 5 | 81% | 87% | 59% | 72% | 2.65% | 0.946 | 8 | 0.0 |
| graph_vec | 10 | 74% | 83% | 59% | 70% | 4.85% | 0.933 | 14 | 0.0 |
| graph_vec | 20 | 72% | 81% | 57% | 66% | 10.94% | 0.923 | 34 | 0.0 |
| graph_vec | 50 | 72% | 81% | 52% | 64% | 21.55% | 0.911 | 70 | 0.0 |

## Per-problem @k=20 (hit-level precision, strict)

| problem | raw_bm25 | nodes_bm25 | nodes_vec | edges_vec | graph_vec | mΔ | sΔ |
|---|---|---|---|---|
| carbon_capture | 65% | 32% | 71% | 100% | 91% | +39 | +20 |
| cement_binders | 44% | 31% | 29% | 43% | 38% | -2 | +9 |
| ner | 57% | 67% | 82% | 96% | 94% | +15 | +12 |
| soil_microbiome | 31% | 22% | 38% | 43% | 42% | +16 | +4 |
| solar_leo | 88% | 79% | 86% | 98% | 91% | +7 | +5 |
| tech_forecasting | 62% | 43% | 53% | 89% | 79% | +10 | +26 |

## Per-problem pairwise comparison (n=5 queries each, @k=20)

| problem | raw_bm25 | nodes_bm25 | nodes_vec | edges_vec | graph_vec | g·rawΔ | W g·raw | W g·nb25 | nDCG g→raw |
|---|---|---|---|---|---|---|---|
| carbon_capture | 65% | 32% | 71% | 100% | 91% | +26 | 0.043 | 0.043 | +0.041 |
| cement_binders | 44% | 31% | 29% | 43% | 38% | -6 | 0.893 | 1.000 | +0.071 |
| ner | 57% | 67% | 82% | 96% | 94% | +37 | 0.043 | 0.043 | +0.134 |
| soil_microbiome | 31% | 22% | 38% | 43% | 42% | +11 | 1.000 | 0.080 | +0.126 |
| solar_leo | 88% | 79% | 86% | 98% | 91% | +3 | 0.345 | 0.043 | +0.003 |
| tech_forecasting | 62% | 43% | 53% | 89% | 79% | +17 | 0.225 | 0.043 | +0.006 |

## Method vs schema decomposition (pooled @k=20, hit-precision)

- method effect (BM25→vector, nodes space): **+14 pp**
- schema effect (nodes→graph, vector): **+13 pp**

## Global positive-DOI coverage at k=50 (distinct positives found per arm)

| arm | distinct positive DOIs | pool share (union of 6 pools) |
|---|---|---|
| raw_bm25 | 410 | 40.5% |
| nodes_bm25 | 462 | 45.7% |
| nodes_vec | 591 | 58.4% |
| edges_vec | 445 | 44.0% |
| graph_vec | 468 | 46.2% |

## Pairwise tests (n=30 queries, @k=20)

| pair | onlyA:onlyB | McNemar p | Wilcoxon p (hitP) |
|---|---|---|---|
| graph_vec vs raw_bm25 | 8:3 | 0.227 | 0.004 |
| graph_vec vs nodes_bm25 | 13:1 | 0.002 | 0.000 |
| edges_vec vs nodes_vec | 6:2 | 0.289 | 0.000 |
| graph_vec vs nodes_vec | 4:1 | 0.375 | 0.000 |

*shown = excludes unlabeled-for-problem hits/DOIs; strict counts them as misses. Recall is bounded by schema coverage above. pass labels grade 1.0 in nDCG only.*
