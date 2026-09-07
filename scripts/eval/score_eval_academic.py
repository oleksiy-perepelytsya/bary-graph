"""Score the academic eval arms against papers_combined human triage labels.

Metrics per (problem, query, arm, k) + pooled summaries, McNemar tests,
and the method/schema decomposition table. Pure stdlib + numpy (no pandas).
Writes academic_eval_metrics.csv and academic_summary.md under evaluation/results/.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from lib.config import Settings
from lib.db import get_client
from scripts.eval.academic_common import (
    K_LIST,
    RESULTS_DIR,
    label_index,
    load_rows,
    norm_doi,
    positive_pools,
    require_poc,
)

ARMS = ["raw_bm25", "nodes_bm25", "nodes_vec", "edges_vec", "graph_vec"]


def ndcg(gains: list[float]) -> float:
    if not gains:
        return 0.0
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gains, reverse=True)))
    return dcg / idcg if idcg else 0.0


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(x for x in pmf if x <= obs + 1e-15))


def wilcoxon_p(diffs: list[float]) -> float:
    d = np.asarray([x for x in diffs if x != 0.0], dtype=float)
    if len(d) < 5:
        return 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = float(np.sum(ranks * (d > 0)))
    n = len(d)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    z = (w - mu) / math.sqrt(var)
    return 2.0 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def main() -> None:
    settings = Settings.load()
    require_poc(settings)
    rows = load_rows()
    labels = label_index(rows)
    pools = positive_pools(rows)
    doi_uc: dict[str, set] = defaultdict(set)
    for r in rows:
        d = norm_doi(r.get("doi"))
        if d:
            doi_uc[d].add(r["use_case_key"])

    client = get_client(settings)
    bridge = client[settings.mongo_db][settings.mongo_doi_bridges_collection]
    graph_dois = {norm_doi(d["_id"]) for d in bridge.find({}, {"_id": 1})}
    graph_dois.discard(None)

    suite = json.loads(RESULTS_DIR.parent.joinpath("queries.json").read_text())
    excludes = {p["problem"]: [t.lower() for t in p["terms_exclude"]] for p in suite}

    arm_recs: dict[str, dict[tuple, list[dict]]] = {}
    for arm in ARMS:
        recs = [json.loads(line) for line in open(RESULTS_DIR / f"{arm}.jsonl")]
        arm_recs[arm] = {(r["problem"], r["query_id"]): r["hits"] for r in recs}

    def eval_query(problem: str, hits: list[dict], k: int) -> dict:
        top = hits[:k]
        pool = pools.get(problem, set())
        distinct: set[str] = set()
        excl = 0
        for h in top:
            distinct.update(h.get("dois") or [])
            w = (h.get("word") or "").lower()
            if any(e and e in w for e in excludes.get(problem, [])):
                excl += 1
        n = len(distinct)
        pos = sum(1 for d in distinct if labels.get((d, problem)) == "positive")
        lab = [d for d in distinct if (d, problem) in labels]
        n_pass = sum(1 for d in lab if labels.get((d, problem)) == "pass")

        hits_pos = hits_labeled = 0
        gains: list[float] = []
        for h in top:
            ds = h.get("dois") or []
            hp = any(labels.get((d, problem)) == "positive" for d in ds)
            hl = any((d, problem) in labels for d in ds)
            hits_pos += hp
            hits_labeled += hl
            g = 2.0 if hp else (1.0 if any(labels.get((d, problem)) == "pass" for d in ds) else 0.0)
            gains.append(g)

        return {
            "distinct_dois": n,
            "labeled_dois": len(lab),
            "pos_dois": pos,
            "pass_dois": n_pass,
            "doi_prec_shown": pos / len(lab) if lab else 0.0,
            "doi_prec_strict": pos / n if n else 0.0,
            "hit_prec_shown": hits_pos / hits_labeled if hits_labeled else 0.0,
            "hit_prec_strict": hits_pos / k,
            "recall": len(distinct & pool) / len(pool) if pool else 0.0,
            "ndcg": ndcg(gains),
            "exclusions": excl,
        }

    metric_cols = [
        "hit_prec_strict",
        "hit_prec_shown",
        "doi_prec_strict",
        "doi_prec_shown",
        "recall",
        "ndcg",
        "distinct_dois",
        "exclusions",
    ]
    rows_out = []
    for arm in ARMS:
        recs = arm_recs[arm]
        for (problem, qid), hits in recs.items():
            for k in K_LIST:
                m = eval_query(problem, hits, k)
                rows_out.append({"problem": problem, "query_id": qid, "arm": arm, "k": k, **m})

    csv_path = RESULTS_DIR / "academic_eval_metrics.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(["problem", "query_id", "arm", "k"] + metric_cols) + "\n")
        for r in rows_out:
            f.write(
                ",".join(
                    [str(r[c]) for c in ["problem", "query_id", "arm", "k"]]
                    + [f"{r[c]:.6f}" for c in metric_cols]
                )
                + "\n"
            )

    by = defaultdict(list)
    for r in rows_out:
        by[(r["arm"], r["k"])].append(r)
    pooled = {
        key: {c: float(np.mean([x[c] for x in vals])) for c in metric_cols}
        for key, vals in by.items()
    }
    prob_means: dict[tuple, dict] = {}
    by_prob = defaultdict(list)
    for r in rows_out:
        by_prob[(r["problem"], r["arm"], r["k"])].append(r)
    for key, vals in by_prob.items():
        prob_means[key] = {c: float(np.mean([x[c] for x in vals])) for c in metric_cols}
        prob_means[key]["problem"] = key[0]
        prob_means[key]["arm"] = key[1]
        prob_means[key]["k"] = key[2]

    lines: list[str] = []
    lines.append("# Academic-dataset retrieval evaluation — barygraph_poc")
    lines.append("")
    lines.append(
        "30 queries (6 problems x 5: statement, objective, must-terms join, "
        "2 template rephrasings). k = " + ", ".join(map(str, K_LIST)) + "."
    )
    lines.append("")

    lines.append("## Schema reach per problem (recall ceiling, from doi_bridges)")
    lines.append("")
    lines.append("| problem | positive pool | with graph terms | coverage |")
    lines.append("|---|---|---|---|")
    for uc, pool in sorted(pools.items()):
        cov = len(pool & graph_dois)
        lines.append(f"| {uc} | {len(pool)} | {cov} | {100 * cov / len(pool):.0f}% |")

    lines.append("")
    lines.append("## Pooled per-query mean (30 queries), k in {5,10,20,50}")
    lines.append("")
    lines.append("| arm | k | hitP | hitP·shown | doiP | doiP·shown | R@k | nDCG | doi# | excl |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for k in K_LIST:
            r = pooled[(arm, k)]
            lines.append(
                f"| {arm} | {k} | {r['hit_prec_strict'] * 100:.0f}% | "
                f"{r['hit_prec_shown'] * 100:.0f}% | {r['doi_prec_strict'] * 100:.0f}% | "
                f"{r['doi_prec_shown'] * 100:.0f}% | {r['recall'] * 100:.2f}% | "
                f"{r['ndcg']:.3f} | {r['distinct_dois']:.0f} | {r['exclusions']:.1f} |"
            )

    lines.append("")
    lines.append("## Per-problem @k=20 (hit-level precision, strict)")
    lines.append("")
    lines.append("| problem | " + " | ".join(ARMS) + " | m\u0394 | s\u0394 |")
    lines.append("|---|---|---|---|---|")
    for problem in sorted(pools):
        vals = {}
        for arm in ARMS:
            vals[arm] = prob_means[(problem, arm, 20)]["hit_prec_strict"] * 100
        m_d = vals["nodes_vec"] - vals["nodes_bm25"]
        s_d = vals["graph_vec"] - vals["nodes_vec"]
        lines.append(
            f"| {problem} | "
            + " | ".join(f"{vals[a]:.0f}%" for a in ARMS)
            + f" | {m_d:+.0f} | {s_d:+.0f} |"
        )

    lines.append("")
    lines.append("## Per-problem pairwise comparison (n=5 queries each, @k=20)")
    lines.append("")
    lines.append(
        "| problem | "
        + " | ".join(ARMS)
        + " | g\u00b7raw\u0394 | W g\u00b7raw | W g\u00b7nb25 | nDCG g\u2192raw |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    qmeans: dict[tuple, dict] = {}
    for r in rows_out:
        if r["k"] == 20:
            key = (r["arm"], r["problem"], r["query_id"])
            qmeans[key] = r

    for problem in sorted(pools):
        vals = {arm: prob_means[(problem, arm, 20)]["hit_prec_strict"] * 100 for arm in ARMS}
        d_raw = []
        d_nb = []
        ndcg_d = []
        for qid in [q["qid"] for p in suite if p["problem"] == problem for q in p["queries"]]:
            try:
                v_g = qmeans[("graph_vec", problem, qid)]["hit_prec_strict"]
                v_r = qmeans[("raw_bm25", problem, qid)]["hit_prec_strict"]
                v_n = qmeans[("nodes_bm25", problem, qid)]["hit_prec_strict"]
                g_ndcg = qmeans[("graph_vec", problem, qid)]["ndcg"]
                r_ndcg = qmeans[("raw_bm25", problem, qid)]["ndcg"]
            except KeyError:
                continue
            d_raw.append(v_g - v_r)
            d_nb.append(v_g - v_n)
            ndcg_d.append(g_ndcg - r_ndcg)
        w_raw = wilcoxon_p(d_raw)
        w_nb = wilcoxon_p(d_nb)
        lines.append(
            f"| {problem} | "
            + " | ".join(f"{vals[a]:.0f}%" for a in ARMS)
            + f" | {np.mean(d_raw) * 100:+.0f} | {w_raw:.3f} | {w_nb:.3f} | {np.mean(ndcg_d):+.3f} |"
        )

    lines.append("")
    lines.append("## Method vs schema decomposition (pooled @k=20, hit-precision)")
    lines.append("")
    p20 = {arm: pooled[(arm, 20)] for arm in ARMS}
    method_effect = p20["nodes_vec"]["hit_prec_strict"] - p20["nodes_bm25"]["hit_prec_strict"]
    schema_effect = p20["graph_vec"]["hit_prec_strict"] - p20["nodes_vec"]["hit_prec_strict"]
    lines.append(
        f"- method effect (BM25\u2192vector, nodes space): **{method_effect * 100:+.0f} pp**"
    )
    lines.append(f"- schema effect (nodes\u2192graph, vector): **{schema_effect * 100:+.0f} pp**")

    lines.append("")
    lines.append("## Global positive-DOI coverage at k=50 (distinct positives found per arm)")
    lines.append("")
    lines.append("| arm | distinct positive DOIs | pool share (union of 6 pools) |")
    lines.append("|---|---|---|")
    first_arm = list(arm_recs)[0]
    all_keys = list(arm_recs[first_arm])
    all_pool = set().union(*pools.values())
    for arm in ARMS:
        pos_set: set = set()
        for problem, qid in all_keys:
            for h in arm_recs[arm][(problem, qid)][:50]:
                for d in h.get("dois") or []:
                    for uc in doi_uc.get(d, ()):
                        if labels.get((d, uc)) == "positive":
                            pos_set.add(d)
        lines.append(
            f"| {arm} | {len(pos_set)} | {100 * len(pos_set & all_pool) / len(all_pool):.1f}% |"
        )

    lines.append("")
    lines.append("## Pairwise tests (n=30 queries, @k=20)")
    lines.append("")
    lines.append("| pair | onlyA:onlyB | McNemar p | Wilcoxon p (hitP) |")
    lines.append("|---|---|---|---|")

    first_arm = list(arm_recs)[0]
    all_keys = list(arm_recs[first_arm])

    def top1_pos(arm, problem, qid):
        hits = arm_recs[arm][(problem, qid)]
        if not hits:
            return False
        return any(labels.get((d, problem)) == "positive" for d in (hits[0].get("dois") or []))

    for a, b in [
        ("graph_vec", "raw_bm25"),
        ("graph_vec", "nodes_bm25"),
        ("edges_vec", "nodes_vec"),
        ("graph_vec", "nodes_vec"),
    ]:
        only_a = only_b = 0
        diffs = []
        for problem, qid in all_keys:
            ra = top1_pos(a, problem, qid)
            rb = top1_pos(b, problem, qid)
            only_a += ra and not rb
            only_b += rb and not ra
            va = eval_query(problem, arm_recs[a][(problem, qid)], 20)["hit_prec_strict"]
            vb = eval_query(problem, arm_recs[b][(problem, qid)], 20)["hit_prec_strict"]
            diffs.append(va - vb)
        p_mc = binom_two_sided(only_a, only_a + only_b)
        p_w = wilcoxon_p(diffs)
        lines.append(f"| {a} vs {b} | {only_a}:{only_b} | {p_mc:.3f} | {p_w:.3f} |")

    lines.append("")
    lines.append(
        "*shown = excludes unlabeled-for-problem hits/DOIs; strict counts them as "
        "misses. Recall is bounded by schema coverage above. pass labels grade 1.0 "
        "in nDCG only.*"
    )
    summary_md = RESULTS_DIR / "academic_summary.md"
    summary_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
