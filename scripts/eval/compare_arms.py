#!/usr/bin/env python3
"""Paired comparison of two or more US-matrix arms on the cases they share.

Uses the same metric definitions as the archive's ``score.py`` but reports the
arms side by side, restricted to the intersection of case ids (so the deltas are
paired), with the metadata fields separated from the semantic ones the graph
structure could plausibly move.

    python scripts/eval/compare_arms.py \
        --gold data/us_matrix/gold/us_matrix_gold_100 \
        A=evaluation/results/us_matrix/llm_fulltext.jsonl \
        B=evaluation/results/us_matrix/llm_graph_struct.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# --- metric definitions (mirrors us_matrix_gold_100/score.py) ---------------
ws = lambda s: re.sub(r"\s+", " ", s or "").strip()            # noqa: E731
nz = lambda s: re.sub(r"[^a-z0-9§]", "", (s or "").lower())    # noqa: E731

META = ["court_level", "jurisdiction", "date_decided"]
SEM = ["procedure_type", "area_of_law", "standard_of_review", "disposition", "winning_party"]
EXACT = META + SEM


def prf(g, c, match):
    if not g or not c:
        return None, None
    hg = sum(1 for x in g if any(match(x, y) for y in c))
    hc = sum(1 for y in c if any(match(x, y) for x in g))
    return hg / len(g), hc / len(c)


def cit_key(x):
    return nz(x.get("citation")) if isinstance(x, dict) else nz(x)


def cit_m(x, y):
    kx, ky = cit_key(x), cit_key(y)
    if kx and ky:
        return kx == ky
    if isinstance(x, dict) and isinstance(y, dict):
        return nz(x.get("cited_case", ""))[:20] == nz(y.get("cited_case", ""))[:20]
    return False


def st_m(x, y):
    x, y = nz(x if isinstance(x, str) else json.dumps(x)), nz(y if isinstance(y, str) else json.dumps(y))
    return bool(x and y) and (x in y or y in x)


def load_jsonl(p: Path) -> dict[str, dict]:
    out = {}
    for ln in p.read_text(errors="replace").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r.get("case_id") is not None:
                out[str(r["case_id"])] = r
    return out


def metrics(gold: dict, cand: dict, cases_dir: Path) -> dict:
    acc = {k: [] for k in EXACT}
    cit_r, cit_p, st_r, st_p, qv, miss = [], [], [], [], [], []
    for cid, g in gold.items():
        c = cand.get(cid)
        if not c:
            continue
        miss.append(len([k for k in g if k not in c]))
        for k in EXACT:
            acc[k].append(int(nz(str(g.get(k))) == nz(str(c.get(k)))))
        r, p = prf(g.get("citations") or [], c.get("citations") or [], cit_m)
        if r is not None:
            cit_r.append(r)
        if p is not None:
            cit_p.append(p)
        r, p = prf(g.get("statutes_applied") or [], c.get("statutes_applied") or [], st_m)
        if r is not None:
            st_r.append(r)
        if p is not None:
            st_p.append(p)
        txt_path = cases_dir / f"{cid}.txt"
        if txt_path.exists():
            txt = ws(txt_path.read_text(errors="ignore"))
            q = [ws(x.get("evidence_quote")) for x in (c.get("arguments") or []) if isinstance(x, dict)]
            if q:
                qv.append(sum(1 for s in q if len(s) >= 12 and s in txt) / len(q))
    m = {k: (sum(v) / len(v) if v else None) for k, v in acc.items()}
    m["semantic_mean"] = (sum(m[k] for k in SEM) / len(SEM)) if all(m[k] is not None for k in SEM) else None
    m["metadata_mean"] = (sum(m[k] for k in META) / len(META)) if all(m[k] is not None for k in META) else None
    m["missing_fields"] = sum(miss) / len(miss) if miss else None
    m["cit_recall"] = sum(cit_r) / len(cit_r) if cit_r else None
    m["cit_precision"] = sum(cit_p) / len(cit_p) if cit_p else None
    m["stat_recall"] = sum(st_r) / len(st_r) if st_r else None
    m["stat_precision"] = sum(st_p) / len(st_p) if st_p else None
    m["quotes_verbatim"] = sum(qv) / len(qv) if qv else None
    m["n"] = len([c for c in gold if c in cand])
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=path.jsonl")
    ap.add_argument("--gold", default="data/us_matrix/gold/us_matrix_gold_100")
    ap.add_argument("--cases", default="")
    args = ap.parse_args()

    gold_dir = Path(args.gold)
    cases_dir = Path(args.cases) if args.cases else gold_dir / "cases"
    gold_all = load_jsonl(gold_dir / "gold.jsonl")

    arms = {}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        arms[name] = load_jsonl(Path(path))

    shared = set.intersection(*(set(a) for a in arms.values())) if arms else set()
    gold = {cid: g for cid, g in gold_all.items() if cid in shared}
    if not gold:
        raise SystemExit("no case ids shared across arms (and present in gold)")

    res = {name: metrics(gold, {c: a[c] for c in gold}, cases_dir) for name, a in arms.items()}
    names = list(arms)
    order = ["n", "missing_fields", "metadata_mean", "court_level", "jurisdiction",
             "date_decided", "semantic_mean", "procedure_type", "area_of_law",
             "standard_of_review", "disposition", "winning_party",
             "quotes_verbatim", "cit_recall", "cit_precision", "stat_recall", "stat_precision"]

    print(f"paired on {res[names[0]]['n']} shared cases\n")
    hdr = f"{'metric':20s}" + "".join(f"{n:>16s}" for n in names)
    if len(names) == 2:
        hdr += f"{'delta':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for k in order:
        row = f"{k:20s}"
        for n in names:
            v = res[n].get(k)
            row += f"{v:>16.3f}" if isinstance(v, (int, float)) else f"{'-':>16s}"
        if len(names) == 2:
            a, b = res[names[0]].get(k), res[names[1]].get(k)
            d = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            row += f"{d:>+10.3f}" if d is not None else f"{'-':>10s}"
        print(row)


if __name__ == "__main__":
    main()
