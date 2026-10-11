#!/usr/bin/env python3
"""oracle_read.py — run the uncertainty oracle over the SMB manifest and record
the reads to a durable ledger.

This is the *formalized* way to consult `not_sure.py` on SMBs: it fixes the
state template and the question so runs are comparable, and appends every read
(both the choice and the probabilities) to a JSONL ledger for later analysis.

Usage
-----
    oracle_read.py [--all | --sample N] [--seed S]
                   [--template term|prose|both]
                   [--out PATH] [--model M] [--host H]

    --all              every structural manifest entry (default: --sample 25)
    --sample N         random N (default 25)
    --seed S           RNG seed for reproducible sampling (default 20261011)
    --template ...     'term' (concise term-lists), 'prose' (narrative), or
                       'both' (two reads per candidate)
    --out PATH         ledger to append to
                       (default cognitive/batches/oracle_reads.jsonl)

Only manifest entries with the structural shape (name + cm1_terms + cm2_terms)
are read; the 6 legacy DOI records are skipped.

NOTE ON STATUS (MISUSE, KEPT AS CALIBRATION): this tool was built to validate
SMBs with tev1:0.8b — that use is a category error, superseded by the user's
correction (Oct-11): SMBs are products of analysis admitted by value judgment
and are NOT to be checked deterministically; the oracle is a decision aid for
the agent's own choices only. This tool is retained because its output is the
negative-result ledger (framing-dominated: 2% yes term-list vs 27% prose)
proving the tiny model cannot gate SMBs. Do not extend it — it exists as
evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics as st
import sys

from cognitive.not_sure import judge

MANIFEST = "cognitive/batches/smb_builds.jsonl"
DEFAULT_OUT = "cognitive/batches/oracle_reads.jsonl"
QUESTION = "Is this a productive one-word-two-lives adjacency?"
CRITERIA = {"yes": "two genuinely distinct senses of the same word",
            "no": "co-located, near-synonymous, or forced"}


def fmt(terms, k=6):
    return ", ".join(str(t) for t in (terms or [])[:k])


def state_term(r):
    return (f"Word: {r['name']}.\n"
            f"Sense cluster A: {fmt(r.get('cm1_terms'))}.\n"
            f"Sense cluster B: {fmt(r.get('cm2_terms'))}.\n"
            f"Shared bridge concept: {fmt(r.get('bridge_terms'))}.\n"
            f"Both clusters are reached from the single word '{r['name']}'.")


def state_prose(r):
    return (f"The word '{r['name']}' has two readings. "
            f"One reading: {fmt(r.get('cm1_terms'))}. "
            f"A quite different reading: {fmt(r.get('cm2_terms'))}. "
            f"They are linked by the shared notion of {fmt(r.get('bridge_terms'), 4)}.")


TEMPLATES = {"term": state_term, "prose": state_prose}


def main() -> int:
    ap = argparse.ArgumentParser(description="Oracle read over the SMB manifest.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--template", choices=["term", "prose", "both"], default="term")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]
    rows = [r for r in rows if r.get("name") and r.get("cm1_terms") and r.get("cm2_terms")]
    if not args.all:
        rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))

    templates = ["term", "prose"] if args.template == "both" else [args.template]
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.host:
        kwargs["host"] = args.host

    out = open(args.out, "a", encoding="utf-8")
    n_written = 0
    try:
        for tmpl in templates:
            yes = 0
            pys = []
            print(f"--- template={tmpl}  n={len(rows)} ---", flush=True)
            for r in rows:
                v = judge(TEMPLATES[tmpl](r), QUESTION, ["yes", "no"],
                          criteria=CRITERIA, **kwargs)
                p = float((v.get("probabilities") or {}).get("yes", 0.0))
                pys.append(p)
                if v.get("choice") == "yes":
                    yes += 1
                rec = {
                    "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model": kwargs.get("model", "tev1:0.8b"),
                    "template": tmpl,
                    "question": QUESTION,
                    "criteria": CRITERIA,
                    "name": r.get("name"),
                    "smb_id": r.get("smb_id"),
                    "author": r.get("author"),
                    "child_cosine": r.get("child_cosine"),
                    "choice": v.get("choice"),
                    "p_yes": p,
                    "probabilities": v.get("probabilities"),
                    "confidence": v.get("confidence"),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            out.flush()
            print(f"    yes {yes}/{len(rows)} ({100*yes/len(rows):.0f}%)  "
                  f"mean p(yes) {st.mean(pys):.3f}", flush=True)
    finally:
        out.close()
    print(f"\nwrote {n_written} reads -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
