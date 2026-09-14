#!/usr/bin/env python3
"""Deterministic SMB-proposal schema validator (no LLM, zero tokens).

Reads cognitive/batches/smb_proposals.jsonl and checks every proposal record
against the schema agreed with the step-2 raw prompt (smb_analysis.md):

  required fields: doi, rationale, cm1_terms, bridge_terms, cm2_terms,
                   relation_summary, grounding_queries
  types:           doi/rationale/relation_summary = non-empty str;
                   cm1/bridge/cm2_terms + grounding_queries = non-empty list[str]
  quality bar:     branch term counts within [3, 8];
                   cm1 ∩ cm2 == ∅ (distinct branches);
                   bridge not a re-labeling of a branch (warn-level);
                   every grounding query prefixed with a known tool name
  cross-check:     proposal doi must exist in papers_batch.jsonl

Output: cognitive/batches/smb_validation.jsonl — one record per proposal:
  {doi, proposal_index, ok, errors: [...], warnings: [...],
   n_cm1, n_bridge, n_cm2, n_queries}

--write gates the report file (default: print-only). Exit code 0 always
(reporting is informational here), unless the input itself is unreadable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = ROOT / "cognitive" / "batches"
TERMS_FILE = BATCH_DIR / "terms_batch.jsonl"
PAPERS_FILE = BATCH_DIR / "papers_batch.jsonl"
PROPOSALS_FILE = BATCH_DIR / "smb_proposals.jsonl"
VALIDATION_FILE = BATCH_DIR / "smb_validation.jsonl"
REPORT_FILE = None  # set by main() when --write

REQUIRED_FIELDS = {
    "doi", "rationale",
    "cm1_terms", "bridge_terms", "cm2_terms",
    "relation_summary", "grounding_queries", "author",
}
STR_FIELDS = {"doi", "rationale", "relation_summary", "author"}
LIST_FIELDS = {"cm1_terms", "bridge_terms", "cm2_terms", "grounding_queries"}
BRANCH_FIELDS = ("cm1_terms", "bridge_terms", "cm2_terms")

MIN_BRANCH = 3
MAX_BRANCH = 8

KNOWN_TOOLS = {
    "context_search", "human_readable_search", "semantic_search",
    "find_word", "word_senses", "word_edges", "edge_info",
    "leaf_nodes", "traverse_up", "sample_metabary",
    "associative_search", "associative_progressive",
}


def load_jsonl(path: Path) -> list[tuple[int, dict | None, object]]:
    """Return [(line_no, parsed_record_or_None, raw_or_error)] per line."""
    if not path.exists():
        sys.exit(f"missing file: {path}")
    lines = path.read_text().splitlines()
    out: list[tuple[int, dict | None, object]] = []
    for i, ln in enumerate(lines, 1):
        if not ln.strip():
            continue
        try:
            out.append((i, json.loads(ln), ln))
        except json.JSONDecodeError as e:
            out.append((i, None, e))
    return out


def papers_dois() -> set[str]:
    if not PAPERS_FILE.exists():
        return set()
    dois: set[str] = set()
    for _, rec, _ in load_jsonl(PAPERS_FILE):
        if rec and rec.get("doi"):
            dois.add(rec["doi"])
    return dois


def validate_record(rec: dict, known_dois: set[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    missing = REQUIRED_FIELDS - set(rec)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    extra = set(rec) - REQUIRED_FIELDS
    if extra:
        warnings.append(f"unknown fields: {sorted(extra)}")

    for f in STR_FIELDS & set(rec):
        v = rec[f]
        if not isinstance(v, str) or not v.strip():
            errors.append(f"{f}: must be a non-empty string")
        elif f == "doi" and len(v.strip()) < 4:
            errors.append("doi: implausibly short")

    for f in LIST_FIELDS & set(rec):
        v = rec[f]
        if not isinstance(v, list) or not v:
            errors.append(f"{f}: must be a non-empty list")
            continue
        if not all(isinstance(x, str) and x.strip() for x in v):
            errors.append(f"{f}: must contain only non-empty strings")

    doi = rec.get("doi", "").strip()
    if doi and known_dois and doi not in known_dois:
        errors.append(f"doi {doi} not found in papers_batch.jsonl")

    for f in BRANCH_FIELDS:
        v = rec.get(f)
        if not isinstance(v, list):
            continue
        n = len(v)
        if n < MIN_BRANCH:
            errors.append(f"{f}: {n} terms, below the {MIN_BRANCH}-term quality bar")
        if n > MAX_BRANCH:
            warnings.append(f"{f}: {n} terms, above the {MAX_BRANCH}-term guidance")

    cm1 = set(rec.get("cm1_terms") or [])
    bridge = set(rec.get("bridge_terms") or [])
    cm2 = set(rec.get("cm2_terms") or [])
    if cm1 & cm2:
        warnings.append(f"cm1 ∩ cm2 overlap: {sorted(cm1 & cm2)}")
    if bridge and bridge <= cm1:
        warnings.append("bridge is a subset of cm1 — likely a re-labeling")
    if bridge and bridge <= cm2:
        warnings.append("bridge is a subset of cm2 — likely a re-labeling")

    for q in rec.get("grounding_queries") or []:
        if not isinstance(q, str) or ":" not in q:
            errors.append(f"grounding query not in 'tool: query' form: {q!r}")
            continue
        tool = q.split(":", 1)[0].split(None, 1)[0].strip().strip("()")
        if tool not in KNOWN_TOOLS:
            warnings.append(f"unknown grounding tool {tool!r} in: {q!r}")

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposals", default=str(PROPOSALS_FILE),
                    help="proposals JSONL to validate")
    ap.add_argument("--papers", default=str(PAPERS_FILE),
                    help="papers metadata JSONL (DOI cross-check)")
    ap.add_argument("--write", action="store_true",
                    help=f"write validation report to {VALIDATION_FILE}")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.proposals))
    known = papers_dois() if PAPERS_FILE.exists() else set()

    report: list[dict] = []
    n_ok = n_bad = 0
    print(f"validating {len(rows)} proposal record(s)\n")
    for line_no, rec, raw in rows:
        proposal_index = report and report[-1]["proposal_index"] + 1 or 1
        if rec is None:
            print(f"  [line {line_no}] UNPARSEABLE: {raw}")
            report.append({
                "doi": None, "proposal_index": proposal_index, "ok": False,
                "errors": [f"line {line_no}: invalid JSON"], "warnings": [],
            })
            n_bad += 1
            continue

        errors, warnings = validate_record(rec, known)
        ok = not errors
        n_ok += ok
        n_bad += not ok
        report.append({
            "doi": rec.get("doi"),
            "proposal_index": proposal_index,
            "ok": ok,
            "errors": errors,
            "warnings": warnings,
            "n_cm1": len(rec.get("cm1_terms") or []),
            "n_bridge": len(rec.get("bridge_terms") or []),
            "n_cm2": len(rec.get("cm2_terms") or []),
            "n_queries": len(rec.get("grounding_queries") or []),
        })
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {rec.get('doi')} (idx {proposal_index})")
        for w in warnings:
            print(f"        warn: {w}")
        for e in errors:
            print(f"        ERR : {e}")

    print(f"\n{n_ok} valid, {n_bad} invalid, {len(rows)} total")
    if args.write:
        VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with VALIDATION_FILE.open("w") as fh:
            for r in report:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"report -> {VALIDATION_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())