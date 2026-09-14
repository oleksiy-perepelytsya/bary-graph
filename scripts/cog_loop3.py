#!/usr/bin/env python3
"""Third cognitive-agent call launcher: SMB value grading.

Runs the deterministic schema validator (scripts.validate_smb) over
smb_proposals.jsonl first, then composes an opencode run message that puts
every not-yet-graded proposal + its validation report + its paired paper
record together with the value-validation prompt, and launches a DETACHED
opencode run so the cycle can be repeated.

Deliverable: cognitive/batches/smb_grades.jsonl — one object per graded
proposal ({doi, proposal_index, axes, value_score, verdict, ...}). Nothing
is written to Mongo here: grading is judgment; the build is a separate
ingestion script (scripts.build_smb).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = ROOT / "cognitive" / "batches"
TERMS_FILE = BATCH_DIR / "terms_batch.jsonl"
PAPERS_FILE = BATCH_DIR / "papers_batch.jsonl"
PROPOSALS_FILE = BATCH_DIR / "smb_proposals.jsonl"
GRADES_FILE = BATCH_DIR / "smb_grades.jsonl"
VALIDATION_FILE = BATCH_DIR / "smb_validation.jsonl"
PROMPT_FILE = ROOT / "cognitive" / "prompts" / "smb_validation.md"
LOG_DIR = Path(os.environ.get("OPENCODE_LOG_DIR", "/tmp/opencode"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def papers_by_doi() -> dict[str, dict]:
    return {r.get("doi"): r for r in load_jsonl(PAPERS_FILE) if r.get("doi")}


def graded_keys(grades: list[dict]) -> set[tuple[str, int]]:
    keys = set()
    for g in grades:
        if g.get("doi") and isinstance(g.get("proposal_index"), (int, float)):
            keys.add((g["doi"], int(g["proposal_index"])))
    return keys


def ungraded_proposals(force: bool) -> list[tuple[int, dict]]:
    """Return [(proposal_index, record)] for proposals not yet graded."""
    proposals = load_jsonl(PROPOSALS_FILE)
    if force:
        return [(i, r) for i, r in enumerate(proposals, 1)]
    graded = graded_keys(load_jsonl(GRADES_FILE))
    return [(i, r) for i, r in enumerate(proposals, 1)
            if (r.get("doi"), i) not in graded]


def validation_by_index() -> dict[int, dict]:
    return {r.get("proposal_index"): r
            for r in load_jsonl(VALIDATION_FILE)
            if isinstance(r.get("proposal_index"), (int, float))}


def compose_message(proposals: list[tuple[int, dict]]) -> str:
    """Build the full step-3 prompt from ungraded proposals + context."""
    if not proposals:
        sys.exit("no proposals to grade (all already graded — use --force to regrade)")

    papers = papers_by_doi()
    if PROMPT_FILE.exists():
        prompt = PROMPT_FILE.read_text()
    else:
        prompt = "(validation prompt stub — cognitive/prompts/smb_validation.md not yet provided)"

    blocks = []
    for idx, rec in proposals:
        doi = rec.get("doi", "")
        paper = papers.get(doi, {})
        terms = "\n".join(
            f'  cm1: {rec.get("cm1_terms") or []}\n'
            f'  bridge: {rec.get("bridge_terms") or []}\n'
            f'  cm2: {rec.get("cm2_terms") or []}'
        )
        queries = "; ".join(rec.get("grounding_queries") or [])
        blocks.append(f"""
--- PROPOSAL {idx} ---
DOI: {doi}
AUTHOR (proposing model): {rec.get("author", "")}
RATIONAL: {rec.get("rationale", "")}
RELATION SUMMARY: {rec.get("relation_summary", "")}
GROUNDING QUERIES: {queries}
TRIAD:
{terms}

PAPER CONTEXT:
TITLE: {paper.get("title", "")}
ABSTRACT: {paper.get("abstract", "")[:2000]}
""")

    message = f"""
You are the SMB value-validation step of the cognitive pipeline (third agent call).

VALIDATION PROMPT (follow it exactly):
{prompt}

PROPOSALS TO GRADE ({len(proposals)}):
{''.join(blocks)}
""".strip()
    return message


def launch(message: str, tag: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:80]
    log_file = LOG_DIR / f"cog_loop3_{slug}.log"
    cmd = ["opencode", "run", "--auto", message]
    with log_file.open("w") as lf:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=lf, stderr=lf,
            start_new_session=True,
        )
    print(f"launched PID {proc.pid} | log {log_file} | grades -> {GRADES_FILE}")
    return proc.pid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the composed message, do not launch")
    ap.add_argument("--validate-only", action="store_true",
                    help="run the schema validator (writes smb_validation.jsonl) and stop")
    ap.add_argument("--force", action="store_true",
                    help="grade every proposal, even if already graded")
    ap.add_argument("--tag", default="smb_grade", help="log file tag")
    args = ap.parse_args()

    if args.validate_only:
        import subprocess as _sp
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "validate_smb.py"), "--write"]
        return _sp.call(cmd)

    proposals = ungraded_proposals(args.force)
    print(f"grading {len(proposals)} proposal(s); total in file: "
          f"{len(load_jsonl(PROPOSALS_FILE))}")

    message = compose_message(proposals)
    if args.dry_run:
        print(message)
        return 0
    launch(message, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())