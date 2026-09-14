#!/usr/bin/env python3
"""Second cognitive-agent call launcher.

Picks up the LAST extracted-terms record (terms_batch.jsonl) and its paired
paper record (papers_batch.jsonl), composes an opencode run message that puts
both in context together with the SMB-analysis prompt, and launches a DETACHED
opencode run (default model from opencode.json) so the cycle can be repeated.

Nothing is written to Mongo here: the agent's deliverable is a set of SMB
proposals with short grounding. Proposals go to cognitive/batches/smb_proposals.jsonl.

Post-processing:
  --post-process <logfile>  Extract JSON proposals from a completed run's log,
                            enforce the correct DOI, validate fields, and
                            append to smb_proposals.jsonl. Use after a detached
                            run finishes, or pipe --wait to do both.
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
PROMPT_FILE = ROOT / "cognitive" / "prompts" / "smb_analysis.md"
LOG_DIR = Path(os.environ.get("OPENCODE_LOG_DIR", "/tmp/opencode"))


def last_record(path: Path) -> dict:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        sys.exit(f"no records in {path}")
    return json.loads(lines[-1])


# ── post-processing (DOI enforcement + file append) ──────────────────────────


def extract_proposals(text: str) -> list[dict]:
    """Find all JSON arrays in *text* and flatten their items into a list.

    Uses a balanced-bracket scan (respecting string literals) so nested
    arrays like ``["a", "b"]`` inside a bigger array parse correctly.
    """
    proposals: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "[":
            i += 1
            continue
        # scan for the balanced bracket pair starting at i
        depth = 0
        j = i
        in_str = False
        escaped = False
        while j < n:
            c = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if j < n and depth == 0:
            try:
                arr = json.loads(text[i : j + 1])
                if isinstance(arr, list):
                    proposals.extend(p for p in arr if isinstance(p, dict))
            except json.JSONDecodeError:
                pass
        i = j + 1 if (j < n and depth == 0) else i + 1
    return proposals


def validate_proposal(prop: dict, expected_doi: str) -> dict | None:
    """Validate and fix a single proposal.  Returns fixed copy or None."""
    required = ("rationale", "cm1_terms", "bridge_terms", "cm2_terms")
    if not all(k in prop and prop[k] for k in required):
        return None
    prop = dict(prop)  # shallow copy
    prop["doi"] = expected_doi  # enforce exact DOI
    for key in ("cm1_terms", "bridge_terms", "cm2_terms"):
        if not isinstance(prop[key], list) or not prop[key]:
            return None
    return prop


def post_process(logfile: Path, expected_doi: str, dry_run: bool = False) -> int:
    """Extract, validate, and append proposals from a run log.

    Returns number of proposals appended (0 if none valid).
    """
    if not logfile.exists():
        print(f"log not found: {logfile}", file=sys.stderr)
        return 0

    # strip ANSI escapes so JSON parsing isn't confused
    raw = logfile.read_text()
    clean = re.sub(r"\x1b\[[0-9;]*m", "", raw)

    proposals = extract_proposals(clean)
    if not proposals:
        print("no JSON proposals found in log")
        return 0

    valid: list[dict] = []
    for p in proposals:
        v = validate_proposal(p, expected_doi)
        if v:
            valid.append(v)
        else:
            print(f"  skipped invalid proposal: {list(p.keys())}")

    if not valid:
        print("no valid proposals after validation")
        return 0

    if dry_run:
        for v in valid:
            print(json.dumps(v, ensure_ascii=False))
        return len(valid)

    # append JSONL
    PROPOSALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROPOSALS_FILE.open("a") as fh:
        for v in valid:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"appended {len(valid)} proposal(s) to {PROPOSALS_FILE}")
    return len(valid)


def compose_message(extra_queries: str = "", extra_message: str = "") -> str:
    """Build the full step-2 prompt from the LAST records in the batch files."""
    terms_rec = last_record(TERMS_FILE)
    paper_rec = last_record(PAPERS_FILE)

    if terms_rec.get("doi") != paper_rec.get("doi"):
        sys.exit(
            f"DOI mismatch between last records: terms={terms_rec.get('doi')} papers={paper_rec.get('doi')}"
        )

    # analysis prompt — stub until user provides it
    if PROMPT_FILE.exists():
        analysis_prompt = PROMPT_FILE.read_text()
    else:
        analysis_prompt = "(analysis prompt stub — cognitive/prompts/smb_analysis.md not yet provided)"

    terms_lines = "\n".join(
        f'  t{i:02d}: "{t["term"]}" — {t["gloss"]}' for i, t in enumerate(terms_rec["terms"], 1)
    )

    message = f"""
You are the SMB-proposal step of the cognitive pipeline (second agent call).

CONTEXT — last extracted paper and its terms:

DOI: {terms_rec["doi"]}
ARXIV: {paper_rec.get("arxiv_id", "")}
TITLE: {paper_rec.get("title", "")}

ABSTRACT:
{paper_rec.get("abstract", "")}

EXTRACTED TERMS ({len(terms_rec["terms"])}):
{terms_lines}

ANALYSIS PROMPT (follow it):
{analysis_prompt}
""".strip()

    if extra_queries:
        message += f"\n\nRun these PoC queries against BaryGraph: {extra_queries}"
    if extra_message:
        message += f"\n\n{extra_message}"

    message += f"""

DELIVERABLE — return your SMB proposals as a JSON array directly in your
final response. The pipeline will parse your output and handle persistence —
you do NOT need to write any files. Each proposal must be a JSON object
with: doi, rationale (1-2 sentences), cm1_terms: [..], bridge_terms: [..],
cm2_terms: [..], relation_summary: one sentence, grounding_queries: the PoC
queries you ran, and author: your model name with version from your identity
line. Do NOT write to smb_proposals.jsonl and do NOT call any cog_create_*
tools — proposal is the deliverable, not the build. Just output the JSON array.
"""
    return message.strip()


def launch(message: str, model: str = "") -> int:
    """Launch a detached opencode run with the composed message; returns PID."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    doi_slug = last_record(TERMS_FILE)["doi"].replace("/", "_")
    log_file = LOG_DIR / f"cog_loop2_{doi_slug}.log"
    cmd = ["opencode", "run", "--auto"]
    if model:
        cmd += ["-m", model]
    cmd += [message]
    with log_file.open("w") as lf:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=lf, stderr=lf,
            start_new_session=True,  # detach so the agent can be cycled
        )
    print(f"launched PID {proc.pid} | log {log_file} | proposals -> {PROPOSALS_FILE}")
    return proc.pid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the composed message, do not launch")
    ap.add_argument("--model", default="", help="opencode model id (provider/model), e.g. ollama/qwen3:8b")
    ap.add_argument("--message", default="", help="extra instructions appended to the composed prompt")
    ap.add_argument("--extra-queries", default="", help="extra PoC BG queries to run (comma-separated)")
    ap.add_argument("--wait", action="store_true", help="launch and block until the agent finishes, then post-process")
    ap.add_argument("--post-process", metavar="LOG", default="", help="extract JSON proposals from LOG, enforce DOI, append")
    args = ap.parse_args()

    # --- post-process mode: parse a completed run's log ---
    if args.post_process:
        paper_rec = last_record(PAPERS_FILE)
        n = post_process(Path(args.post_process), paper_rec["doi"])
        return 0 if n > 0 else 1

    message = compose_message(args.extra_queries, args.message)

    if args.dry_run:
        print(message)
        return 0

    pid = launch(message, args.model)

    if args.wait:
        log = LOG_DIR / f"cog_loop2_{last_record(TERMS_FILE)['doi'].replace('/', '_')}.log"
        _, status = os.waitpid(pid, 0)  # blocks until the opencode run exits
        n = post_process(log, last_record(PAPERS_FILE)["doi"])
        return 0 if n > 0 else 1

    return pid


if __name__ == "__main__":
    sys.exit(main())