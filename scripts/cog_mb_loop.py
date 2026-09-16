#!/usr/bin/env python3
"""Random-MB SMB-candidate discovery loop.

Alternative cognitive flow: there is no paper, no extracted term list, no
DOI. Instead each cycle launches ONE detached `opencode run` that:

  1. samples a random MetaBary from the graph (sample_metabary)
  2. grounds/analyzes it with enrich_request
  3. proposes 1-3 SMB candidates — NOT necessarily the sampled MB itself,
     any new relation built up during the analysis
  4. returns a JSON array (the pipeline parses it; the agent writes nothing)

The driver polls the run's log, post-processes the stdout JSON into
cognitive/batches/smb_candidates.jsonl, and reports per-cycle timing.

Post-processing:
  --post-process <logfile>  Extract JSON proposals from a completed run's
                            log, validate candidates, append to
                            smb_candidates.jsonl.

Usage:
  python3.11 scripts/cog_mb_loop.py --cycles 3
  python3.11 scripts/cog_mb_loop.py --minutes 120
  python3.11 scripts/cog_mb_loop.py --cycles 10 --model ollama/deepseek-r1-distill-qwen-32b
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = ROOT / "cognitive" / "batches"
CANDIDATES_FILE = BATCH_DIR / "smb_candidates.jsonl"
PROMPT_FILE = ROOT / "cognitive" / "prompts" / "mb_candidate.md"
LOG_DIR = Path("/tmp/opencode")

REQUIRED = ("source_mb_id", "rationale", "cm1_terms", "bridge_terms", "cm2_terms")


def canonical_author(model: str) -> str:
    """Derive author string from --model config arg (deterministic)."""
    if not model:
        return ""
    m = model.split("/", 1)[-1] if "/" in model else model
    return f"{m}@opencode"


def count_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.read_text().splitlines() if l.strip())


def compose_message() -> str:
    """Build the full prompt for one random-MB discovery run."""
    if PROMPT_FILE.exists():
        prompt = PROMPT_FILE.read_text()
    else:
        sys.exit(f"prompt file not found: {PROMPT_FILE}")

    return f"""
You are the random-MB SMB-candidate discovery step of the cognitive pipeline.

ANALYSIS PROMPT (follow it exactly):
{prompt}

DELIVERABLE — return your SMB candidates as a JSON array directly in your
final response. The pipeline parses your output and handles persistence —
you do NOT need to write any files. Each candidate must be a JSON object
with: source_mb_id, source_level, rationale (1-2 sentences), cm1_terms: [..],
bridge_terms: [..], cm2_terms: [..], relation_summary: one sentence,
grounding_probes: the sample_metabary/enrich_request calls you ran, and
author: your model name with version from your identity line — do NOT add this
field yourself; the driver stamps it automatically. If you found
no genuine triad, output []. Do NOT write to smb_candidates.jsonl and do
NOT call any create/build tools — the candidate is the deliverable.
""".strip()


# ── post-processing (validation + file append) ────────────────────────────────


def extract_proposals(text: str) -> list[dict]:
    """Find all JSON arrays in *text* and flatten their items into a list.

    Uses a balanced-bracket scan (respecting string literals) so nested
    arrays inside a bigger array parse correctly.
    """
    proposals: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "[":
            i += 1
            continue
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


def validate_candidate(cand: dict) -> dict | None:
    """Validate a single candidate. Returns fixed copy or None.

    Unlike the paper-driven proposals there is no DOI — the seed is the
    sampled MetaBary (_id + level), which the candidate must provenance.
    """
    if not all(k in cand and cand[k] for k in REQUIRED):
        return None
    cand = dict(cand)  # shallow copy
    for key in ("cm1_terms", "bridge_terms", "cm2_terms"):
        if not isinstance(cand[key], list) or not cand[key]:
            return None
    if not isinstance(cand.get("source_mb_id"), str):
        return None
    if not isinstance(cand.get("source_level", 0), (int, float)):
        cand["source_level"] = int(cand["source_level"])
    return cand


def post_process(logfile: Path, dry_run: bool = False, model: str = "") -> int:
    """Extract, validate, and append candidates from a run log.

    Returns number of candidates appended (0 if none valid).
    If *model* is set, author is overwritten with the config-derived
    canonical form (ignoring any self-reported value from the agent).
    """
    if not logfile.exists():
        print(f"log not found: {logfile}", file=sys.stderr)
        return 0

    raw = logfile.read_text()
    clean = re.sub(r"\x1b\[[0-9;]*m", "", raw)

    proposals = extract_proposals(clean)
    if not proposals:
        print("no JSON candidates found in log")
        return 0

    author = canonical_author(model)

    valid: list[dict] = []
    for p in proposals:
        v = validate_candidate(p)
        if v:
            if author:
                v["author"] = author
            valid.append(v)
        else:
            print(f"  skipped invalid candidate: {list(p.keys())}")

    if not valid:
        print("no valid candidates after validation")
        return 0

    if dry_run:
        for v in valid:
            print(json.dumps(v, ensure_ascii=False))
        return len(valid)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_FILE.open("a") as fh:
        for v in valid:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"appended {len(valid)} candidate(s) to {CANDIDATES_FILE}")
    return len(valid)


# ── launch + cycle driver ─────────────────────────────────────────────────────


def launch(message: str, cycle: int, model: str = "") -> subprocess.Popen:
    """Launch a detached opencode run with the composed message."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"cog_mb_cycle{cycle}.log"
    cmd = ["opencode", "run", "--auto", "--title", f"cog-cycle-{cycle}"]
    if model:
        cmd += ["-m", model]
    cmd += [message]
    with log_file.open("w") as fh:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=fh, stderr=fh, start_new_session=True,
        )
    print(f"launched PID {proc.pid} | log {log_file} | candidates -> {CANDIDATES_FILE}")
    return proc


def poll_exit(proc: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(10)
    return False


def session_ids() -> set[str]:
    """Snapshot current opencode session ids (first column of `session list`)."""
    try:
        out = subprocess.run(
            ["opencode", "session", "list"], capture_output=True, text=True,
            cwd=ROOT, timeout=60,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return set()
    return set(re.findall(r"^(ses_\S+)\s", out, re.M))


def delete_session(sid: str) -> bool:
    """Delete an opencode session by id (best-effort)."""
    if not sid:
        return False
    try:
        r = subprocess.run(
            ["opencode", "session", "delete", sid], capture_output=True,
            text=True, cwd=ROOT, timeout=30,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=0, help="max number of cycles")
    ap.add_argument("--minutes", type=int, default=0, help="stop after this many minutes total")
    ap.add_argument("--poll-timeout", type=int, default=900, help="s to wait for each run (default 15 min)")
    ap.add_argument("--model", default="", help="opencode model id (provider/model)")
    ap.add_argument("--dry-run", action="store_true", help="print the composed message, do not launch")
    ap.add_argument("--post-process", metavar="LOG", default="", help="extract candidates from LOG and append (instead of cycling)")
    args = ap.parse_args()

    if args.post_process:
        return 0 if post_process(Path(args.post_process), model=args.model) > 0 else 1

    if args.dry_run:
        print(compose_message())
        return 0

    if args.cycles <= 0 and args.minutes <= 0:
        sys.exit("give --cycles N and/or --minutes M")

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    cycle = 0
    t_start = time.time()
    summary = []

    while args.cycles == 0 or cycle < args.cycles:
        if deadline and time.time() >= deadline:
            print(f"[cycle] total time limit reached ({args.minutes} min); stopping")
            break
        cycle += 1
        print(f"\n=== cycle {cycle} (seed: random MetaBary) ===", flush=True)

        baseline = count_records(CANDIDATES_FILE)
        t0 = time.time()
        before_sessions = session_ids()
        proc = launch(compose_message(), cycle, args.model)
        if not poll_exit(proc, args.poll_timeout):
            print(f"[cycle {cycle}] agent did not exit within {args.poll_timeout}s", flush=True)
            if proc.poll() is None:
                proc.kill()
            continue

        logfile = LOG_DIR / f"cog_mb_cycle{cycle}.log"
        n_cand = post_process(logfile, model=args.model)
        t1 = time.time()
        for sid in session_ids() - before_sessions:
            if delete_session(sid):
                print(f"[cycle {cycle}] cleared session {sid}", flush=True)
        if n_cand == 0:
            print(f"[cycle {cycle}] no valid candidates parsed in {t1-t0:.0f}s", flush=True)
            continue
        summary.append({"cycle": cycle, "s": round(t1 - t0, 1), "candidates": n_cand})
        print(f"[cycle {cycle}] landed in {t1-t0:.0f}s | new candidates={n_cand}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== done: {cycle} cycles in {elapsed/60:.1f} min ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())