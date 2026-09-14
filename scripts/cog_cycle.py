#!/usr/bin/env python3
"""Cognitive agent cycle driver — runs step1 (extraction) then step2 (SMB
proposal) repeatedly, measuring per-cycle wall time so we can see how the
pipeline scales over a sustained run.

Each iteration:
  1. launches a detached `opencode run` (step 1) instructing the agent to pick
     its OWN research topic and skip any DOI already in terms_batch.jsonl
  2. waits for terms_batch.jsonl to grow by one record (poll)
  3. launches step 2 via scripts.cog_loop2.py, which picks up the last record
  4. waits for the step-2 agent to exit, then post-processes its stdout JSON
     into smb_proposals.jsonl
  5. logs: cycle, doi, terms count, step1 time, step2 time

Usage:
  python3.11 scripts/cog_cycle.py --cycles 5
  python3.11 scripts/cog_cycle.py --minutes 120
  python3.11 scripts/cog_cycle.py --cycles 10 --model ollama/deepseek-r1-distill-qwen-32b
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = ROOT / "cognitive" / "batches"
TERMS_FILE = BATCH_DIR / "terms_batch.jsonl"
PROPOSALS_FILE = BATCH_DIR / "smb_proposals.jsonl"
LOG_DIR = Path("/tmp/opencode")

STEP1_STUB = """Run the cognitive extraction pipeline step 1 on a SINGLE arXiv paper.

ALREADY-STORED DOIs (you MUST NOT use any of these — pick a paper with a DIFFERENT DOI):
{doi_list}

1. CHOOSE YOUR OWN RESEARCH TOPIC for this run — any scientific area of interest.
2. Call cog_arxiv_search with that topic (top_k=5).
3. From the results, find a paper whose DOI is NOT in the already-stored list above. If the top result's DOI is stored, try later results. If all results are stored, pick a different topic and search again. Call cog_arxiv_get_paper with the NEW paper's arxiv_id.
4. Read cognitive/prompts/extraction_v7_1.md. Apply it AS-IS to the paper's abstract and extract terms. Build exactly the JSON it describes: {{"doi": "...", "terms": [...]}}.
5. Call cog_store_paper_extraction with doi (must start with "10." — have a real DOI or the 10.48550/arXiv.<id> form ready), terms_json (the terms array as a JSON string), arxiv_id, title, abstract (full), authors, categories.
6. If the store call FAILS (e.g. invalid DOI), read the error, fix the DOI, and retry with a corrected value. Only AFTER a store call returns a success message, reply: PAPER doi=<doi> arxiv=<id> terms=<n>.
If no paper with a new DOI can be found, reply that clearly and store nothing.
AUTONOMY RULE: You run unattended. NEVER ask a confirmation/permission question — the operator is not watching. Choose the best candidate (prefer a real journal/Crossref DOI starting with 10.; if none of a paper's identifiers is a 10. DOI, do NOT invent one and do NOT use its arXiv ID as a DOI — use the 10.48550/arXiv.<arxiv_id> form as the last resort, and if even that is rejected, try a different paper), proceed, and report. Do not stall.
"""


def count_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.read_text().splitlines() if l.strip())


def load_existing_dois(path: Path) -> set[str]:
    dois: set[str] = set()
    if not path.exists():
        return dois
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doi = json.loads(line).get("doi")
            if doi:
                dois.add(doi)
        except Exception:
            pass
    return dois


def last_doi(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1]).get("doi")
    except Exception:
        return None


def poll_growth(path: Path, baseline: int, timeout_s: float, proc: subprocess.Popen | None = None) -> bool:
    """Wait until `path` grows past `baseline`. Aborts early if `proc` exits
    without growth (so a step that lands nothing doesn't burn the timeout)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if count_records(path) > baseline:
            return True
        if proc is not None and proc.poll() is not None:
            # process finished but nothing landed
            return False
        time.sleep(10)
    return False


def poll_exit(proc: subprocess.Popen, timeout_s: float) -> bool:
    """Wait until *proc* exits; True if it exited, False on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(10)
    return False


def run_step1(cycle: int, model: str = "") -> subprocess.Popen:
    dois = load_existing_dois(TERMS_FILE)
    doi_list = "\n".join(f"  {d}" for d in sorted(dois)) if dois else "  (none yet)"
    msg = STEP1_STUB.format(doi_list=doi_list)
    cmd = ["opencode", "run", "--auto"]
    if model:
        cmd += ["-m", model]
    cmd += [msg]
    log = LOG_DIR / f"cog_cycle{cycle}_step1.log"
    with log.open("w") as fh:
        return subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=fh, start_new_session=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=0, help="max number of cycles")
    ap.add_argument("--minutes", type=int, default=0, help="stop after this many minutes total")
    ap.add_argument("--poll-timeout", type=int, default=900, help="s to wait for each step to land (default 15 min)")
    ap.add_argument("--model", default="", help="opencode model id (provider/model), e.g. ollama/deepseek-r1-distill-qwen-32b")
    args = ap.parse_args()

    if args.cycles <= 0 and args.minutes <= 0:
        sys.exit("give --cycles N and/or --minutes M")

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    cycle = 0
    t_start_total = time.time()
    summary = []

    while args.cycles == 0 or cycle < args.cycles:
        if deadline and time.time() >= deadline:
            print(f"[cycle] total time limit reached ({args.minutes} min); stopping")
            break
        cycle += 1
        print(f"\n=== cycle {cycle} (topic: agent's choice) ===", flush=True)

        t0 = time.time()
        terms_baseline = count_records(TERMS_FILE)
        doi_before = last_doi(TERMS_FILE)
        proc = run_step1(cycle, args.model)
        print(f"[step1] launched PID {proc.pid}", flush=True)

        if not poll_growth(TERMS_FILE, terms_baseline, args.poll_timeout, proc):
            print(f"[step1] no new record (agent exited without storing, or {args.poll_timeout}s timeout)", flush=True)
            if proc.poll() is None:
                proc.kill()
            continue
        t1 = time.time()
        doi_after = last_doi(TERMS_FILE)
        terms_count = count_records(TERMS_FILE) - terms_baseline
        print(f"[step1] landed in {t1-t0:.0f}s | new rcords={terms_count}", flush=True)

        if doi_after == doi_before:
            print("[step1] stored nothing new — skipping step2", flush=True)
            continue

        # step 2 — launch the REAL agent, then post-process its stdout JSON
        t2 = time.time()
        from scripts.cog_loop2 import compose_message, post_process
        msg2 = compose_message()
        log2 = LOG_DIR / f"cog_cycle{cycle}_step2.log"
        with log2.open("w") as fh:
            cmd2 = ["opencode", "run", "--auto"]
            if args.model:
                cmd2 += ["-m", args.model]
            cmd2 += [msg2]
            p = subprocess.Popen(
                cmd2,
                cwd=ROOT, stdout=fh, stderr=fh, start_new_session=True,
            )
        print(f"[step2] launched PID {p.pid}", flush=True)
        if not poll_exit(p, args.poll_timeout):
            print(f"[step2] agent did not exit within {args.poll_timeout}s", flush=True)
            if p.poll() is None:
                p.kill()
            continue
        # agent done — extract JSON from its log, enforce DOI, append
        expected_doi = last_doi(TERMS_FILE)
        n_prop = post_process(log2, expected_doi)
        t3 = time.time()
        if n_prop == 0:
            print(f"[step2] no valid proposals parsed in {t3-t2:.0f}s", flush=True)
            continue
        summary.append({
            "cycle": cycle, "topic": "agent-choice", "doi": doi_after,
            "step1_s": round(t1 - t0, 1), "step2_s": round(t3 - t2, 1),
            "proposals": n_prop,
        })
        print(f"[step2] landed in {t3-t2:.0f}s | new proposals={n_prop}", flush=True)

    elapsed = time.time() - t_start_total
    print(f"\n=== done: {cycle} cycles in {elapsed/60:.1f} min ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())