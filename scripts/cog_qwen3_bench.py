#!/usr/bin/env python3
"""qwen3-on-CPU cognitive agent benchmark driver.

Runs the cognitive pipeline (extraction -> SMB analysis) with the LOCAL
qwen3:8b-cog model on CPU (via opencode run -m ollama/qwen3:8b-cog), one
paper at a time, timing each step. NO Mongo writes — the SMB step only
appends proposals to smb_proposals.jsonl.

Each extraction appends one {"doi","terms"} record to terms_batch.jsonl;
each analysis appends 1..3 proposal records to smb_proposals.jsonl. Records
are appended (never rewritten) so earlier pipeline records survive.

Usage:
  python3.11 scripts/cog_qwen3_bench.py --limit 1 --stage extraction
  python3.11 scripts/cog_qwen3_bench.py --limit 9       # full pass, both stages
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "cognitive" / "batches"
PAPERS = BATCH / "papers_batch.jsonl"
TERMS = BATCH / "terms_batch.jsonl"
PROPOSALS = BATCH / "smb_proposals.jsonl"
EXTRACTION_PROMPT = ROOT / "cognitive" / "prompts" / "extraction_v7_1.md"
ANALYSIS_PROMPT = ROOT / "cognitive" / "prompts" / "smb_analysis.md"
LOG_DIR = Path("/tmp/opencode/qwen3_cog")
MODEL = "ollama/gpt-oss"
AUTHOR = "gpt-oss@cpu-bench"


def n_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.read_text().splitlines() if l.strip())


def last_record(path: Path, doi: str | None = None) -> dict | None:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    for l in reversed(lines):
        d = json.loads(l)
        if doi is None or d.get("doi") == doi:
            return d
    return None


def slug(doi: str) -> str:
    return doi.replace("/", "_").replace(":", "_")


def compose_extraction(paper: dict) -> str:
    return f"""
You are the cognitive-extraction step of the pipeline, running on a SINGLE paper.

Read /workspace/bary-vector/cognitive/prompts/extraction_v7_1.md — the
extraction prompt. Apply it AS-IS to the paper below and extract terms per
its rules. The DOI to use is EXACTLY: {paper["doi"]} (bare, do not transform).

TITLE: {paper["title"]}
ABSTRACT:
{paper["abstract"]}

DELIVERABLE — build exactly the JSON object the prompt describes
({{"doi": "<the DOI above, unchanged>", "terms": [{{"id","term","gloss"}}...]}})
and APPEND it as ONE JSON line to /workspace/bary-vector/cognitive/batches/terms_batch.jsonl
with a trailing newline. Read the file first; append only, never rewrite the
file. Do NOT create anything in Mongo. Do NOT ask questions — run unattended.
Then reply one line: DONE doi=<doi> terms=<n>.
""".strip()


def compose_analysis(paper: dict, terms_rec: dict) -> str:
    t = terms_rec.get("terms", [])
    terms_lines = "\n".join(
        f'  t{i:02d}: "{x["term"]}" — {x["gloss"]}' for i, x in enumerate(t, 1)
    )
    return f"""
You are the SMB-proposal step of the cognitive pipeline (second agent call),
running on qwen3:8b-cog.

CONTEXT — the paper and its extracted terms (read
/workspace/bary-vector/cognitive/prompts/smb_analysis.md and FOLLOW it):

DOI: {paper["doi"]}
TITLE: {paper["title"]}
ABSTRACT:
{paper["abstract"]}

EXTRACTED TERMS ({len(t)}):
{terms_lines}

ANALYSIS PROMPT (follow it — IN PARTICULAR the Procedure: read terms, run
PoC queries against BaryGraph, hunt for the tension):
{ANALYSIS_PROMPT.read_text()}

You have the barygraph MCP tools (context_search, human_readable_search,
semantic_search, find_word, word_edges) — use them for the probing steps;
every word you put in a branch or bridge must come from a hit you actually
retrieved. The DOI in every proposal must be EXACTLY: {paper["doi"]}.

DELIVERABLE — append your proposals, one JSON object per line, to
/workspace/bary-vector/cognitive/batches/smb_proposals.jsonl (read it first;
append only, never rewrite). Each object: doi, rationale (1-2 sentences),
cm1_terms, bridge_terms, cm2_terms, relation_summary, grounding_queries
(the PoC queries you ran), author: "{AUTHOR}". At most 3 proposals; fewer,
stronger. If you find no genuine triad, append nothing and say so — never
invent words that did not appear in your retrieved hits.
Do NOT create anything in Mongo — proposal is the only deliverable.
Do NOT ask questions — run unattended.
Then reply one line: DONE doi=<doi> proposals=<n>.
""".strip()


def run_agent(message: str, log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["opencode", "run", "--auto", "-m", MODEL, message]
    with log_file.open("w") as fh:
        return subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=fh, start_new_session=True)


def poll_growth(path: Path, baseline: int, timeout_s: float, proc) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if n_records(path) > baseline:
            return True
        if proc.poll() is not None:
            return False
        time.sleep(15)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=9)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stage", choices=["extraction", "analysis", "both"], default="both")
    ap.add_argument("--model", default="", help="opencode model id, e.g. ollama/gpt-oss")
    ap.add_argument("--poll-timeout", type=int, default=3600)
    global MODEL, AUTHOR
    args = ap.parse_args()
    if args.model:
        MODEL = args.model
        AUTHOR = f"{args.model.split('/')[-1]}@cpu-bench"

    papers = [json.loads(l) for l in PAPERS.read_text().splitlines() if l.strip()]
    papers = papers[args.offset: args.offset + args.limit]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    t_total = time.time()

    for i, paper in enumerate(papers, args.offset + 1):
        doi = paper["doi"]
        tcyc = time.time()
        print(f"\n=== paper {i}/{args.offset + len(papers)} | doi={doi} ===", flush=True)
        row = {"paper": i, "doi": doi, "step1_s": None, "step2_s": None, "terms": 0, "proposals": 0, "err": None}

        if args.stage in ("extraction", "both"):
            base = n_records(TERMS)
            msg = compose_extraction(paper)
            p = run_agent(msg, LOG_DIR / f"step1_{slug(doi)}.log")
            print(f"[step1] launch pid={p.pid}", flush=True)
            ok = poll_growth(TERMS, base, args.poll_timeout, p)
            if ok:
                rec = last_record(TERMS, doi)
                row["step1_s"] = round(time.time() - tcyc, 1)
                row["terms"] = len(rec.get("terms", [])) if rec else 0
                print(f"[step1] landed in {row['step1_s']:.0f}s | terms={row['terms']}", flush=True)
            else:
                row["err"] = f"step1: no record after {args.poll_timeout}s"
                print(f"[step1] FAILED: {row['err']}", flush=True)
                if p.poll() is None:
                    p.kill()
                results.append(row)
                continue

        if args.stage in ("analysis", "both"):
            terms_rec = last_record(TERMS, doi)
            if terms_rec is None:
                row["err"] = "step2: no terms record for doi"
                print(f"[step2] FAILED: {row['err']}", flush=True)
                results.append(row)
                continue
            base = n_records(PROPOSALS)
            t2 = time.time()
            msg = compose_analysis(paper, terms_rec)
            p = run_agent(msg, LOG_DIR / f"step2_{slug(doi)}.log")
            print(f"[step2] launch pid={p.pid}", flush=True)
            ok = poll_growth(PROPOSALS, base, args.poll_timeout, p)
            row["step2_s"] = round(time.time() - t2, 1)
            row["proposals"] = n_records(PROPOSALS) - base
            if ok:
                print(f"[step2] landed in {row['step2_s']:.0f}s | proposals={row['proposals']}", flush=True)
            else:
                row["err"] = (row["err"] or "") + f" step2: no growth after {args.poll_timeout}s"
                print(f"[step2] NO GROWTH in {row['step2_s']:.0f}s | proposals={row['proposals']}", flush=True)
                if p.poll() is None:
                    p.kill()

        results.append(row)
        with (LOG_DIR / "bench_timings.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n=== total {len(results)} papers in {(time.time()-t_total)/60:.1f} min ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())