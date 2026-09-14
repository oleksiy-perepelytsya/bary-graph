#!/usr/bin/env python3
"""Ingestion-side build step: turn accepted SMB proposals into real SMBs.

Reads proposals (cognitive/batches/smb_proposals.jsonl) and their grades
(cognitive/batches/smb_grades.jsonl). For every proposal whose verdict is
`keep`, resolves the branch term lists to ACTUAL existing node ids and calls
the shared SMB-creation body (scripts.mcp_server._create_structure_meta_bary_body)
so geometry/levels stay identical to the pipeline — then registers the
paper's DOI in the doi_bridges provenance index, exactly like the academic
batch ingestion path (ingest_batch.py -> doi_bridge.register/propagate).

This script is the ONLY place SMBs are built from cognitive proposals and the
ONLY place their DOI provenance is written. The MCP tool
(create_structure_meta_bary) never touches doi_bridges, by design.

Resolution rules (deterministic, no LLM, no embedding):
  - cm1/cm2 branches resolve to L15 SENSE nodes (first sense idx per word);
  - the bridge resolves to an L14 WORD node;
  - thus child level = 15, bridge level = 14, SMB inserted at L13.
  A proposal whose branches cannot all resolve to existing nodes is reported
  `pending` (needs sense/word creation, which needs the Ollama embedder —
  deferred while the s04 pipeline owns Ollama), not built.

--dry-run prints the exact plan (including the doi_bridge.register calls)
  and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.mcp_server import (
    _coll,
    _create_structure_meta_bary_body,
    _settings,
)
from lib import doi_bridge
from lib.db import get_collection

ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = ROOT / "cognitive" / "batches"
PROPOSALS_FILE = BATCH_DIR / "smb_proposals.jsonl"
GRADES_FILE = BATCH_DIR / "smb_grades.jsonl"
BUILD_LOG_FILE = BATCH_DIR / "smb_builds.jsonl"

KEEP_VERDICTS = {"keep", "revise"}
# revise = keep geometry, fix later; build it so the DOI is linked, log verdict.
# Only 'drop' blocks a build.


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


def resolve_word_ids(word: str) -> list[str]:
    """Exact-match L14 word node ids (same query as mcp find_word)."""
    docs = _coll.find(
        {"doc_type": "node", "node_type": "word", "properties.word": word},
        {"_id": 1},
    )
    return [str(d["_id"]) for d in docs]


def sense_ids_for_word(word: str) -> list[str]:
    """L15 sense node ids for a word, ordered by sense_idx (same query as mcp word_senses)."""
    docs = _coll.find(
        {"doc_type": "node", "node_type": "sense", "properties.word": word},
        {"_id": 1, "properties.sense_idx": 1},
    ).sort("properties.sense_idx", 1)
    return [str(d["_id"]) for d in docs]


def resolve_branch_sense(terms: list[str]) -> str | None:
    """First resolvable sense node across the branch's words."""
    for term in terms:
        senses = sense_ids_for_word(term)
        if senses:
            return senses[0]
    return None


def resolve_bridge_word(terms: list[str]) -> str | None:
    """First resolvable L14 word node across the bridge's words."""
    for term in terms:
        words = resolve_word_ids(term)
        if words:
            return words[0]
    return None


def _parse_created(body_result: str) -> tuple[dict, str | None]:
    """Return (parsed_result, error). The shared body returns _fmt json on
    success and a plain error string on failure."""
    try:
        data = json.loads(body_result)
    except json.JSONDecodeError:
        return {}, body_result
    if data.get("ok") is not True:
        return data, body_result
    return data, None


def build_proposal(idx: int, proposal: dict, grade: dict | None, dry_run: bool) -> dict:
    doi = proposal.get("doi", "")
    verdict = (grade or {}).get("verdict", "keep")
    author = proposal.get("author") or (grade or {}).get("author") or ""

    rec = {
        "doi": doi,
        "proposal_index": idx,
        "verdict": verdict,
        "status": "pending",
        "reason": "",
        "cm1_id": None, "bridge_id": None, "cm2_id": None,
        "smb_id": None, "child_cosine": None,
        "doi_registered": False,
        "author": author,
    }

    cm1 = resolve_branch_sense(proposal.get("cm1_terms") or [])
    bridge = resolve_bridge_word(proposal.get("bridge_terms") or [])
    cm2 = resolve_branch_sense(proposal.get("cm2_terms") or [])

    if not (cm1 and cm2 and bridge):
        missing = []
        if not cm1:
            missing.append("cm1")
        if not bridge:
            missing.append("bridge")
        if not cm2:
            missing.append("cm2")
        rec["reason"] = f"unresolved nodes for: {', '.join(missing)} (needs embedding/creation)"
        rec["cm1_id"], rec["bridge_id"], rec["cm2_id"] = cm1, bridge, cm2
        return rec

    rec["cm1_id"], rec["bridge_id"], rec["cm2_id"] = cm1, bridge, cm2
    if dry_run:
        rec["status"] = "planned"
        rec["reason"] = "dry-run — no insert performed"
        return rec
    body_result = _create_structure_meta_bary_body(cm1, cm2, bridge, author)
    data, err = _parse_created(body_result)
    if err:
        rec["status"] = "failed"
        rec["reason"] = err
        return rec

    rec["status"] = "created"
    rec["smb_id"] = data.get("id")
    rec["child_cosine"] = data.get("child_cosine")
    rec["level"] = data.get("level")
    return rec


def register_doi(rec: dict, dry_run: bool) -> None:
    """doi_bridges.register + propagate — mirrors ingest_batch.py / s08."""
    if rec["status"] not in ("created", "planned"):
        return
    if not rec["doi"]:
        return
    smb_id = rec["smb_id"] or "<new_smb_id>"
    bridge_coll = doi_bridge.get_bridge_collection(_settings)
    mode = "DRY" if dry_run else "WRT"
    print(f"    [{mode}] doi_bridge.register(doi_bridges, doi={rec['doi']}, smb_id={smb_id})")
    print(f"    [{mode}] doi_bridge.propagate(doi_bridges, smb_id={smb_id}, "
          f"constituents=[{rec['cm1_id']}, {rec['cm2_id']}, {rec['bridge_id']}])")
    if dry_run:
        return
    doi_bridge.ensure_indexes(bridge_coll)
    doi_bridge.register(bridge_coll, [rec["doi"]], rec["smb_id"])
    doi_bridge.propagate(
        bridge_coll, rec["smb_id"],
        [rec["cm1_id"], rec["cm2_id"], rec["bridge_id"]],
    )
    rec["doi_registered"] = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, write nothing (no Mongo, no files)")
    ap.add_argument("--log", action="store_true",
                    help=f"append build log to {BUILD_LOG_FILE}")
    ap.add_argument("--also-revise", action="store_true",
                    help="build 'revise' verdicts too (default: only 'keep')")
    ap.add_argument("--max-proposals", type=int, default=0,
                    help="cap how many proposals to attempt (0 = all)")
    args = ap.parse_args()

    proposals = load_jsonl(PROPOSALS_FILE)
    grades = load_jsonl(GRADES_FILE)
    grades_by = {(g.get("doi"), g.get("proposal_index")): g for g in grades if g.get("doi")}

    if not proposals:
        print(f"no proposals in {PROPOSALS_FILE}")
        return 0

    print(f"building {len(proposals)} proposal(s) | mode={'DRY' if args.dry_run else 'WRITE'}\n")
    builds: list[dict] = []
    built = n_pending = n_failed = n_skipped = 0

    for idx, proposal in enumerate(proposals, 1):
        grade = grades_by.get((proposal.get("doi"), idx))
        verdict = (grade or {}).get("verdict", "keep")
        if verdict == "drop" or (verdict == "revise" and not args.also_revise):
            print(f"  [{idx}] {proposal.get('doi')} — verdict={verdict}, skipped (--also-revise to build)")
            n_skipped += 1
            continue

        rec = build_proposal(idx, proposal, grade, dry_run=args.dry_run)
        register_doi(rec, dry_run=args.dry_run)
        builds.append(rec)

        if rec["status"] in ("created", "planned"):
            built += 1
        elif rec["status"] == "pending":
            n_pending += 1
        else:
            n_failed += 1

        print(f"  [{idx}] {proposal.get('doi')} — {rec['status']}"
              f"{' smb=' + rec['smb_id'] if rec['smb_id'] else ''}"
              f"{' cosine=' + str(rec['child_cosine']) if rec['child_cosine'] is not None else ''}")
        if rec["reason"]:
            print(f"        reason: {rec['reason']}")

        if args.max_proposals and built + n_pending + n_failed + n_skipped >= args.max_proposals:
            break

    print(f"\nbuilt={built} pending={n_pending} failed={n_failed} skipped={n_skipped} "
          f"({len(builds)} attempted; {built} of those would-be/are created)")

    if args.log and not args.dry_run:
        with BUILD_LOG_FILE.open("a") as fh:
            for rec in builds:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"build log -> {BUILD_LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())