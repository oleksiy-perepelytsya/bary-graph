#!/usr/bin/env python3
"""ledger.py — durable record writers for the native SMB loop.

Formalizes the record schema so every loop entry carries ANALYSIS, not just a
verdict (see cognitive/prompts/native_loop.md):

  loop_cycles.jsonl  →  {cycle, seed_mb_id, seed_level, outcome,
                         probes_run: [...], rationale: "1-2 sentences",
                         note: short label, at}
  smb_builds.jsonl   →  structural manifest row + "analysis" block holding
                         the rubric scores that informed the build.

The writers append only (they never rewrite or dedupe); state rolling stays
with the caller/loop_state.json.

Usage
-----
    from cognitive.ledger import record_cycle, record_build

    record_cycle(401, seed_id, 13, "similarity-cluster",
                 probes_run=["loop_pivot: 16 seeds '13,12'",
                             "leaf check: coven<->karass same-domain"],
                 rationale="All neighbours are group-nouns (coven/fellowship/"
                           "karass); one cluster with a seam, not two lives.",
                 note="coven/fellowship/karass group-nouns")

    record_build(name="lawn:grass-ground/linen-fabric", smb_id=..., cm1_id=...,
                 ..., bridge_terms=["lawn"], ..., child_cosine=0.4212,
                 levels={"children":15,"bridge":14,"smb":13}, seed_mb_id=...,
                 cycle=406,
                 analysis={"distance":4,"distinctness":5,"mechanism":5,
                           "grounding":3, "mechanism_note": "..."},
                 rationale="...")
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

BATCH_DIR = Path(__file__).resolve().parent / "batches"
CYCLES = BATCH_DIR / "loop_cycles.jsonl"
BUILDS = BATCH_DIR / "smb_builds.jsonl"
DEFAULT_AUTHOR = "big-pickle@opencode-0.5"


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record_cycle(cycle: int, seed_mb_id: str, seed_level: int, outcome: str,
                 probes_run: list[str], rationale: str, note: str = "") -> dict:
    """Append one cycle record. `probes_run` and `rationale` are REQUIRED so a
    skip is auditable (a skip without a reason is noise)."""
    rec = {
        "cycle": cycle,
        "seed_mb_id": seed_mb_id,
        "seed_level": seed_level,
        "outcome": outcome,
        "probes_run": probes_run,
        "rationale": rationale,
        "note": note,
        "at": _utcnow(),
    }
    with CYCLES.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def record_build(*, name: str, smb_id: str, cm1_id: str, cm2_id: str,
                 bridge_id: str, cm1_terms: list[str], bridge_terms: list[str],
                 cm2_terms: list[str], child_cosine: float, levels: dict,
                 cycle: int, rationale: str, seed_mb_id: str,
                 analysis: dict, author: str = DEFAULT_AUTHOR,
                 note: str = "") -> dict:
    """Append one structural build row with its analysis block."""
    row = {
        "name": name,
        "smb_id": smb_id,
        "cm1_id": cm1_id,
        "cm2_id": cm2_id,
        "bridge_id": bridge_id,
        "cm1_terms": cm1_terms,
        "bridge_terms": bridge_terms,
        "cm2_terms": cm2_terms,
        "child_cosine": child_cosine,
        "levels": levels,
        "source": "structural",
        "author": author,
        "cycle": cycle,
        "seed_mb_id": seed_mb_id,
        "analysis": analysis,
        "rationale": rationale,
        "note": note,
    }
    with BUILDS.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row