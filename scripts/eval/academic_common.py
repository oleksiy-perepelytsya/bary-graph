"""Shared plumbing for the academic-dataset retrieval evaluation.

Targets barygraph_poc ONLY (the academic term corpus was ingested there; it is
independent of the multilingual barygraph_all build). Every entrypoint
hard-checks the DB name so a mis-sourced .env.build-all aborts immediately.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from bson import ObjectId

from lib.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = Path("/workspace/papers_combined.parquet")
QUERIES_PATH = PROJECT_ROOT / "evaluation" / "queries.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

K_LIST = [5, 10, 20, 50]
K_MAX = 50
NUM_CANDIDATES = 200

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOP = {
    "a",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "to",
    "up",
    "we",
    "with",
    "an",
    "can",
    "currently",
    "which",
    "both",
}

BARYEDGE_FILTER = {"doc_type": "baryedge"}
NODE_FILTER = {"doc_type": "node"}


def norm_doi(doi: Any) -> str | None:
    if not doi:
        return None
    d = str(doi).strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:", "doi.org/"):
        if d.startswith(p):
            d = d[len(p) :]
    return d or None


def require_poc(settings: Settings) -> None:
    if settings.mongo_db != "barygraph_poc":
        sys.exit(
            f"academic eval must run against barygraph_poc, got "
            f"{settings.mongo_db!r} — refusing to touch the build DB"
        )


def load_rows() -> list[dict[str, Any]]:
    return pq.read_table(str(PARQUET_PATH)).to_pylist()


def label_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for r in rows:
        d = norm_doi(r.get("doi"))
        uc = r.get("use_case_key")
        if d and uc and r.get("triage_label"):
            out[(d, uc)] = r["triage_label"]
    return out


def positive_pools(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    pools: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        d = norm_doi(r.get("doi"))
        uc = r.get("use_case_key")
        if d and uc and r.get("triage_label") == "positive":
            pools[uc].add(d)
    return dict(pools)


def doi_use_case_map(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        d = norm_doi(r.get("doi"))
        uc = r.get("use_case_key")
        if d and uc:
            out[d].add(uc)
    return dict(out)


def reverse_doi_index(bridge_coll: Any) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for doc in bridge_coll.find({}, {"node_ids": 1}):
        d = norm_doi(doc.get("_id"))
        if not d:
            continue
        for nid in doc.get("node_ids") or []:
            out[str(nid)].add(d)
    return dict(out)


def doi_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    all_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        d = norm_doi(r.get("doi"))
        if d:
            all_rows[d].append(r)
    for d, rs in all_rows.items():
        uc = sorted({r["use_case_key"] for r in rs if r.get("use_case_key")})
        pos = any(r.get("triage_label") == "positive" for r in rs)
        meta[d] = {"use_cases": uc, "positive_anywhere": pos}
    return meta


def to_str(oid: Any) -> str:
    return str(ObjectId(oid)) if isinstance(oid, (ObjectId,)) else str(oid)
