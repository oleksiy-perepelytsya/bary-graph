#!/usr/bin/env python3
"""build_gate.py — STRICT pre-creation schema gate for structural SMBs.

The loop builds SMBs with direct lib calls, NOT via the MCP server. This gate
guarantees that whatever gets inserted is schema-clean *before* the insert —
the same invariants `scripts/mcp_server._create_structure_meta_bary_body`
enforces, run as a dry check first:

  ERRORS (hard — block the build):
    - ids not 24-hex ObjectIds, or not mutually distinct
    - cm1 / cm2 / bridge doc missing from barygraph_poc
    - any of the three missing a vector
    - cm1.level != cm2.level (children must share level L)
    - bridge.level != L-1
    - L-2 < 1 (SMB level below the minimum)

  WARNINGS (report-only — never block; value judgment is the sole admission
  criterion, `child_cosine` is a diagnostic, never a denial):
    - pivot word not found in the bridge's leaf words (surface, do NOT refuse)
    - child_cosine < 0.20 (flag for later recheck, still build)

Usage
-----
    build_gate.py --check CM1 CM2 BRIDGE [--pivot WORD] [--author A]
    build_gate.py CM1 CM2 BRIDGE [--pivot WORD] [--author A]     # check + insert
    build_gate.py --check ... --json                              # machine-readable

Importable: `gate(...)` (pure check) and `create_structural(...)` (check+insert).
"""
from __future__ import annotations

import argparse
import json

from bson import ObjectId

from lib.bary_vec import compute_metabary_vec, level_factor
from lib.config import Settings
from lib.db import cm_leaf_words, get_collection
from lib.docs import metabary as _make_metabary
from lib.vector import unpack_vec

DEFAULT_AUTHOR = "big-pickle@opencode-0.5"


def _cos(v1, v2) -> float:
    n1 = sum(x * x for x in v1) ** 0.5
    n2 = sum(y * y for y in v2) ** 0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    return sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)


def gate(coll, cm1_id, cm2_id, bridge_id, pivot: str | None = None,
         author: str = DEFAULT_AUTHOR, alpha: float = 0.5) -> dict:
    """Strict structural pre-check. Returns {ok, errors, warnings, info}."""
    errors, warnings = [], []
    info = {}

    try:
        oid1, oid2, oid3 = ObjectId(cm1_id), ObjectId(cm2_id), ObjectId(bridge_id)
    except Exception:
        return {"ok": False,
                "errors": ["cm1/cm2/bridge must be 24-char hex ObjectIds"],
                "warnings": [], "info": {}}
    if len({oid1, oid2, oid3}) != 3:
        errors.append("cm1_id, cm2_id, bridge_id must be distinct")

    fields = {"vector": 1, "level": 1, "doc_type": 1, "accumulated_weight": 1}
    cm1 = coll.find_one({"_id": oid1}, fields)
    cm2 = coll.find_one({"_id": oid2}, fields)
    bridge = coll.find_one({"_id": oid3}, fields)

    for label, doc in (("cm1", cm1), ("cm2", cm2), ("bridge", bridge)):
        if doc is None:
            errors.append(f"{label} doc not found ({locals()[label + '_id']})")
            continue
        if not doc.get("vector"):
            errors.append(f"{label} has no vector")
        if doc.get("doc_type") not in ("baryedge", "node"):
            errors.append(f"{label} doc_type must be baryedge|node (got {doc.get('doc_type')})")

    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "info": info}

    try:
        L = int(cm1["level"])
        if int(cm2["level"]) != L:
            errors.append(f"cm1.level {L} != cm2.level {cm2['level']}")
        if int(bridge["level"]) != L - 1:
            errors.append(f"bridge.level {bridge['level']} != child_level-1 = {L-1}")
        mb_level = L - 2
        if mb_level < 1:
            errors.append(f"child_level {L} gives SMB level {mb_level} — minimum is 1")
    except (KeyError, TypeError, ValueError):
        errors.append("level fields must be ints")

    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "info": info}

    v1 = [float(x) for x in unpack_vec(cm1["vector"])]
    v2 = [float(x) for x in unpack_vec(cm2["vector"])]
    vb = [float(x) for x in unpack_vec(bridge["vector"])]
    child_cosine = round(_cos(v1, v2), 4)
    w1 = float(cm1.get("accumulated_weight", 1.0))
    w2 = float(cm2.get("accumulated_weight", 1.0))
    w3 = float(bridge.get("accumulated_weight", 1.0))

    if child_cosine < 0.20:
        warnings.append(f"child_cosine {child_cosine} < 0.20 — flag for recheck (not a denial)")

    if pivot:
        # Content check, surface-only: the bridge (or a child) should carry the
        # pivot word. NEVER a refusal — value judgment is the admission criterion.
        leaves = set()
        try:
            leaves.update(cm_leaf_words(coll, oid3, max_depth=15, max_words=64))
        except Exception:
            pass
        if pivot.lower() not in {w.lower() for w in leaves}:
            warnings.append(f"pivot {pivot!r} not found in bridge leaf words "
                            f"({sorted(leaves)[:6]}…) — verify by judgment")

    vec, q_mb_raw = compute_metabary_vec(v1, v2, vb, w1, w2, w3)
    acc_w = q_mb_raw * level_factor(mb_level, alpha)

    info = {
        "child_level": L, "bridge_level": L - 1, "smb_level": mb_level,
        "child_cosine": child_cosine, "q_mb_raw": round(float(q_mb_raw), 6),
        "accumulated_weight": round(float(acc_w), 6),
        "cm1_doc_type": cm1.get("doc_type"), "cm2_doc_type": cm2.get("doc_type"),
        "bridge_doc_type": bridge.get("doc_type"),
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info}


def create_structural(coll, cm1_id, cm2_id, bridge_id, pivot: str | None = None,
                      author: str = DEFAULT_AUTHOR, alpha: float = 0.5) -> dict:
    """Gate first, insert only when structurally clean. Returns result dict."""
    g = gate(coll, cm1_id, cm2_id, bridge_id, pivot, author, alpha)
    if not g["ok"]:
        return {"ok": False, "errors": g["errors"], "warnings": g["warnings"],
                "info": g["info"], "inserted_id": None}
    i = g["info"]
    try:
        oid1, oid2, oid3 = ObjectId(cm1_id), ObjectId(cm2_id), ObjectId(bridge_id)
        fields = {"vector": 1, "level": 1, "accumulated_weight": 1}
        cm1 = coll.find_one({"_id": oid1}, fields)
        cm2 = coll.find_one({"_id": oid2}, fields)
        bridge = coll.find_one({"_id": oid3}, fields)
        v1 = [float(x) for x in unpack_vec(cm1["vector"])]
        v2 = [float(x) for x in unpack_vec(cm2["vector"])]
        vb = [float(x) for x in unpack_vec(bridge["vector"])]
        vec, q_mb_raw = compute_metabary_vec(
            v1, v2, vb,
            float(cm1.get("accumulated_weight", 1.0)),
            float(cm2.get("accumulated_weight", 1.0)),
            float(bridge.get("accumulated_weight", 1.0)))
        doc = _make_metabary(
            oid1, oid2, level=i["smb_level"], vector=vec, q_mb_raw=q_mb_raw,
            accumulated_weight=q_mb_raw * level_factor(i["smb_level"], alpha))
        doc["source"] = "structural"
        doc["bridge_id"] = oid3
        if author.strip():
            doc["author"] = author.strip()
        inserted = coll.insert_one(doc)
        return {"ok": True, "errors": [], "warnings": g["warnings"], "info": i,
                "inserted_id": str(inserted.inserted_id)}
    except Exception as e:  # pragma: no cover - safety net
        return {"ok": False, "errors": [f"insert failed: {e!r}"],
                "warnings": g["warnings"], "info": i, "inserted_id": None}


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict pre-creation schema gate.")
    ap.add_argument("cm1_id")
    ap.add_argument("cm2_id")
    ap.add_argument("bridge_id")
    ap.add_argument("--pivot", default=None, help="pivot word (content warning only)")
    ap.add_argument("--author", default=DEFAULT_AUTHOR)
    ap.add_argument("--check", action="store_true", help="check only, do not insert")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    coll = get_collection(Settings.load())
    if args.check:
        g = gate(coll, args.cm1_id, args.cm2_id, args.bridge_id, args.pivot, args.author)
        verdict = "GATE PASS" if g["ok"] else "GATE FAIL"
    else:
        g = create_structural(coll, args.cm1_id, args.cm2_id, args.bridge_id,
                              args.pivot, args.author)
        verdict = (f"INSERTED {g['inserted_id']}" if g["ok"]
                   else f"REFUSED ({len(g['errors'])} errors)")

    out = {"verdict": verdict, **g}
    if args.json:
        print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False))
    else:
        print(verdict)
        for e in g["errors"]:
            print("  ERROR:", e)
        for w in g["warnings"]:
            print("  WARN :", w)
        if g.get("info"):
            print("  info:", json.dumps(g["info"], ensure_ascii=False))
    return 0 if g["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())