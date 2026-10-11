#!/usr/bin/env python3
"""smb_audit.py — STRUCTURAL consistency audit of built structural SMBs.

This is NOT a content/analysis check. It verifies, per SMB, three mechanical
properties against the live `barygraph_poc` database:

  1. LEVELING RULE   — SMB.level == L-2, both children at level L (cm1, cm2 same
                       level), bridge at level L-1, i.e. children = SMB.level+2
                       and bridge = SMB.level+1. All three referenced docs must
                       exist and carry a vector.
  2. NODE REFERENCES — cm1_id, cm2_id, bridge_id stored *on the SMB doc* resolve
                       to real documents; the three ids are mutually distinct.
  3. IDs             — the SMB doc has a stable _id, an explicit bridge_id field,
                       doc_type='baryedge', source='structural'.

The manifest (`smb_builds.jsonl`) is cross-checked against the DB: every run
reports how many manifest rows have a matching DB doc and vice versa.

Usage
-----
    smb_audit.py [--sample N] [--seed S] [--all]
      --sample N   random structural-rows to audit (default 10)
      --seed S     RNG seed for reproducibility (default 20261011)
      --all        audit every structural row in the manifest
"""
from __future__ import annotations

import argparse
import json
import random

from bson import ObjectId

from lib.config import Settings
from lib.db import get_collection

MANIFEST = "cognitive/batches/smb_builds.jsonl"


def audit_row(coll, row) -> dict:
    smb_id = row["smb_id"]
    out = {"name": row.get("name", smb_id), "smb_id": smb_id, "checks": {}, "info": {}}

    smb = coll.find_one({"_id": ObjectId(smb_id)})
    out["checks"]["smb_doc_exists"] = smb is not None
    if smb is None:
        return out

    ok = True
    def chk(label, cond):
        nonlocal ok
        out["checks"][label] = bool(cond)
        if not cond:
            ok = False
        return cond

    chk("smb.doc_type=baryedge", smb.get("doc_type") == "baryedge")
    chk("smb.source=structural", smb.get("source") == "structural")
    chk("smb.has_vector", bool(smb.get("vector")))
    chk("smb.has_explicit_bridge_id", bool(smb.get("bridge_id")))

    try:
        L = int(smb["level"])
    except (KeyError, TypeError, ValueError):
        chk("smb.level_is_int", False)
        return out
    chk("smb.level_is_int", True)

    cm1_id, cm2_id, bridge_id = smb.get("cm1_id"), smb.get("cm2_id"), smb.get("bridge_id")
    chk("ids_distinct", len({str(cm1_id), str(cm2_id), str(bridge_id)}) == 3)

    def fetch(oid):
        if oid is None:
            return None
        try:
            return coll.find_one({"_id": ObjectId(oid)})
        except Exception:
            return None

    cm1, cm2, bridge = fetch(cm1_id), fetch(cm2_id), fetch(bridge_id)
    chk("cm1_exists", cm1 is not None)
    chk("cm2_exists", cm2 is not None)
    chk("bridge_exists", bridge is not None)

    if cm1 is None or cm2 is None or bridge is None:
        return out

    exp_child, exp_bridge = L + 2, L + 1
    out["info"] = {"smb.level": L, "expected child/bridge": f"{exp_child}/{exp_bridge}",
                   "child_cosine_manifest": row.get("child_cosine")}
    def lvl(d):
        try:
            return int(d["level"])
        except Exception:
            return None
    l_cm1, l_cm2, l_br = lvl(cm1), lvl(cm2), lvl(bridge)

    chk("cm1.level==L+2", l_cm1 == exp_child)
    chk("cm2.level==L+2", l_cm2 == exp_child)
    chk("cm1==cm2 level", l_cm1 == l_cm2)
    chk("bridge.level==L+1", l_br == exp_bridge)

    for label, d in (("cm1", cm1), ("cm2", cm2), ("bridge", bridge)):
        chk(f"{label}.has_vector", bool(d.get("vector")))
        # children/bridge are edges — except at the leaf end (L15 senses /
        # L14 words) where the CM lineage bottoms out in 'node' docs.
        chk(f"{label}.doc_type=edge|node", d.get("doc_type") in ("baryedge", "node"))

    # manifest cross-check
    chk("manifest_children_match", {str(row.get("cm1_id")), str(row.get("cm2_id"))}
        == {str(cm1_id), str(cm2_id)})
    chk("manifest_bridge_match", str(row.get("bridge_id")) == str(bridge_id))

    out["info"]["manifest_author"] = row.get("author")
    out["checks"]["ALL_CONSISTENT"] = ok
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural audit of built SMBs.")
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]
    struct = [r for r in rows if r.get("smb_id") and r.get("cm1_id")
              and r.get("cm2_id") and r.get("bridge_id")]
    if args.all:
        sample = struct
    else:
        sample = random.Random(args.seed).sample(struct, min(args.sample, len(struct)))

    coll = get_collection(Settings.load())

    # how many manifest rows land in the DB, and DB rows not in the manifest
    db_smb_ids = {
        str(d["_id"]) for d in coll.find(
            {"source": "structural"}, {"_id": 1}).limit(10000)}
    manifest_ids = {str(r["smb_id"]) for r in struct}

    print(f"structural rows in manifest: {len(struct)}; "
          f"DB docs with source='structural': {len(db_smb_ids)}; "
          f"manifest ids missing from DB: {len(manifest_ids - db_smb_ids)}; "
          f"DB docs not in manifest: {len(db_smb_ids - manifest_ids)}")
    print(f"auditing {len(sample)} random rows (seed {args.seed})\n")

    hdr = f"{'name':34} {'L':>3} {'smb':>6} {'cm1':>5} {'cm2':>5} {'brdg':>5} {'rule':>5} {'ids':>5} {'mr':>4}"
    print(hdr)
    print("-" * (len(hdr) - 8))
    passes = 0
    for r in sample:
        res = audit_row(coll, r)
        c = res["checks"]
        core = all(c.get(k, False) for k in
                   ("smb_doc_exists", "smb.level_is_int", "cm1.level==L+2",
                    "cm2.level==L+2", "bridge.level==L+1", "cm1_exists",
                    "cm2_exists", "bridge_exists", "ids_distinct",
                    "smb.has_explicit_bridge_id"))
        n_fails = sum(1 for k, v in c.items() if k != "ALL_CONSISTENT" and k != "smb_doc_exists" and not v)
        fails = " ".join(k for k, v in c.items() if k != "ALL_CONSISTENT" and not v)
        verdict = "OK" if core and not fails else "FAIL"
        if verdict == "OK":
            passes += 1
        print(f"{str(res['name'])[:33]:34} {res['info'].get('smb.level', '-'):>3} "
              f"{'Y' if c.get('smb_doc_exists') else 'N':>6} "
              f"{'Y' if c.get('cm1_exists') else 'N':>5} "
              f"{'Y' if c.get('cm2_exists') else 'N':>5} "
              f"{'Y' if c.get('bridge_exists') else 'N':>5} "
              f"{'Y' if core else 'N':>5} "
              f"{'Y' if c.get('ids_distinct') else 'N':>5} "
              f"{'Y' if c.get('manifest_children_match') and c.get('manifest_bridge_match') else 'N':>4}")
        if fails:
            print(f"    -> FAIL: {fails}")

    print(f"\nSTRUCTURAL PASS: {passes}/{len(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())