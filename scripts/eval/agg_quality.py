"""Aggregation-quality: does the schema cluster the academic terms correctly?

Scope: barygraph_poc BaryEdge/MetaBary docs registered in doi_bridges
(the academic subgraph). Per level (15..10) reports:
  - n docs with DOI provenance, mean distinct-DOI richness
  - trivial-cluster share (single-DOI clusters)
  - z of observed richness vs a slot-permutation shuffle (random grouping)
  - mean pairwise cosine among {doc, cm1, cm2} vectors (coherence)

Also stratified by modal use-case per level (10..13) where n >= 20.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import bson
import numpy as np

from lib.config import Settings
from lib.db import get_client
from scripts.eval.academic_common import (
    doi_use_case_map,
    load_rows,
    require_poc,
    reverse_doi_index,
    to_str,
)

LEVELS = [15, 14, 13, 12, 11, 10]
SHUFFLE_TRIALS = 25
COHERENCE_SAMPLE = 300
MIN_N_FOR_Z = 50


def main() -> None:
    settings = Settings.load()
    require_poc(settings)
    client = get_client(settings)
    bridge = client[settings.mongo_db][settings.mongo_doi_bridges_collection]
    coll = client[settings.mongo_db][settings.mongo_collection]

    doi_uc = doi_use_case_map(load_rows())
    reverse = reverse_doi_index(bridge)

    dois_by_node: dict[str, list[str]] = defaultdict(list)
    for nid, dois in reverse.items():
        for doi in dois:
            dois_by_node[nid].append(doi)

    all_ids = sorted(dois_by_node)
    info: dict[str, dict] = {}
    for i in range(0, len(all_ids), 1000):
        ids = [bson.ObjectId(s) for s in all_ids[i : i + 1000]]
        for doc in coll.find(
            {"_id": {"$in": ids}}, {"level": 1, "cm1_id": 1, "cm2_id": 1, "doc_type": 1}
        ):
            if doc.get("doc_type") != "baryedge":
                continue
            oid = to_str(doc["_id"])
            if doc.get("level") not in LEVELS:
                continue
            info[oid] = {
                "level": doc["level"],
                "dois": dois_by_node[oid],
                "cm1": to_str(doc.get("cm1_id")) if doc.get("cm1_id") else None,
                "cm2": to_str(doc.get("cm2_id")) if doc.get("cm2_id") else None,
            }

    by_level: dict[int, list[str]] = defaultdict(list)
    for oid, d in info.items():
        by_level[d["level"]].append(oid)

    def shuffle_stats(ids: list[str]) -> tuple[float, float] | None:
        if len(ids) < MIN_N_FOR_Z:
            return None
        idx = {sid: i for i, sid in enumerate(ids)}
        slots: list[int] = []
        doi_slots: list[str] = []
        for sid in ids:
            for doi in info[sid]["dois"]:
                slots.append(idx[sid])
                doi_slots.append(doi)
        slots_arr = np.asarray(slots, dtype=np.int64)
        doi_arr = np.asarray(doi_slots, dtype=object)
        order = np.argsort(slots_arr, kind="stable")
        srt = slots_arr[order]
        starts = np.r_[0, np.nonzero(np.diff(srt))[0] + 1, len(srt)]
        bounds = list(zip(starts[:-1], starts[1:], strict=False))
        rng = np.random.default_rng(0)
        trial_means = []
        for _ in range(SHUFFLE_TRIALS):
            perm = doi_arr[rng.permutation(len(doi_arr))][order]
            distincts = [len(set(perm[a:b])) for a, b in bounds]
            trial_means.append(float(np.mean(distincts)))
        mu = float(np.mean(trial_means))
        sigma = float(np.std(trial_means) or 1e-9)
        return mu, sigma

    def coherence_mean(ids: list[str]) -> float | None:
        sample = ids[:COHERENCE_SAMPLE]
        todo = {sid for sid in sample}
        for sid in sample:
            if info[sid]["cm1"]:
                todo.add(info[sid]["cm1"])
            if info[sid]["cm2"]:
                todo.add(info[sid]["cm2"])
        ids_todo = [bson.ObjectId(t) for t in todo if t]
        vecs: dict[str, np.ndarray] = {}
        for i in range(0, len(ids_todo), 1000):
            for d in coll.find(
                {"_id": {"$in": ids_todo[i : i + 1000]}}, {"vector": 1, "type_vector": 1}
            ):
                v = d.get("vector") or d.get("type_vector")
                if v is not None:
                    vecs[to_str(d["_id"])] = np.asarray(v, dtype=np.float32)
        sims: list[float] = []
        for sid in sample:
            c1 = vecs.get(info[sid]["cm1"])
            c2 = vecs.get(info[sid]["cm2"])
            if c1 is not None and c2 is not None:
                sims.append(float(np.dot(c1, c2)))
            a = vecs.get(sid)
            if a is not None and c1 is not None:
                sims.append(float(np.dot(a, c1)))
            if a is not None and c2 is not None:
                sims.append(float(np.dot(a, c2)))
        return float(np.mean(sims)) if sims else None

    def _f(x):
        return round(float(x), 3) if x is not None else ""

    rows: list[list] = []
    for level in LEVELS:
        ids = by_level[level]
        n = len(ids)
        if not n:
            rows.append(["all", level, 0, "", "", "", ""])
            continue
        ndois = [len(info[sid]["dois"]) for sid in ids]
        obs = float(np.mean(ndois))
        trivial = float(np.mean([x == 1 for x in ndois]))
        ss = shuffle_stats(ids)
        z = (obs - ss[0]) / ss[1] if ss else None
        coh = coherence_mean(ids)
        rows.append(["all", level, n, _f(obs), _f(trivial), _f(z), _f(coh)])

    for level in (13, 12, 11, 10):
        ids = by_level[level]
        modal: dict[str, list[str]] = defaultdict(list)
        for sid in ids:
            us = set()
            for d in info[sid]["dois"]:
                us.update(doi_uc.get(d, ()))
            if us:
                modal[
                    max(
                        us,
                        key=lambda u: sum(1 for d in info[sid]["dois"] if u in doi_uc.get(d, ())),
                    )
                ].append(sid)
        for uc, uids in modal.items():
            if len(uids) < 20:
                continue
            ndois = [len(info[sid]["dois"]) for sid in uids]
            obs = float(np.mean(ndois))
            trivial = float(np.mean([x == 1 for x in ndois]))
            ss = shuffle_stats(uids)
            z = (obs - ss[0]) / ss[1] if ss else None
            coh = coherence_mean(uids)
            rows.append([uc, level, len(uids), _f(obs), _f(trivial), _f(z), _f(coh)])

    out_path = Path("evaluation/results/aggregation_quality.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scope",
                "level",
                "n_docs",
                "mean_distinct_dois",
                "trivial_share",
                "z_vs_shuffle",
                "coherence",
            ]
        )
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
