"""Absorb orphan L14 word CMs into L14 BEs (corpus-agnostic).

Two phases, run by default in sequence:

Phase ``pair`` — pair unparented L14 words into new L14 BEs via incremental
_id-window rounds (formerly s07b_pair_orphans). Strict cosine floor
(``S07_MIN_COS``, default 0.70), same-language pairs drained first within
each window. Edge type is ``same_lang`` / ``cross_lang`` by ``properties.lang``.
q = measured pair cosine; no DOI/provenance propagation.

Phase ``absorb`` — absorb every remaining orphan word into its nearest
existing L14 BE with no similarity floor (formerly s07_orphan_reentry), so
every word node ends up parented. The new BE inherits ``edge_type`` /
``type_vector`` / ``q`` from that partner (no new embedding call).

Corpus-agnostic: the L14 word-node _id band is discovered from the database
when ``S07_WORD_LO``/``S07_WORD_HI`` are not set, and the L14 BE pool band
defaults to the whole collection (``S07_BE_LO``/``S07_BE_HI`` override).

Write tuning: acknowledged BE inserts + w=0 parent stamps, batch
``S07_WRITE_BATCH`` (default 2900). Resume bookmark in
``pipeline_state_dir/07_orphans.json`` (legacy ``07b_rounds.json`` is read
if the new file is missing).
"""

from __future__ import annotations

import itertools
import json
import logging
import operator
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne
from pymongo.write_concern import WriteConcern

from lib.bary_vec import TYPE_SENTENCES, compute_bary_vec
from lib.config import scratch_dir
from lib.db import get_collection
from lib.docs import baryedge
from lib.embed import get_embedder
from lib.match import (
    ANN_EF_CONSTRUCTION,
    ANN_M,
    ANN_THRESHOLD,
    MATCH_DIM,
    _gaussian_projector,
    greedy_unique_match,
    top_k_pairs,
)
from lib.vector import unpack_vec
from scripts._base import bootstrap, finish

_log = logging.getLogger(__name__)

STAGE = "07_orphan_reentry"

# --- Config (env-overridable; empty = discover from the build's own data) ---
S07_WORD_LO = os.environ.get("S07_WORD_LO", "")
S07_WORD_HI = os.environ.get("S07_WORD_HI", "")
S07_BE_LO = os.environ.get("S07_BE_LO", "")
S07_BE_HI = os.environ.get("S07_BE_HI", "")
S07_MIN_COS = float(os.environ.get("S07_MIN_COS", os.environ.get("S07B_MIN_COS", "0.70")))
S07_WINDOW = int(os.environ.get("S07_WINDOW", os.environ.get("S07B_WINDOW", "1000000")))
S07_MEM_SWITCH = int(os.environ.get("S07_MEM_SWITCH", os.environ.get("S07B_MEM_SWITCH", "300000")))
S07_MMAP_PATH = os.environ.get("S07_MMAP_PATH") or str(scratch_dir() / "s07_OV.mmap")
S07_BEV_PATH = os.environ.get("S07_BEV_PATH") or str(scratch_dir() / "s07_be_pool.bin")
S07_ORPHAN_LIMIT = int(os.environ.get("S07_ORPHAN_LIMIT", "1000000"))
S07_LOAD_WORKERS = int(os.environ.get("S07_LOAD_WORKERS", "8"))
S07_WRITE_BATCH = int(os.environ.get("S07_WRITE_BATCH", "2900"))
S07_STAMPS_UNACK = os.environ.get(
    "S07_STAMPS_UNACK", os.environ.get("S07B_STAMPS_UNACK", "1")
) not in ("0", "false")
_S07_NN_EF = 400

_LEDGER_NAME = "07_orphans.json"
_LEGACY_LEDGER_NAME = "07b_rounds.json"


def _id_int(oid: ObjectId) -> int:
    return int.from_bytes(oid.binary, "big")


def _split_ranges(lo: ObjectId, hi: ObjectId, n: int) -> list[tuple[ObjectId, ObjectId]]:
    lo_i = _id_int(lo)
    hi_i = _id_int(hi)
    span = (hi_i - lo_i) // n
    out = []
    for k in range(n):
        a = lo_i + k * span
        b = lo_i + (k + 1) * span if k < n - 1 else hi_i
        out.append((ObjectId(a.to_bytes(12, "big")),
                    ObjectId(b.to_bytes(12, "big"))))
    return out


def _word_band(coll) -> tuple[ObjectId | None, ObjectId | None]:
    """Derive the L14 word-node _id band from the DB when env vars are unset."""
    q = {"doc_type": "node", "node_type": "word", "level": 14}
    lo = coll.find_one(q, sort=[("_id", 1)], projection={"_id": 1})
    hi = coll.find_one(q, sort=[("_id", -1)], projection={"_id": 1})
    return (lo["_id"] if lo else None, hi["_id"] if hi else None)


# ---------------------------------------------------------------- ledger -----

def _ledger_path(settings) -> Path:
    return Path(settings.pipeline_state_dir) / _LEDGER_NAME


def _load_ledger(settings) -> dict:
    p = _ledger_path(settings)
    if not p.exists():
        legacy = p.parent / _LEGACY_LEDGER_NAME
        p = legacy if legacy.exists() else p
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_ledger(settings, ledger: dict) -> None:
    p = _ledger_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    tmp.replace(p)


# ------------------------------------------------------------ pair phase -----

def _load_window(
    coll, start_id: ObjectId, block_hi: ObjectId, window: int,
    embed_dim: int, log: logging.Logger,
) -> tuple[list, list[str], np.ndarray, ObjectId | None, int]:
    """Stream the next ``window`` orphans from ``start_id`` in _id order."""
    big = window >= S07_MEM_SWITCH
    mmap_path = Path(S07_MMAP_PATH)
    if big:
        mmap_path.parent.mkdir(parents=True, exist_ok=True)
        OV = np.memmap(mmap_path, mode="w+", dtype=np.float32,
                       shape=(window, embed_dim))
    else:
        OV = np.zeros((window, embed_dim), dtype=np.float32)

    orphan_ids: list = []
    langs: list[str] = []
    examined = 0
    n_collected = 0
    resume: ObjectId | None = None
    cur = coll.find(
        {"doc_type": "node", "node_type": "word", "level": 14,
         "_id": {"$gte": start_id, "$lt": block_hi}},
        {"_id": 1, "vector": 1, "parent_edge_id": 1, "properties.lang": 1},
    ).sort("_id", 1).hint("_id_")
    try:
        for doc in cur:
            examined += 1
            resume = doc["_id"]
            if doc.get("parent_edge_id") is not None or doc.get("vector") is None:
                continue
            orphan_ids.append(doc["_id"])
            langs.append(doc.get("properties", {}).get("lang", ""))
            OV[n_collected] = unpack_vec(doc["vector"])
            n_collected += 1
            if n_collected >= window:
                break
    finally:
        cur.close()

    n = n_collected
    OV = OV[:n]
    log.info(
        "window from %s: %d orphans (examined %d docs, %.0f%% skip)",
        start_id, n, examined,
        100.0 * (examined - n) / examined if examined else 0.0,
    )
    return orphan_ids, langs, OV, resume, examined


def _flush(coll, docs: list[dict], pair_idxs: list[tuple[int, int]],
           orphan_ids: list, now, stamps_unack: bool = True) -> int:
    """Insert one batch of BEs and stamp both CM words' parent_edge_id."""
    res = coll.insert_many(docs)
    ups: list[UpdateOne] = []
    for (i, j), eid in zip(pair_idxs, res.inserted_ids, strict=True):
        ups.append(UpdateOne({"_id": orphan_ids[i]},
                             {"$set": {"parent_edge_id": eid, "updated_at": now}}))
        ups.append(UpdateOne({"_id": orphan_ids[j]},
                             {"$set": {"parent_edge_id": eid, "updated_at": now}}))
    target = coll
    if stamps_unack:
        target = coll.with_options(write_concern=WriteConcern(w=0))
    target.bulk_write(ups, ordered=False)
    return len(docs)


def _phase_pair(settings, args, log, coll, block_lo, block_hi, window,
                batch_n, stamps_unack, now) -> int:
    """Strict-floor word↔word pairing rounds. Returns BEs written."""
    embedder = get_embedder(settings)
    tkeys = ["same_lang", "cross_lang"]
    tvecs = embedder.embed([TYPE_SENTENCES[k] for k in tkeys])
    type_vec: dict[str, np.ndarray] = dict(zip(tkeys, tvecs, strict=True))

    ledger = _load_ledger(settings)
    if args.reset:
        resume = block_lo
    elif ledger.get("resume"):
        resume = ObjectId(ledger["resume"])
    else:
        resume = block_lo
    log.info("pair phase: resuming from %s", resume)

    budget = args.limit  # dev cap on TOTAL orphans across all rounds
    n_rounds = 0
    n_total = 0
    n_words_pooled = 0

    while resume is not None:
        round_window = window
        if budget is not None:
            round_window = min(window, budget - n_words_pooled)
        orphan_ids, langs, OV, next_resume, examined = _load_window(
            coll, resume, block_hi, round_window, settings.embed_dim, log
        )
        n = len(orphan_ids)
        if n < 2:
            log.info("window yielded <2 orphans (%d) — parched; stopping", n)
            OV = None
            Path(S07_MMAP_PATH).unlink(missing_ok=True)
            break

        raw = list(top_k_pairs(OV, min_score=S07_MIN_COS))
        raw.sort(key=operator.itemgetter(2), reverse=True)
        same = [p for p in raw if langs[p[0]] == langs[p[1]]]
        cross = [p for p in raw if langs[p[0]] != langs[p[1]]]
        pairs = greedy_unique_match(
            itertools.chain(same, cross), threshold=S07_MIN_COS
        )
        log.info(
            "round match: %d pairs (floor %.2f, same_lang-first), "
            "%d orphans left unpaired in this window",
            len(pairs), S07_MIN_COS, n - 2 * len(pairs),
        )
        raw = same = cross = None

        n_written = 0
        if not args.dry_run and pairs:
            docs: list[dict] = []
            pair_idxs: list[tuple[int, int]] = []
            for i, j, q_cos in pairs:
                et = "same_lang" if langs[i] == langs[j] else "cross_lang"
                tv = type_vec[et]
                bv = compute_bary_vec(OV[i], OV[j], tv, q_cos)
                docs.append(baryedge(orphan_ids[i], orphan_ids[j], 14, bv,
                                     q_cos, accumulated_weight=q_cos,
                                     edge_type=et, type_vector=tv,
                                     source="inferred", confidence=q_cos))
                pair_idxs.append((i, j))
                if len(docs) >= batch_n:
                    n_written += _flush(coll, docs, pair_idxs, orphan_ids, now,
                                        stamps_unack)
                    docs = []
                    pair_idxs = []
                    log.info("  inserted %d BEs this round", n_written)
            if docs:
                n_written += _flush(coll, docs, pair_idxs, orphan_ids, now,
                                    stamps_unack)
        else:
            n_written = len(pairs)

        n_rounds += 1
        n_total += n_written
        n_words_pooled += n
        if not args.dry_run:
            ledger[f"round_{n_rounds}"] = {
                "words": n,
                "pairs": n_written,
                "examined": examined,
                "at": datetime.now(timezone.utc).isoformat(),
            }
            ledger["resume"] = str(next_resume)
            ledger["min_cos"] = S07_MIN_COS
            _save_ledger(settings, ledger)
        log.info("round %d done: %d BEs written (cum %d)", n_rounds, n_written, n_total)

        OV = None
        Path(S07_MMAP_PATH).unlink(missing_ok=True)

        if not next_resume or next_resume >= block_hi:
            log.info("band exhausted (%s)", next_resume)
            resume = None
        else:
            resume = next_resume
            if budget is not None and n_words_pooled >= budget:
                log.info("--limit %d reached (pooled %d) — stopping", budget, n_words_pooled)
                resume = None

    unpaired = n_words_pooled - 2 * n_total
    log.info("pair phase: %d rounds, %d BEs written, ~%d words pooled, "
             "~%d words still unpaired",
             n_rounds, n_total, n_words_pooled, unpaired)
    return n_total


# ----------------------------------------------------------- absorb phase ----

def _load_orphans(coll, limit: int, embed_dim: int, OV, OVP, proj,
                  block_lo: ObjectId, block_hi: ObjectId,
                  log: logging.Logger) -> tuple[list, int]:
    """Stream up to ``limit`` unparented L14 words into OV/OVP memmaps."""
    ids: list = []
    lock = threading.Lock()
    n = 0
    skipped = 0
    fields = {"_id": 1, "vector": 1, "parent_edge_id": 1,
              "doc_type": 1, "node_type": 1, "level": 1}

    def _scan(lo, hi):
        nonlocal n, skipped
        q = {"_id": {"$gte": lo, "$lt": hi}}
        for doc in coll.find(q, fields).sort("_id", 1).hint("_id_").batch_size(10_000):
            if doc.get("doc_type") != "node" or doc.get("node_type") != "word" \
                    or doc.get("level") != 14 or doc.get("vector") is None \
                    or doc.get("parent_edge_id") is not None:
                skipped += 1
                continue
            row = unpack_vec(doc["vector"])
            rp = row @ proj.T
            norm = float(np.linalg.norm(rp))
            rp = rp / norm if norm else rp
            with lock:
                if n >= limit:
                    break
                ids.append(doc["_id"])
                OV[n] = row
                OVP[n] = rp
                n += 1
                if n % 500_000 == 0:
                    log.info("  %d orphans loaded", n)

    ranges = _split_ranges(block_lo, block_hi, S07_LOAD_WORKERS)
    with ThreadPoolExecutor(max_workers=S07_LOAD_WORKERS) as ex:
        list(ex.map(lambda r: _scan(*r), ranges))
    return ids, skipped


def _stream_be_pool(coll, embed_dim: int, proj, out_path: Path,
                    log: logging.Logger) -> list:
    """Stream the L14 BE pool as projected-normalised rows into a binary file."""
    if S07_BE_LO:
        lo = ObjectId(S07_BE_LO)
    else:
        lo = ObjectId("000000000000000000000000")
    if S07_BE_HI:
        hi = ObjectId(S07_BE_HI)
    else:
        max_id = coll.find_one(sort=[("_id", -1)], projection={"_id": 1})
        hi = max_id["_id"] if max_id else None
    if hi is None or _id_int(hi) <= _id_int(lo):
        return []
    ids: list = []
    lock = threading.Lock()
    n = 0
    fields = {"_id": 1, "vector": 1, "level": 1, "source": 1, "doc_type": 1}

    log.info("streaming L14 BE pool band [%s, %s)", lo, hi)
    ranges = _split_ranges(lo, hi, S07_LOAD_WORKERS)
    with open(out_path, "wb") as f:
        def _scan(a, b):
            nonlocal n
            q = {"_id": {"$gte": a, "$lt": b}}
            for doc in coll.find(q, fields).sort("_id", 1).hint("_id_").batch_size(10_000):
                if doc.get("doc_type") != "baryedge" or doc.get("level") != 14 \
                        or doc.get("vector") is None or doc.get("source") == "structural":
                    continue
                row = unpack_vec(doc["vector"])
                rp = (row @ proj.T).astype(np.float32)  # float32! BVP is
                # reopened as float32 memmap; float64 bytes corrupted it
                norm = float(np.linalg.norm(rp))
                rp = rp / norm if norm else rp
                assert rp.dtype == np.float32
                with lock:
                    ids.append(doc["_id"])
                    f.write(rp.tobytes())
                    n += 1
                    if n % 500_000 == 0:
                        log.info("  %d L14 BEs in pool", n)

        with ThreadPoolExecutor(max_workers=S07_LOAD_WORKERS) as ex:
            list(ex.map(lambda r: _scan(*r), ranges))
    return ids


def _phase_absorb(settings, args, log, coll, block_lo, block_hi, now) -> int:
    """Absorb remaining orphans into nearest existing L14 BE (no floor)."""
    limit = args.limit if args.limit is not None else S07_ORPHAN_LIMIT
    if limit < 1:
        log.info("orphan limit < 1 (%d) — nothing to do", limit)
        return 0

    embed_dim = settings.embed_dim
    proj = _gaussian_projector(embed_dim)

    ov_path = Path(S07_MMAP_PATH)
    ovp_path = Path(f"{S07_MMAP_PATH}_P")
    bev_path = Path(S07_BEV_PATH)
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (ov_path, ovp_path, bev_path):
        p.unlink(missing_ok=True)

    try:
        OV = np.memmap(ov_path, mode="w+", dtype=np.float32,
                       shape=(limit, embed_dim))
        OVP = np.memmap(ovp_path, mode="w+", dtype=np.float32,
                        shape=(limit, MATCH_DIM))
        orphan_ids, n_skipped = _load_orphans(
            coll, limit, embed_dim, OV, OVP, proj, block_lo, block_hi, log)
        n_orphans = len(orphan_ids)
        if n_orphans == 0:
            log.info("no L14 orphans found (skipped=%d)", n_skipped)
            return 0
        OV = OV[:n_orphans]
        OVP = OVP[:n_orphans]
        log.info("L14 orphans=%d (skipped=%d)", n_orphans, n_skipped)

        be_ids = _stream_be_pool(coll, embed_dim, proj, bev_path, log)
        n_bes = len(be_ids)
        if n_bes == 0:
            log.info("no L14 BE pool — nothing to absorb into")
            return 0
        log.info("L14 BE pool=%d", n_bes)

        if n_bes <= ANN_THRESHOLD:
            BVP = np.memmap(bev_path, dtype=np.float32, mode="r",
                            shape=(n_bes, MATCH_DIM))
            CHUNK = 1024
            best_bi = np.empty(n_orphans, dtype=np.int64)
            for start in range(0, n_orphans, CHUNK):
                end = min(start + CHUNK, n_orphans)
                best_bi[start:end] = np.argmax(
                    OVP[start:end].copy() @ BVP.T, axis=1)
            del BVP
        else:
            try:
                import hnswlib
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("hnswlib required for ANN nearest-BE search") from e
            BVP = np.memmap(bev_path, dtype=np.float32, mode="r",
                            shape=(n_bes, MATCH_DIM))
            index = hnswlib.Index(space="cosine", dim=MATCH_DIM)
            index.init_index(max_elements=max(n_bes, 16),
                             ef_construction=ANN_EF_CONSTRUCTION, M=ANN_M)
            log.info("building HNSW on %d BEs (ef_c=%d M=%d)",
                     n_bes, ANN_EF_CONSTRUCTION, ANN_M)
            index.add_items(BVP)
            log.info("HNSW built")
            del BVP
            bev_path.unlink(missing_ok=True)
            index.set_ef(_S07_NN_EF)
            best_bi = np.empty(n_orphans, dtype=np.int64)
            for start in range(0, n_orphans, 16_384):
                end = min(start + 16_384, n_orphans)
                labs, _ = index.knn_query(
                    np.ascontiguousarray(OVP[start:end]), k=1)
                best_bi[start:end] = labs[:, 0]
            del index
        del OVP

        bev_path.unlink(missing_ok=True)

        used_be_ids = [be_ids[bi] for bi in sorted(set(best_bi.tolist()))]
        log.info("winning partners=%d", len(used_be_ids))
        be_meta: dict = {}
        _FIELDS = {"vector": 1, "edge_type": 1, "type_vector": 1,
                   "q": 1, "accumulated_weight": 1}
        for s in range(0, len(used_be_ids), 50_000):
            chunk = used_be_ids[s:s + 50_000]
            be_meta.update(
                {doc["_id"]: doc
                 for doc in coll.find({"_id": {"$in": chunk}}, _FIELDS)}
            )
        pvec = {pid: unpack_vec(meta["vector"]) for pid, meta in be_meta.items()
                if meta.get("vector") is not None}

        batch_n = args.batch_size or S07_WRITE_BATCH
        n_written = 0
        if not args.dry_run:
            stamps_coll = coll.with_options(
                write_concern=WriteConcern(w=0)) if S07_STAMPS_UNACK else coll
            batch_docs: list[dict] = []
            batch_oids: list = []
            for idx in range(n_orphans):
                oid = orphan_ids[idx]
                bi = int(best_bi[idx])
                partner = be_meta[be_ids[bi]]
                tv = unpack_vec(partner["type_vector"])
                q = float(partner["q"])
                acc_w = float(partner.get("accumulated_weight", q))
                bv = compute_bary_vec(OV[idx], pvec[be_ids[bi]], tv, q)
                batch_docs.append(
                    baryedge(oid, be_ids[bi], 14, bv, q,
                             accumulated_weight=acc_w,
                             edge_type=partner.get("edge_type"), type_vector=tv,
                             source="inferred", confidence=q)
                )
                batch_oids.append(oid)
                if len(batch_docs) >= batch_n:
                    res = coll.insert_many(batch_docs)
                    ups = [UpdateOne({"_id": o},
                                     {"$set": {"parent_edge_id": e, "updated_at": now}})
                           for o, e in zip(batch_oids, res.inserted_ids, strict=True)]
                    stamps_coll.bulk_write(ups, ordered=False)
                    n_written += len(batch_docs)
                    log.info("  written %d (%d/%d)", n_written, idx + 1, n_orphans)
                    batch_docs = []
                    batch_oids = []
            if batch_docs:
                res = coll.insert_many(batch_docs)
                ups = [UpdateOne({"_id": o},
                                 {"$set": {"parent_edge_id": e, "updated_at": now}})
                       for o, e in zip(batch_oids, res.inserted_ids, strict=True)]
                stamps_coll.bulk_write(ups, ordered=False)
                n_written += len(batch_docs)

        if args.dry_run:
            log.info("dry-run: would absorb %d orphan words into %d existing BEs",
                     n_orphans, n_bes)
            n_written = n_orphans
        else:
            log.info("absorbed %d orphan words into %d existing BEs",
                     n_written, n_bes)
        return n_written
    finally:
        for p in (ov_path, ovp_path, bev_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                log.warning("could not remove memmap file %s", p)


# ------------------------------------------------------------------- run -----

def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    window = args.window if args.window is not None else S07_WINDOW
    phase = args.phase
    now = datetime.now(timezone.utc)

    if S07_WORD_LO:
        block_lo = ObjectId(S07_WORD_LO)
    else:
        block_lo = None
    if S07_WORD_HI:
        block_hi = ObjectId(S07_WORD_HI)
    else:
        block_hi = None
    if block_lo is None or block_hi is None:
        d_lo, d_hi = _word_band(coll)
        block_lo = block_lo or d_lo
        block_hi = block_hi or (
            ObjectId((_id_int(d_hi) + 1).to_bytes(12, "big")) if d_hi else None)
    if block_lo is None or block_hi is None or _id_int(block_hi) <= _id_int(block_lo):
        log.info("no L14 word nodes found — nothing to do")
        cp.processed = 0
        cp.total = 0
        if not args.dry_run:
            finish(cp, settings, log)
        return
    log.info("word band [%s, %s)", block_lo, block_hi)

    if window < 2:
        log.warning("window < 2 (%d) — nothing to pair", window)
        cp.processed = 0
        cp.total = 0
        if not args.dry_run:
            finish(cp, settings, log)
        return

    n_written = 0
    if phase in ("pair", "both"):
        n_written += _phase_pair(
            settings, args, log, coll, block_lo, block_hi, window,
            args.batch_size or S07_WRITE_BATCH, S07_STAMPS_UNACK, now)
    if phase in ("absorb", "both"):
        n_written += _phase_absorb(settings, args, log, coll, block_lo, block_hi, now)

    cp.processed = n_written
    cp.total = n_written
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
