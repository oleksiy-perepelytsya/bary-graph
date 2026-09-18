"""Pair unparented L14 words into L14 BEs via incremental _id-window rounds.

Plan B: after s06 (kaikki-relation fermion tiers) and s07 (bounded re-entry)
only a fraction of L14 words are parented. This stage pairs the remaining
orphan words directly, word<->word, by vector cosine — so far more words get a
parent L14 BE and the L14 bridge pool expands toward n_orphans/2 edges.

User decision (2026-09-17): process the whole orphan pool incrementally, in
*consecutive _id windows*, so that same-language words are more likely paired
first (within each window the greedy match consumes same-language candidate
pairs before cross-language ones, at the same 0.70 cosine floor). Bounding the
pool per round keeps each round memory-safe and fast, and makes resume
well-defined: a window ledger records the resume _id, so a re-run starts right
where the last round stopped.

Pairing at 0.70 is used for every round — same-language AND cross-language
(leftover) rounds. There is deliberately no lower-threshold second round; the
strict floor stands (q rule unchanged: q = measured pair cosine, no denial
gate, leftovers simply stay orphan and are reported).

Edge type is decided by language equality (properties.lang):
  - same  lang pair -> edge_type "same_lang"
  - cross lang pair -> edge_type "cross_lang"
Deliberately NOT same_phenomenon: we assert relationship, not equivalence.

q = measured pair cosine (the actual similarity evidence), stored as q and
confidence; no q_seed lookup, no DOI/provenance propagation in this stage.

Mechanics: each round streams its window's orphan vectors into a disk memmap,
then greedy_unique_match over top_k_pairs (HNSW at MATCH_DIM) with
same-language pairs drained first. Under ANN_THRESHOLD rows the brute-force
path is used instead. New BEs are minted at _id >= now, i.e. inside the
[_S08_TODAY_LO, inf) bridge band s08 already loads — so they feed the L14
bridge pool for the L13 pass.
"""

from __future__ import annotations

import itertools
import json
import logging
import operator
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne
from pymongo.write_concern import WriteConcern

from lib.bary_vec import TYPE_SENTENCES, compute_bary_vec
from lib.db import get_collection
from lib.docs import baryedge
from lib.embed import OllamaEmbedder
from lib.match import greedy_unique_match, top_k_pairs
from lib.vector import unpack_vec
from scripts._base import bootstrap, finish

STAGE = "07b_pair_orphans"

# Word _id block for this build — same bounds s06/s07 use (L14 words are a
# contiguous _id band; first word 6a9a2b40…, block ends before 6a9a5300…).
S07B_WORD_LO = os.environ.get("S07B_WORD_LO", "6a9a2b400000000000000000")
S07B_WORD_HI = os.environ.get("S07B_WORD_HI", "6a9a53000000000000000000")
# Strict cosine floor for every round (user decision 2026-09-17): only
# confident pairs; no lower-threshold follow-up round is planned.
S07B_MIN_COS = float(os.environ.get("S07B_MIN_COS", "0.70"))
# Orphan words per round. Bounding the pool keeps memory low and matching
# fast; consecutive rounds advance the resume _id through the band.
S07B_WINDOW = int(os.environ.get("S07B_WINDOW", "1000000"))
# Above this many rows OV is a disk memmap instead of a dense array.
S07B_MEM_SWITCH = int(os.environ.get("S07B_MEM_SWITCH", "300000"))
S07B_MMAP_PATH = os.environ.get("S07B_MMAP_PATH", "/storage/bary/s07b_OV.mmap")

_LEDGER_NAME = "07b_rounds.json"


def _ledger_path(settings) -> Path:
    return Path(settings.pipeline_state_dir) / _LEDGER_NAME


def _load_ledger(settings) -> dict:
    p = _ledger_path(settings)
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


def _cleanup_mmap(log: logging.Logger) -> None:
    try:
        Path(S07B_MMAP_PATH).unlink(missing_ok=True)
    except Exception:
        log.warning("could not remove OV memmap file %s", S07B_MMAP_PATH)


def _split_custom_args(argv: Sequence[str] | None) -> tuple[list[str], int | None]:
    """Strip s07b-specific flags (--window) so bootstrap's parser stays happy."""
    out: list[str] = []
    window = None
    it = iter(argv or [])
    for a in it:
        if a == "--window":
            window = int(next(it))
        elif a.startswith("--window="):
            window = int(a.split("=", 1)[1])
        else:
            out.append(a)
    return out, window


def _load_window(
    coll, start_id: ObjectId, block_hi: ObjectId, window: int,
    embed_dim: int, log: logging.Logger,
) -> tuple[list, list[str], np.ndarray, ObjectId | None, int]:
    """Stream the next ``window`` orphans from ``start_id`` in _id order.

    Returns (orphan_ids, langs, OV, resume_id, examined). ``resume_id`` is the
    _id just past the last document examined (the point the next round starts
    from); ``examined`` counts docs read so the caller can log skip ratio.
    """
    # Size the backing store upfront; rows map 1:1 to collected orphans.
    big = window >= S07B_MEM_SWITCH
    if big:
        mmap_path = Path(S07B_MMAP_PATH)
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


def run(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        import sys
        argv = sys.argv[1:]
    clean, window_opt = _split_custom_args(argv)
    settings, args, log, cp = bootstrap(STAGE, clean)
    floor = S07B_MIN_COS
    window = window_opt if window_opt is not None else S07B_WINDOW
    budget = args.limit  # dev cap on TOTAL orphans across all rounds
    coll = get_collection(settings)
    log.info("start processed=%d dry_run=%s min_cos=%.2f window=%d limit=%s",
             cp.processed, args.dry_run, floor, window, budget)
    if window < 2:
        log.warning("window < 2 (%d) — nothing to pair", window)
        cp.processed = 0
        cp.total = 0
        if not args.dry_run:
            finish(cp, settings, log)
        return

    # Type-sentence vectors embedded once (build-all env uses the bare
    # embedder because the 413 GB cache file can't be mmapped comfortably).
    embedder = OllamaEmbedder(settings)
    tkeys = ["same_lang", "cross_lang"]
    tvecs = embedder.embed([TYPE_SENTENCES[k] for k in tkeys])
    type_vec: dict[str, np.ndarray] = dict(zip(tkeys, tvecs, strict=True))

    block_lo = ObjectId(S07B_WORD_LO)
    block_hi = ObjectId(S07B_WORD_HI)
    # Write tuning (benchmarked 2026-09-18 on barygraph_all wire shapes):
    # batch 2900 + unacknowledged parent stamps ≈ 2× baseline (378 vs 191 BE/s
    # on a throwaway coll with cold-ish cache; live ≈57/s at 2048+ack). The
    # BE insert_many stays acknowledged — the stamps are idempotent
    # bookkeeping, so losing one on crash only leaves the word re-parentable.
    batch_n = args.batch_size or int(os.environ.get("S07B_WRITE_BATCH", "2900"))
    stamps_unack = os.environ.get("S07B_STAMPS_UNACK", "1") not in ("0", "false")

    ledger = _load_ledger(settings)
    # Resume: last round's bookmark (unless --reset/--force on the whole run).
    if args.reset:
        resume = block_lo
    elif ledger.get("resume"):
        resume = ObjectId(ledger["resume"])
    else:
        resume = block_lo
    log.info("resuming from %s", resume)

    n_rounds = 0
    n_total = 0
    n_words_pooled = 0
    now = datetime.now(timezone.utc)

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
            _cleanup_mmap(log)
            break

        # Strict greedy mutual-cosine matching, same-language pairs first.
        raw = list(top_k_pairs(OV, min_score=floor))
        raw.sort(key=operator.itemgetter(2), reverse=True)
        same = [p for p in raw if langs[p[0]] == langs[p[1]]]
        cross = [p for p in raw if langs[p[0]] != langs[p[1]]]
        pairs = greedy_unique_match(
            itertools.chain(same, cross), threshold=floor
        )
        log.info(
            "round match: %d pairs (floor %.2f, same_lang-first), "
            "%d orphans left unpaired in this window",
            len(pairs), floor, n - 2 * len(pairs),
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
            ledger["min_cos"] = floor
            _save_ledger(settings, ledger)
        log.info("round %d done: %d BEs written (cum %d)", n_rounds, n_written, n_total)

        OV = None
        _cleanup_mmap(log)

        if not next_resume or next_resume >= block_hi:
            log.info("band exhausted (%s)", next_resume)
            resume = None
        else:
            resume = next_resume
            if budget is not None and n_words_pooled >= budget:
                log.info("--limit %d reached (pooled %d) — stopping", budget, n_words_pooled)
                resume = None

    unpaired = n_words_pooled - 2 * n_total
    log.info("stage: %d rounds, %d BEs written, ~%d words pooled, "
             "~%d words still unpaired",
             n_rounds, n_total, n_words_pooled, unpaired)
    cp.processed = n_total
    cp.total = n_total
    _cleanup_mmap(log)
    if not args.dry_run:
        finish(cp, settings, log)


def _flush(coll, docs: list[dict], pair_idxs: list[tuple[int, int]],
           orphan_ids: list, now, stamps_unack: bool = True) -> int:
    """Insert one batch of BEs and stamp both CM words' parent_edge_id.

    This stage deliberately performs NO DOI propagation — the pair edges are
    inferred from vector similarity, not provenance chains, and the per-edge
    reverse lookups were the throughput bottleneck.

    The BE insert_many is always acknowledged (the BEs are the durable
    payload). The word parent stamps are idempotent bookkeeping (a loss on
    crash leaves the word unparented and re-pair-able by the next window), so
    they run with ``w=0`` by default: ~0.2s vs ~7s per 2048-BE batch on the
    bench, and the round ledger is the source of truth for resumption either
    way.
    """
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


if __name__ == "__main__":
    run()
