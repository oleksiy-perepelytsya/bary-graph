"""Kaikki-relation-driven L14 BaryEdge formation (fermion order, v0.4 §7.1).

Iterates the six fermion tiers in priority order. Within each tier every
word's kaikki relations of the tier's kind(s) are considered; words that
already carry a ``parent_edge_id`` are skipped (unique-parent invariant).

``v(type) = embed(TYPE_SENTENCES[edge_type])`` — embedded once per
edge_type, not per pair.

Safeguard: refuses to run if any L14 BaryEdge already exists.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne

from lib import checkpoint as cp_mod
from lib import doi_bridge
from lib.bary_vec import TYPE_SENTENCES, compute_bary_vec
from lib.db import get_collection
from lib.docs import baryedge
from lib.embed import OllamaEmbedder
from lib.match import FERMION_TIERS
from lib.vector import unpack_vec
from scripts._base import bootstrap, finish

STAGE = "06_l14_edges"

# V (n_words x embed_dim) is ~156 GB at 4096-dim for the full build — far too
# large to hold resident alongside the pairing working set. Back it by a disk
# memmap in /storage (RAID, ~6 TB free) so only touched rows page into RAM;
# the file is removed when the stage finishes. Mirrors s04's V handling.
S06_MMAP_PATH = os.environ.get("S06_MMAP_PATH", "/storage/bary/s06_V.mmap")


def _cleanup_mmap(log) -> None:
    """Release + remove the disk-backed V file (~156 GB)."""
    try:
        Path(S06_MMAP_PATH).unlink(missing_ok=True)
    except Exception:
        log.warning("could not remove V memmap file %s", S06_MMAP_PATH)


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    bridge_coll = doi_bridge.get_bridge_collection(settings)
    log.info("start processed=%d dry_run=%s", cp.processed, args.dry_run)

    if not args.dry_run and not args.force:
        if coll.count_documents({"doc_type": "baryedge", "level": 14}, limit=1):
            raise RuntimeError(
                "L14 baryedges already present — re-running would double-parent words. "
                "Drop them and use --reset, or pass --force."
            )

    # Load all L14 word nodes (stage 05 must have run). The dead zero-sense
    # word (if any) carries no vector — skip it client-side rather than filter
    # on vector $exists/$ne (unindexable full scan that stalls under load).
    # Size V from the pure (indexed) word count.
    _WORD_FILTER = {"doc_type": "node", "node_type": "word", "level": 14}
    _MAX_WORDS = args.limit or coll.count_documents(_WORD_FILTER)

    # Loader. A natural fat-projection scan COLLSCANs the whole collection and
    # stalls on a >120 s getMore; _id $in detail windows are ~5k random seeks
    # (measured ~2.5k/s); a single fat _id-range cursor is capped by mongod's
    # doc-fetch throughput (~3k/s) — all too slow for 10.25M words. mongot's
    # planner ignores the compound cover index entirely (s05's identical
    # wedge), so the one fast primitive is the pure _id index. s03 inserts
    # senses then words, so this build's word nodes sit in a contiguous _id
    # block (verified: first word 6a9a2b40…, L15 baryedges start 6a9aefaf…).
    # Split that range across N parallel fat-index cursors (disjoint _id
    # sub-ranges, one get_collection client shared across threads): the
    # aggregate doc-fetch throughput scales ~2.5× over a single cursor. Each
    # worker appends (id, properties) and writes its V row under one lock, so
    # ids[i] ↔ words[i] ↔ V[i] stay consistent regardless of completion order.
    # Non-words / dead (vector-less) words are filtered client-side. Bounds and
    # worker count are env-tunable.
    block_lo = ObjectId(os.environ.get("S06_WORD_LO", "6a9a2b400000000000000000"))
    block_hi = ObjectId(os.environ.get("S06_WORD_HI", "6a9a53000000000000000000"))
    load_workers = int(os.environ.get("S06_LOAD_WORKERS", "8"))
    lo = int.from_bytes(block_lo.binary, "big")
    hi = int.from_bytes(block_hi.binary, "big")
    span = (hi - lo) // load_workers

    ids: list = []
    words: list[dict] = []   # lightweight: _id stripped, vector not stored here
    mmap_path = Path(S06_MMAP_PATH)
    mmap_path.parent.mkdir(parents=True, exist_ok=True)
    V = np.memmap(
        mmap_path, mode="w+", dtype=np.float32,
        shape=(_MAX_WORDS, settings.embed_dim),
    )
    fat_proj = {"_id": 1, "node_type": 1, "vector": 1,
                "properties.word": 1, "properties.pos": 1, "properties.lang": 1,
                "properties.relations": 1}
    lock = threading.Lock()
    dead = 0
    j = 0

    def _load_range(k: int) -> None:
        nonlocal dead, j
        a = lo + k * span
        b = lo + (k + 1) * span if k < load_workers - 1 else hi
        q = {"_id": {"$gte": ObjectId(a.to_bytes(12, "big")),
                     "$lt": ObjectId(b.to_bytes(12, "big"))}}
        for doc in coll.find(q, fat_proj).batch_size(5000):
            if args.limit and j >= args.limit:
                break
            if doc.get("node_type") != "word":
                with lock:
                    dead += 1
                continue
            vec = doc.get("vector")
            if vec is None:
                with lock:
                    dead += 1
                continue
            with lock:
                ids.append(doc["_id"])
                words.append({"properties": doc["properties"]})
                V[j] = unpack_vec(vec)
                j += 1
                if j % 100_000 == 0:
                    log.info("  loaded %d L14 words", j)

    if load_workers > 1:
        with ThreadPoolExecutor(max_workers=load_workers) as ex:
            list(ex.map(_load_range, range(load_workers)))
    else:
        _load_range(0)
    n_words = len(ids)
    V = V[:n_words]
    log.info("loaded %d L14 word nodes with vectors (dead=%d, V memmap %s)",
             n_words, dead, mmap_path)
    if not args.dry_run:
        cp.processed = j
        cp_mod.save(cp, settings)
    if n_words < 2:
        log.warning("fewer than 2 L14 words (%d) — nothing to pair", n_words)
        del V
        _cleanup_mmap(log)
        if not args.dry_run:
            finish(cp, settings, log)
        return

    by_key: dict[tuple[str, str, str], int] = {}
    by_word: dict[tuple[str, str], list[int]] = {}
    for i, w in enumerate(words):
        p = w["properties"]
        key = (p["word"], p["pos"], p.get("lang", "en"))
        by_key[key] = i
        by_word.setdefault((p["word"], key[2]), []).append(i)

    # Embed TYPE_SENTENCES once. Use the bare OllamaEmbedder, NOT
    # get_embedder(): the build-all env sets EMBED_CACHE_FILE to the 413 GB
    # s04 orphan-reentry cache, and CachedEmbedder loads it all into RAM on
    # init — dozens of minutes and tens of GB for five fixed strings. A direct
    # OllamaEmbedder (external ollama, dim 4096) embeds them in well under a
    # second.
    embedder = OllamaEmbedder(settings)
    et_keys = list(TYPE_SENTENCES)
    et_vecs = embedder.embed([TYPE_SENTENCES[k] for k in et_keys])
    type_vec: dict[str, np.ndarray] = dict(zip(et_keys, et_vecs, strict=True))

    # Pre-load already-paired word indices so --force re-runs don't double-parent them.
    already_parented: set = {
        doc["_id"]
        for doc in coll.find(
            {"doc_type": "node", "node_type": "word", "level": 14,
             "parent_edge_id": {"$ne": None}},
            {"_id": 1},
        )
    }
    paired: set[int] = {i for i, oid in enumerate(ids) if oid in already_parented}
    if paired:
        log.info("skipping %d already-paired words (pre-loaded from DB)", len(paired))
    n_edges = 0
    batch_n = args.batch_size or settings.batch_size

    def _flush(edge_docs: list, pair_idxs: list[tuple[int, int]]) -> None:
        """Insert a batch of edges and stamp their parents. Streamed so a tier's
        edges never accumulate: each 4096-d edge doc expands to a ~130 KB
        Python-list vector, so a 1M-edge tier held whole would be ~130 GB."""
        nonlocal n_edges
        if not edge_docs:
            return
        n_edges += len(edge_docs)
        if args.dry_run:
            return
        res = coll.insert_many(edge_docs)
        now = datetime.now(timezone.utc)
        ups = []
        for (a, b), eid in zip(pair_idxs, res.inserted_ids, strict=True):
            ups.append(UpdateOne({"_id": ids[a]},
                                 {"$set": {"parent_edge_id": eid, "updated_at": now}}))
            ups.append(UpdateOne({"_id": ids[b]},
                                 {"$set": {"parent_edge_id": eid, "updated_at": now}}))
            doi_bridge.propagate(bridge_coll, eid, [ids[a], ids[b]])
        coll.bulk_write(ups, ordered=False)

    for tier in FERMION_TIERS:
        q_seed = settings.q_seeds[tier.q_seed_key]
        tv = type_vec[tier.edge_type]
        tier_start = n_edges
        edge_docs = []
        pair_idxs: list[tuple[int, int]] = []
        for i, w in enumerate(words):
            if i in paired:
                continue
            for rel in w["properties"].get("relations", []):
                if rel["kind"] not in tier.kaikki_fields:
                    continue
                # Target may exist under any pos; prefer same pos, else first match.
                # kaikki relations stay within one language — resolve only
                # against words of the source word's language.
                target = rel["word"]
                src_lang = w["properties"].get("lang", "en")
                cand = by_key.get((target, w["properties"]["pos"], src_lang))
                if cand is None:
                    cands = by_word.get((target, src_lang), [])
                    cand = cands[0] if cands else None
                if cand is None or cand == i or cand in paired or i in paired:
                    continue
                bv = compute_bary_vec(V[i], V[cand], tv, q_seed)
                edge_docs.append(
                    baryedge(ids[i], ids[cand], 14, bv, q_seed,
                             accumulated_weight=q_seed,
                             edge_type=tier.edge_type,
                             type_vector=tv, source="ingested", confidence=1.0)
                )
                pair_idxs.append((i, cand))
                paired.add(i)
                paired.add(cand)
                if len(edge_docs) >= batch_n:
                    _flush(edge_docs, pair_idxs)
                    edge_docs = []
                    pair_idxs = []
                break  # one parent per word per tier — move to next word
        _flush(edge_docs, pair_idxs)
        log.info("tier %d (%s/%s): %d edges, %d words now paired",
                 tier.priority, tier.edge_type, tier.q_seed_key,
                 n_edges - tier_start, len(paired))

    cp.processed = n_edges
    cp.total = n_edges
    del V
    _cleanup_mmap(log)
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
