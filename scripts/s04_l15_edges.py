"""Cosine-driven greedy L15 BaryEdge formation + L15 orphan re-entry.

L15 orphan re-entry MUST complete here (before s05_word_vectors) — see
v0.4 §2.4: word vectors depend on the finalized set of L15 BEs.

Safeguard: refuses to run if any L15 BaryEdge already exists (would
violate the unique-parent invariant on re-run). Use ``--reset`` after
dropping edges, or ``--force`` to override.

``--force`` with existing L15 BEs present auto-detects the partially-done
state (main pairs written, orphan re-entry not yet run) and skips straight
to orphan re-entry.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from pymongo import UpdateOne

from lib import doi_bridge
from lib.bary_vec import build_l15_type_text, compute_bary_vec
from lib.db import get_collection
from lib.docs import baryedge
from lib.embed import get_embedder
from lib.match import greedy_unique_match, top_k_pairs
from scripts._base import bootstrap, finish

STAGE = "04_l15_edges"


def _word_neighborhood(coll, word: str, pos: str) -> tuple[list[str], list[str]]:
    """Return (antonyms, synonyms) for the L14 word node, for L15 type_text."""
    doc = coll.find_one(
        {"doc_type": "node", "node_type": "word", "properties.word": word, "properties.pos": pos},
        {"properties.relations": 1},
    )
    ants: list[str] = []
    syns: list[str] = []
    for r in (doc or {}).get("properties", {}).get("relations", []):
        if r["kind"] == "antonyms":
            ants.append(r["word"])
        elif r["kind"] == "synonyms":
            syns.append(r["word"])
    return ants, syns


def _run_orphan_reentry(
    coll,
    bridge_coll,
    ids: list,
    words: list[tuple[str, str]],
    V: np.ndarray,
    be_ids: list,
    be_vecs: list[np.ndarray],
    be_q: list[float],
    paired: set[int],
    batch_n: int,
    embedder,
    nb,
    log,
) -> int:
    """Pair unpaired senses with the nearest existing L15 BE (batched embed)."""
    orphans = [i for i in range(len(ids)) if i not in paired]
    if not orphans or not be_vecs:
        log.info("L15 orphan re-entry: %d orphans → 0 new BEs", len(orphans))
        return 0

    BEV = np.stack(be_vecs)
    OV = V[orphans]  # (n_orphans, dim)
    # Chunked matmul: full OV@BEV.T would be ~(477K×630K×4B)=1.1TB; CHUNK keeps it to ~614MB/step
    CHUNK = 1024
    n_orphans = len(orphans)
    best_bi = np.empty(n_orphans, dtype=np.int64)
    for start in range(0, n_orphans, CHUNK):
        end = min(start + CHUNK, n_orphans)
        best_bi[start:end] = np.argmax(OV[start:end] @ BEV.T, axis=1)
    re_meta: list[tuple[int, int]] = [(oi, int(bi)) for oi, bi in zip(orphans, best_bi.tolist())]

    n_reentry = 0
    for start in range(0, len(re_meta), batch_n):
        chunk = re_meta[start : start + batch_n]
        re_texts = []
        for oi, _ in chunk:
            ant_a, syn_a = nb(words[oi])
            re_texts.append(
                build_l15_type_text(words[oi][0], ant_a, syn_a, words[oi][0], [], [])
            )
        type_vecs = embedder.embed(re_texts)
        re_docs = []
        for (oi, bi), tv in zip(chunk, type_vecs, strict=True):
            q = be_q[bi]
            bv = compute_bary_vec(V[oi], BEV[bi], tv, q)
            re_docs.append(
                baryedge(ids[oi], be_ids[bi], 15, bv, q, accumulated_weight=q,
                         edge_type=None, type_vector=tv,
                         source="inferred", confidence=float(q))
            )
        res = coll.insert_many(re_docs)
        ups = []
        now = datetime.now(timezone.utc)
        for (oi, bi), eid in zip(chunk, res.inserted_ids, strict=True):
            ups.append(
                UpdateOne({"_id": ids[oi]}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
            )
            doi_bridge.propagate(bridge_coll, eid, [ids[oi], be_ids[bi]])
        coll.bulk_write(ups, ordered=False)
        n_reentry += len(re_docs)

    log.info("L15 orphan re-entry: %d orphans → %d new BEs", len(orphans), n_reentry)
    return n_reentry


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    bridge_coll = doi_bridge.get_bridge_collection(settings)
    log.info("start processed=%d dry_run=%s q_min=%.2f", cp.processed, args.dry_run,
             settings.q_min_l15)

    existing_be_count = coll.count_documents({"doc_type": "baryedge", "level": 15}, limit=1)

    if not args.dry_run and not args.force and existing_be_count:
        raise RuntimeError(
            "L15 baryedges already present — re-running would double-parent senses. "
            "Drop them and use --reset, or pass --force."
        )

    # --- Load all L15 sense nodes ---
    # Pre-allocate numpy array with a safe upper bound so we never hold two
    # copies of the vector data in memory simultaneously (~37 GB Python-list
    # overhead with list(cur), ~10 GB double-copy with np.stack(vecs)).
    _MAX_SENSES = args.limit or 2_000_000
    log.info("streaming L15 senses from MongoDB (pre-alloc %d)", _MAX_SENSES)
    cur = coll.find(
        {"doc_type": "node", "node_type": "sense", "level": 15},
        {"_id": 1, "vector": 1, "properties.word": 1, "properties.pos": 1},
    ).sort("_id", 1)

    ids: list = []
    words: list[tuple[str, str]] = []
    V = np.empty((_MAX_SENSES, 768), dtype=np.float32)
    for i, doc in enumerate(cur):
        if i >= _MAX_SENSES:
            break
        ids.append(doc["_id"])
        words.append((doc["properties"]["word"], doc["properties"]["pos"]))
        V[i] = doc["vector"]
        if i % 100_000 == 0 and i > 0:
            log.info("  loaded %d senses", i)
    n = len(ids)
    V = V[:n]

    if n < 2:
        log.warning("fewer than 2 L15 senses (%d) — nothing to pair", n)
        cp.total = n
        if not args.dry_run:
            finish(cp, settings, log)
        return

    embedder = get_embedder(settings)
    batch_n = args.batch_size or settings.embed_batch_size

    nb_cache: dict[tuple[str, str], tuple[list[str], list[str]]] = {}

    def nb(wp: tuple[str, str]) -> tuple[list[str], list[str]]:
        if wp not in nb_cache:
            nb_cache[wp] = _word_neighborhood(coll, wp[0], wp[1])
        return nb_cache[wp]

    be_ids: list = []
    be_vecs: list[np.ndarray] = []
    be_q: list[float] = []
    paired: set[int] = set()
    n_pairs = 0

    # --- --force with existing BEs: skip main pairing, resume orphan re-entry ---
    if args.force and existing_be_count:
        log.info(
            "--force: %d L15 BEs already present — loading state, skipping to orphan re-entry",
            coll.count_documents({"doc_type": "baryedge", "level": 15}),
        )
        for be in coll.find({"doc_type": "baryedge", "level": 15}, {"vector": 1, "q": 1}):
            be_ids.append(be["_id"])
            be_vecs.append(np.asarray(be["vector"], dtype=np.float32))
            be_q.append(float(be.get("q") or 0.5))
        n_pairs = len(be_ids)
        parented_ids = {
            s["_id"]
            for s in coll.find(
                {"doc_type": "node", "node_type": "sense",
                 "parent_edge_id": {"$ne": None}},
                {"_id": 1},
            )
        }
        paired = {i for i, sid in enumerate(ids) if sid in parented_ids}
        log.info(
            "loaded %d existing BEs, %d paired senses, %d orphans to process",
            n_pairs, len(paired), n - len(paired),
        )
    else:
        # --- Same-headword pairs get the polysemy q floor ---
        by_word: dict[tuple[str, str], list[int]] = {}
        for i, wp in enumerate(words):
            by_word.setdefault(wp, []).append(i)
        same_word: set[frozenset[int]] = set()
        for idxs in by_word.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    same_word.add(frozenset((idxs[a], idxs[b])))

        # --- 4a/4b: greedy highest-cosine matching ---
        pairs = greedy_unique_match(
            top_k_pairs(V),
            threshold=settings.q_min_l15,
            same_word=same_word,
            polysemy_floor=settings.polysemy_q_floor,
        )
        log.info("greedy match: %d pairs from %d senses", len(pairs), n)
        n_pairs = len(pairs)

        parent_updates: list[UpdateOne] = []

        # --- 4c/4d: build type_text per pair, batch-embed, compute bary_vec ---
        for start in range(0, len(pairs), batch_n):
            chunk = pairs[start : start + batch_n]
            texts = []
            for i, j, _q in chunk:
                ant_a, syn_a = nb(words[i])
                ant_b, syn_b = nb(words[j])
                texts.append(
                    build_l15_type_text(words[i][0], ant_a, syn_a, words[j][0], ant_b, syn_b)
                )
            type_vecs = embedder.embed(texts)
            edge_docs = []
            for (i, j, q), tv in zip(chunk, type_vecs, strict=True):
                bv = compute_bary_vec(V[i], V[j], tv, q)
                edge_docs.append(
                    baryedge(ids[i], ids[j], 15, bv, q, accumulated_weight=q,
                             edge_type=None, type_vector=tv,
                             source="inferred", confidence=float(q))
                )
                paired.add(i)
                paired.add(j)
            if args.dry_run:
                continue
            res = coll.insert_many(edge_docs)
            for (i, j, q), eid, doc in zip(chunk, res.inserted_ids, edge_docs, strict=True):
                be_ids.append(eid)
                be_vecs.append(np.asarray(doc["vector"], dtype=np.float32))
                be_q.append(q)
                now = datetime.now(timezone.utc)
                parent_updates.append(
                    UpdateOne({"_id": ids[i]}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
                )
                parent_updates.append(
                    UpdateOne({"_id": ids[j]}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
                )
                doi_bridge.propagate(bridge_coll, eid, [ids[i], ids[j]])
            cp.processed = start + len(chunk)
        if parent_updates and not args.dry_run:
            coll.bulk_write(parent_updates, ordered=False)

    # --- 4e: L15 orphan re-entry ---
    n_reentry = 0
    if not args.dry_run:
        n_reentry = _run_orphan_reentry(
            coll, bridge_coll, ids, words, V, be_ids, be_vecs, be_q, paired, batch_n,
            embedder, nb, log
        )

    cp.processed = n_pairs + n_reentry
    cp.total = n_pairs + n_reentry
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
