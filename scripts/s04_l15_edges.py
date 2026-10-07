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

import gc
import os
import pickle
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne

from lib import checkpoint as cp_mod
from lib import doi_bridge
from lib.bary_vec import build_l15_type_text, compute_bary_vec
from lib.config import scratch_dir
from lib.db import get_collection
from lib.docs import baryedge
from lib.embed import get_embedder
from lib.match import (
    MATCH_DIM,
    ann_index,
    greedy_unique_match,
    project_match,
    top_k_pairs,
)
from lib.vector import unpack_vec
from scripts._base import bootstrap, finish

STAGE = "04_l15_edges"

# V (n_senses x embed_dim) is ~208 GB at 4096-dim for the full build — far too
# large to hold resident alongside the ANN pairing working set. Back it by a
# disk memmap in /storage (RAID, ~6 TB free) so only touched rows page into
# RAM; the file is deleted when the stage finishes.
S04_MMAP_PATH = os.environ.get("S04_MMAP_PATH") or str(scratch_dir() / "s04_V.mmap")

# ids/words are ~5 GB of process-only state rebuilt by re-streaming all 12.7M
# senses (~3.5 h). Persisting them lets a crashed resume skip that scan entirely.
S04_SIDECAR_PATH = os.environ.get("S04_SIDECAR_PATH") or str(
    scratch_dir() / "s04_ids_words.pkl"
)

# Orphan re-entry picks each orphan's parent BE via projected-space ANN; refine
# among the top-K candidates so HNSW's approximate ranking can't pick a dud.
S04_ORPHAN_ANN_K = int(os.environ.get("S04_ORPHAN_ANN_K", 32))

# sense/word count filters are `{doc_type, node_type, level}` — no existing
# index covers all three, so full counts scan ~12.7M docs and exceed the 120s
# socket timeout. This covering index makes those counts index-only. It also
# serves s05–s07 counts for free. Safe to drop manually after the build; the
# name is exported so removal is a one-liner.
S04_COVER_INDEX = "doc_type_1_node_type_1_level_1"
_COVER_FILTER = {"doc_type": "node", "node_type": "sense", "level": 15}
# Fallback size if the cover index can't be created/used (avoids hard-blocking).
S04_MAX_SENSES_CAP = int(os.environ.get("S04_MAX_SENSES_CAP", 13_500_000))
# Persisted ANN sweep result: lets a post-sweep crash resume straight into
# embed+insert, skipping BE vector projection, the parented scan, the HNSW
# build, and the ~37 min sweep (~2 h of wall time). Guarded by the sense-row
# count and L15 BE count current at build time.
S04_ANN_BUNDLE = os.environ.get("S04_ANN_BUNDLE") or str(scratch_dir() / "s04_ann_bundle.npz")

# Greedy-match output persisted the moment matching finishes: lets a crashed
# run skip the HNSW build + knn sweep + dedup + sort + greedy (~43 min on the
# T470s, ~4 h on the full build) and resume straight into embed+insert.
# Guarded by the sense-row count and both matching thresholds — any of them
# changes the pair set, so a mismatch forces a fresh match.
S04_PAIRS_BUNDLE = os.environ.get("S04_PAIRS_BUNDLE") or str(
    scratch_dir() / "s04_pairs.npz"
)


def _ensure_cover_index(coll, log) -> bool:
    """Create the covering index; return True if it's serving counts."""
    try:
        coll.create_index(
            [("doc_type", 1), ("node_type", 1), ("level", 1)],
            name=S04_COVER_INDEX,
            background=True,
        )
        return True
    except Exception as e:  # noqa: BLE001 — count must never hard-block the build
        log.warning("could not create cover index (%s); falling back to cap", e)
        return False


def _cleanup_mmap(log) -> None:
    """Release + remove the disk-backed V file (~208 GB) and ids/words sidecar."""
    for path, tag in ((S04_MMAP_PATH, "V memmap"), (S04_SIDECAR_PATH, "ids/words sidecar")):
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            log.warning("could not remove %s file %s", tag, path)


def _count_filled_rows(dim: int) -> int:
    """Rows [0, n0) of the preallocated V file hold real vectors; find n0 by
    scanning the zero tail backwards. Filled float32 rows are never all-zero,
    so the first non-zero row from the end is the last valid one."""
    n = Path(S04_MMAP_PATH).stat().st_size // (4 * dim)
    Vp = np.memmap(S04_MMAP_PATH, mode="r", dtype=np.float32, shape=(n, dim))
    block = 2000
    i = n
    while i > 0:
        lo = max(0, i - block)
        nz = np.flatnonzero(Vp[lo:i].any(axis=1))
        if nz.size:
            return lo + int(nz[-1]) + 1
        i = lo
    return 0


def _max_sense_id(coll, log):
    """Upper _id bound for the sense stream (index-free, deterministic).

    The stream iterates senses sorted by _id, but the *full* collection also
    holds L15 BEs with LATER _ids (inserted by s04 itself). An unbounded _id
    scan must fetch+filter that entire non-matching BE tail to prove EOF — a
    single getMore can then exceed any client socket timeout. We avoid the
    tail by capping the scan at the newest L15 sense doc's own _id: every
    sense doc sits at or below it (s03 was the only pass that wrote senses),
    and BEs inserted later are excluded by construction. Corpus-agnostic —
    no hardcoded cutover date.
    """
    doc = coll.find_one(
        {"doc_type": "node", "node_type": "sense", "level": 15},
        sort=[("_id", -1)],
        projection={"_id": 1},
    )
    if doc is None:
        return ObjectId("000000000000000000000000")
    return doc["_id"]


def _word_neighborhood(coll, word: str, pos: str, lang: str) -> tuple[list[str], list[str]]:
    """Return (antonyms, synonyms) for the L14 word node, for L15 type_text."""
    doc = coll.find_one(
        {"doc_type": "node", "node_type": "word", "properties.word": word,
         "properties.pos": pos, "properties.lang": lang},
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


def _neighborhoods_for_keys(
    coll, keys: Sequence[tuple[str, str, str]]
) -> dict[tuple[str, str, str], tuple[list[str], list[str]]]:
    """Batch equivalent of _word_neighborhood for a chunk of distinct keys.

    One $in query on the word-index prefix, then a Python-side map from the
    returned (word,pos,lang) docs. Avoids ~batch_n serial find_one round-trips
    per reentry chunk (the per-key mongo latency was roughly doubling the
    keys-embed phase wall time).
    """
    words_uniq = sorted({k[0] for k in keys})
    out: dict[tuple[str, str, str], tuple[list[str], list[str]]] = {}
    if not words_uniq:
        return out
    cur = coll.find(
        {"doc_type": "node", "node_type": "word", "properties.word": {"$in": words_uniq}},
        {"properties": 1},
    ).batch_size(4096)
    for d in cur:
        pr = d.get("properties", {})
        ants: list[str] = []
        syns: list[str] = []
        for r in pr.get("relations", []):
            if r["kind"] == "antonyms":
                ants.append(r["word"])
            elif r["kind"] == "synonyms":
                syns.append(r["word"])
        out[(pr.get("word"), pr.get("pos"), pr.get("lang"))] = (ants, syns)
    return out


def _load_bes_projected(coll, log) -> tuple[list, list, np.ndarray]:
    """One Mongo pass over L15 BE docs → (be_ids, be_q, BEVp).

    Streams vectors in 16k chunks and projects each immediately, so the
    full-dim be_vecs list (~87 GB at the full build) is never materialised.
    Holding be_vecs resident across the whole orphan phase is what OOM-killed
    s04 on the 2026-09-08 resume (be_vecs 87 GB + BEVp 22 GB + hnswlib's own
    ~22 GB copy peaked near the host's RAM while other tenants used it).
    """
    be_ids: list = []
    be_q: list[float] = []
    chunks: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    BATCH = 16_384
    n = 0
    t0 = datetime.now(timezone.utc)
    for be in coll.find(
        {"doc_type": "baryedge", "level": 15, "source": "inferred"},
        {"vector": 1, "q": 1},
    ):
        be_ids.append(be["_id"])
        be_q.append(float(be.get("q") or 0.5))
        cur.append(unpack_vec(be["vector"]))
        n += 1
        if len(cur) >= BATCH:
            chunks.append(project_match(np.stack(cur)))
            cur = []
            if n % (BATCH * 4) == 0:
                log.info("  loaded+projected %d BE vectors (elapsed=%s)",
                         n, datetime.now(timezone.utc) - t0)
    if cur:
        chunks.append(project_match(np.stack(cur)))
    dim = MATCH_DIM
    BEVp = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, dim), dtype=np.float32)
    )
    log.info("loaded+projected %d BE vectors -> (%d, %d) in %s",
             n, BEVp.shape[0], BEVp.shape[1], datetime.now(timezone.utc) - t0)
    return be_ids, be_q, BEVp


def _fetch_be_ids_q(coll, log):
    """Light pass over L15 BE docs for (be_ids, be_q) only — no vectors.

    Used by ANN-bundle resume: the insert loop needs best_bi→be_id and q, but
    not BEVp (the sweep that needs it is skipped). ~5-15 min vs ~40 min for the
    full vector projection.
    """
    be_ids: list = []
    be_q: list[float] = []
    for be in coll.find(
        {"doc_type": "baryedge", "level": 15, "source": "inferred"},
        {"_id": 1, "q": 1},
    ):
        be_ids.append(be["_id"])
        be_q.append(float(be.get("q") or 0.5))
    log.info("  fetched %d BE ids/q (no vectors)", len(be_ids))
    return be_ids, be_q


def _load_ann_bundle(n_senses: int, be_count: int, log) -> dict | None:
    """Load a persisted ANN sweep iff it matches the current world.

    Guards: sense-row count identical and L15 BE count identical (the two
    things that would change the sweep's input/output). Returns None to force a
    fresh sweep. be_ids/be_q are NOT stored in the bundle — resume fetches them
    lightly via _fetch_be_ids_q.
    """
    path = Path(S04_ANN_BUNDLE)
    if not path.exists():
        return None
    try:
        with np.load(path) as b:
            if int(b["sense_rows"]) != n_senses or int(b["be_count"]) != be_count:
                log.info(
                    "ANN bundle %s stale (%s rows / %s BEs vs %s / %s) — resweeping",
                    path, b["sense_rows"], b["be_count"], n_senses, be_count)
                return None
            orphans = np.asarray(b["orphans"], dtype=np.int64)
            best_bi = np.asarray(b["best_bi"], dtype=np.int64)
        log.info("  ANN bundle %s loaded (%d orphans)", path, len(orphans))
        return {"orphans": orphans, "best_bi": best_bi}
    except Exception as e:  # noqa: BLE001 — bundle failure must degrade to a sweep
        log.warning("could not load ANN bundle %s (%s) — resweeping", path, e)
        return None


def _load_pairs_bundle(
    n_senses: int, settings, log
) -> list[tuple[int, int, float]] | None:
    """Persisted greedy-match output, or None if absent/stale.

    Guards: sense-row count and both matching thresholds must match the
    current world — all three change which pairs greedy selects.
    """
    path = Path(S04_PAIRS_BUNDLE)
    if not path.exists():
        return None
    try:
        with np.load(path) as b:
            if (
                int(b["sense_rows"]) != n_senses
                or float(b["q_min"]) != float(settings.q_min_l15)
                or float(b["polysemy_floor"]) != float(settings.polysemy_q_floor)
            ):
                log.info(
                    "pairs bundle %s stale (%s rows / q_min=%s / floor=%s vs "
                    "%s / %s / %s) — re-matching",
                    path, int(b["sense_rows"]), float(b["q_min"]),
                    float(b["polysemy_floor"]), n_senses,
                    settings.q_min_l15, settings.polysemy_q_floor,
                )
                return None
            pairs = [
                (int(a), int(c), float(q))
                for a, c, q in zip(b["i"], b["j"], b["q"], strict=True)
            ]
        log.info(
            "pairs bundle %s loaded (%d pairs; HNSW/knn/greedy skipped)",
            path, len(pairs),
        )
        return pairs
    except Exception as e:  # noqa: BLE001 — bundle failure degrades to a re-match
        log.warning("could not load pairs bundle %s (%s) — re-matching", path, e)
        return None


def _save_pairs_bundle(
    pairs: list[tuple[int, int, float]], n_senses: int, settings, log
) -> None:
    """Persist greedy-match output for crash resume (non-fatal on failure)."""
    path = Path(S04_PAIRS_BUNDLE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            i=np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs)),
            j=np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs)),
            q=np.fromiter((p[2] for p in pairs), dtype=np.float64, count=len(pairs)),
            sense_rows=np.int64(n_senses),
            q_min=np.float64(settings.q_min_l15),
            polysemy_floor=np.float64(settings.polysemy_q_floor),
        )
        log.info("pairs bundle saved to %s (%d pairs)", path, len(pairs))
    except Exception as e:  # noqa: BLE001 — non-fatal: next run just re-matches
        log.warning("could not save pairs bundle (%s)", e)


def _filter_pending(
    pairs: list[tuple[int, int, float]],
    coll,
    ids: list,
    paired: set[int],
    be_ids: list,
    be_q: list[float],
    *,
    adopt: bool,
    log,
) -> list[tuple[int, int, float]]:
    """Drop pairs whose BaryEdge already exists in Mongo (crash resume).

    A pair counts as done iff its (cm1_id, cm2_id) BE is present — this
    catches the insert/stamp crash window too, where the BE exists but the
    senses were never stamped. Matched pairs mark both senses ``paired`` (so
    4e doesn't orphan them) and, when ``adopt`` (fresh-run path, be_ids empty),
    register the existing BE in the parent pool. Returns the pairs still owed.
    """
    if not pairs:
        return []
    existing: dict[tuple, tuple] = {}
    for be in coll.find(
        {"doc_type": "baryedge", "level": 15, "source": "inferred"},
        {"cm1_id": 1, "cm2_id": 1, "q": 1},
    ):
        existing[(be["cm1_id"], be["cm2_id"])] = (
            be["_id"], float(be.get("q") or 0.5),
        )
    pending: list[tuple[int, int, float]] = []
    n_done = 0
    for i, j, q in pairs:
        hit = existing.get((ids[i], ids[j])) or existing.get((ids[j], ids[i]))
        if hit is None:
            pending.append((i, j, q))
            continue
        n_done += 1
        paired.add(i)
        paired.add(j)
        if adopt:
            be_ids.append(hit[0])
            be_q.append(hit[1])
    if n_done:
        log.info(
            "pairs resume: %d/%d pairs already embedded — %d left to do",
            n_done, len(pairs), len(pending),
        )
    return pending


_T = TypeVar("_T")
_R = TypeVar("_R")


def _parallel_embed_batches(
    chunks: Iterable[_T],
    embed_fn: Callable[[_T], _R],
    workers: int,
) -> Iterator[tuple[_T, _R]]:
    """Yield ``(chunk, result)`` in chunk order with ``workers`` embeds in flight.

    Only the embed_fn calls overlap; the consumer (build docs → insert →
    stamp parents → checkpoint) still runs strictly sequentially in chunk
    order, so crash-resume semantics are byte-identical to the sequential
    path. ``workers <= 1`` degrades to plain sequential iteration.
    """
    if workers <= 1:
        for chunk in chunks:
            yield chunk, embed_fn(chunk)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs: deque[tuple[_T, Future[_R]]] = deque()
        it = iter(chunks)
        while True:
            # Refill the in-flight window before blocking on the oldest one,
            # so a slow response never idles the other workers.
            while len(futs) < workers:
                chunk = next(it, None)
                if chunk is None:
                    break
                futs.append((chunk, pool.submit(embed_fn, chunk)))
            if not futs:
                return
            chunk, fut = futs.popleft()
            yield chunk, fut.result()


def _run_orphan_reentry(
    coll,
    bridge_coll,
    ids: list,
    words: list[tuple[str, str, str]],
    V: np.ndarray,
    be_ids: list,
    be_q: list[float],
    BEVp: np.ndarray | None,
    paired: set[int],
    batch_n: int,
    embedder,
    nb,
    log,
    workers: int = 1,
    ann_bundle: dict | None = None,
) -> int:
    """Pair unpaired senses with the nearest existing L15 BE (batched embed).

    Two accelerations over the naïve scan+embed-per-orphan:
      B) nearest-BE search runs in projected (MATCH_DIM) space via hnswlib ANN,
         refined among the top-K candidates — the exact 4096-d argmax over all
         5.33M BEs would be ~8 days of BLAS vs ~1 h of ANN.
      A) the orphan type_text uses only (word,pos,lang) + neighborhood — no
         sense gloss — so all senses of a word are byte-identical: embed once
         per distinct (word,pos,lang), not once per orphan (~4.5 d → ~1 d).

    BEVp is the projected BE matrix; pass None to stream+project it here (via
    _load_bes_projected), in which case be_ids/be_q are overridden by the read
    so best_bi indices line up with BEVp's rows. The full-dim be_vecs list
    (~87 GB) is never materialised: the insert loop re-fetches only the chosen
    parents' vectors per chunk, keeping the working set flat over the ~1.5 d
    run. Per-chunk senses already stamped with a parent_edge_id are skipped so
    a crash between insert_many and the stamp bulk_write stays resume-safe.

    ann_bundle: a previously-persisted sweep (orphan rows + best_bi), loaded
    when the world state is unchanged. Resumes straight into embed+insert —
    skips the parented-scan orphans derivation, BE projection, HNSW build, and
    the ANN sweep. be_ids/be_q are still required for the insert loop and are
    fetched light (no vectors) by the caller. Correctness holds on partial
    re-entry because the per-chunk stamp check dedups already-parented senses.
    """
    orphans = [i for i in range(len(ids)) if i not in paired]
    if not orphans:
        log.info("L15 orphan re-entry: %d orphans → 0 new BEs", len(orphans))
        return 0
    # The persisted-bundle path skips both of these (~35 min BE projection and
    # the ~8 min HNSW build): the sweep was already done and persisted.
    if ann_bundle is None:
        if BEVp is None:
            be_ids, be_q, BEVp = _load_bes_projected(coll, log)
        n_orphans = len(orphans)
        n_be = BEVp.shape[0]
        log.info("L15 orphan re-entry: %d orphans vs %d BEs", n_orphans, n_be)

        # --- B: projected ANN nearest-BE search (built once, chunked queries) ---
        idx = ann_index(BEVp, preprojected=True, log_build=True)
        k_ann = min(S04_ORPHAN_ANN_K, max(1, n_be - 1))
        idx.set_ef(max(2 * (k_ann + 1), 100))
    # Evict V's served rows from page cache per chunk (POSIX_FADV_DONTNEED):
    # sweeping all 6.7M orphan rows through the memmap otherwise pins ~110GB of
    # file-backed pages resident (counted in RSS), ramping the process past the
    # supervisor OOM guard mid-sweep (observed death at 191GiB, 09-09-09).
    vfd = os.open(S04_MMAP_PATH, os.O_RDONLY)
    row_bytes = V.shape[1] * V.dtype.itemsize

    def evict_rows(oi_idx: np.ndarray) -> None:
        if oi_idx.size == 0:
            return
        ks = np.flatnonzero(np.diff(oi_idx) != 1)
        starts = np.concatenate([[0], ks + 1])
        ends = np.concatenate([ks + 1, [oi_idx.size]])
        for s, e in zip(starts, ends, strict=True):
            r0, r1 = int(oi_idx[s]), int(oi_idx[e - 1])
            os.posix_fadvise(vfd, r0 * row_bytes, (r1 - r0 + 1) * row_bytes,
                             os.POSIX_FADV_DONTNEED)

    if ann_bundle is not None:
        # --- resume from a persisted sweep: no scan, no projection, no ANN ---
        orphans = [int(i) for i in ann_bundle["orphans"]]
        best_bi = np.asarray(ann_bundle["best_bi"], dtype=np.int64)
        n_orphans = len(orphans)
        n_be = len(be_ids)
        log.info(
            "L15 orphan re-entry: ANN bundle resume — %d orphans vs %d BEs "
            "(sweep skipped)", n_orphans, n_be)
    else:
        orphans = [i for i in range(len(ids)) if i not in paired]
        if not orphans:
            os.close(vfd)
            log.info("L15 orphan re-entry: %d orphans → 0 new BEs", len(orphans))
            return 0
        if BEVp is None:
            be_ids, be_q, BEVp = _load_bes_projected(coll, log)
        n_orphans = len(orphans)
        n_be = BEVp.shape[0]
        log.info("L15 orphan re-entry: %d orphans vs %d BEs", n_orphans, n_be)

        # --- B: projected ANN nearest-BE search (built once, chunked queries) ---
        idx = ann_index(BEVp, preprojected=True, log_build=True)
        k_ann = min(S04_ORPHAN_ANN_K, max(1, n_be - 1))
        idx.set_ef(max(2 * (k_ann + 1), 100))
        best_bi = np.empty(n_orphans, dtype=np.int64)
        CHUNK_Q = 16_384
        t0 = datetime.now(timezone.utc)
        for start in range(0, n_orphans, CHUNK_Q):
            end = min(start + CHUNK_Q, n_orphans)
            oi_c = orphans[start:end]
            # Fancy-index V (memmap) in slices so only ~1 GB is materialised.
            ocp = project_match(np.asarray(V[oi_c], dtype=np.float32))
            lab, _ = idx.knn_query(ocp, k=k_ann)
            # Refine among ANN candidates: exact argmax in projected space. Reverted
            # from a single BEVp[lab] gather: that materialises an (n, k, dim)
            # tensor (~8.6 GB per 65k-chunk at k=32) whose malloc arenas never
            # returned to the OS, ramping RSS ~150 GB+ over the whole sweep. The
            # column-wise gather keeps only one (n, dim) slice live at a time.
            best = np.zeros(lab.shape[0], dtype=np.int64)
            best_val = np.einsum("nd,nd->n", BEVp[lab[:, 0]], ocp)
            for c in range(1, k_ann):
                val = np.einsum("nd,nd->n", BEVp[lab[:, c]], ocp)
                upd = val > best_val
                if upd.any():
                    best[upd] = c
                    best_val[upd] = val[upd]
            best_bi[start:end] = lab[np.arange(lab.shape[0]), best]
            evict_rows(np.asarray(oi_c))
            del ocp, lab, best, best_val
            if (start // CHUNK_Q) % 4 == 0:
                log.info("  ANN parent search %d/%d orphans (elapsed=%s)",
                         end, n_orphans, datetime.now(timezone.utc) - t0)
        del BEVp, idx
        gc.collect()
        log.info("  ANN parent search done in %s", datetime.now(timezone.utc) - t0)
        try:
            np.savez(
                S04_ANN_BUNDLE,
                orphans=np.asarray(orphans, dtype=np.int64),
                best_bi=best_bi.astype(np.int64),
                sense_rows=np.int64(len(ids)),
                be_count=np.int64(n_be),
            )
            log.info("  ANN sweep saved to %s", S04_ANN_BUNDLE)
        except Exception as e:  # noqa: BLE001 — non-fatal: resweep on next run
            log.warning("  could not save ANN bundle (%s)", e)

    re_meta: list[tuple[int, int]] = [
        (oi, int(bi)) for oi, bi in zip(orphans, best_bi.tolist(), strict=True)
    ]

    # --- A: embed once per distinct (word,pos,lang) ---
    # The vectors are kept in key_vecs and read back per-orphan below. That
    # read-back used to go through embedder.vector_for(), which re-embeds on
    # a cold cache — with no EMBED_CACHE_FILE configured that meant one HTTP
    # request *per orphan* (~488k requests). The dict (~keys × dim × 4 B,
    # ~2 GB at poc scale) replaces both the disk cache and those re-embeds.
    keys = list({words[oi] for oi, _ in re_meta})
    key_vecs: dict[tuple[str, str, str], np.ndarray] = {}
    n_keys_embedded = 0
    next_log_at = 16384
    t0 = datetime.now(timezone.utc)

    def _embed_key_batch(ks: list) -> tuple[list[str], np.ndarray]:
        nbs = _neighborhoods_for_keys(coll, ks)
        texts = [
            build_l15_type_text(
                k[0], *nbs.get(k, ([], [])), k[0], [], []
            )
            for k in ks
        ]
        return texts, embedder.embed(texts)

    key_batches = (
        keys[start : start + batch_n] for start in range(0, len(keys), batch_n)
    )
    for ks, (texts, vecs) in _parallel_embed_batches(
        key_batches, _embed_key_batch, workers
    ):
        for k, v in zip(ks, vecs, strict=True):
            key_vecs[k] = v
        n_keys_embedded += len(texts)
        if n_keys_embedded >= next_log_at:
            log.info("  embedded %d/%d distinct keys (elapsed=%s)",
                     n_keys_embedded, len(keys),
                     datetime.now(timezone.utc) - t0)
            next_log_at += 16384
    log.info("  embedded %d distinct word keys (vs %d orphans) in %s",
             n_keys_embedded, n_orphans, datetime.now(timezone.utc) - t0)

    # --- per-orphan BE build + stamp (same semantics as before) ---
    n_reentry = 0
    t0 = datetime.now(timezone.utc)
    for start in range(0, len(re_meta), batch_n):
        chunk = re_meta[start : start + batch_n]
        # Idempotent resume: skip senses already stamped with a parent edge.
        oi_ids = [ids[oi] for oi, _ in chunk]
        taken = {
            d["_id"]
            for d in coll.find(
                {"_id": {"$in": oi_ids}, "parent_edge_id": {"$ne": None}},
                {"_id": 1},
            )
        }
        if taken:
            chunk = [(oi, bi) for oi, bi in chunk if ids[oi] not in taken]
        if not chunk:
            continue
        # Fetch this chunk's parents' 4096-d vectors once (never the full set).
        parent_ids = [be_ids[bi] for bi in {bi for _, bi in chunk}]
        parent_vecs: dict = {}
        for be in coll.find({"_id": {"$in": parent_ids}}, {"vector": 1}):
            parent_vecs[be["_id"]] = unpack_vec(be["vector"])
        re_docs = []
        for (oi, bi), tv in zip(
            chunk,
            [key_vecs[words[oi]] for oi, _ in chunk],
            strict=True,
        ):
            q = be_q[bi]
            bv = compute_bary_vec(V[oi], parent_vecs[be_ids[bi]], tv, q)
            re_docs.append(
                baryedge(ids[oi], be_ids[bi], 15, bv, q, accumulated_weight=q,
                         edge_type=None, type_vector=tv,
                         source="reentry", confidence=float(q))
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
        evict_rows(np.asarray([oi for oi, _ in chunk]))
        n_reentry += len(re_docs)
        if n_reentry % 100_000 < batch_n:
            log.info("  re-entered %d/%d orphans (elapsed=%s)",
                     n_reentry, n_orphans, datetime.now(timezone.utc) - t0)

    log.info("L15 orphan re-entry: %d orphans → %d new BEs", len(orphans), n_reentry)
    os.close(vfd)
    return n_reentry


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    if args.reset:
        # --reset means "start over": drop persisted match/sweep state too, or
        # a re-match with unchanged thresholds would silently reuse it.
        for stale in (S04_PAIRS_BUNDLE, S04_ANN_BUNDLE):
            try:
                Path(stale).unlink(missing_ok=True)
            except OSError:
                pass
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
    # Pre-allocate with an exact upper bound so we never hold two copies of the
    # vector data in memory simultaneously. The covering index makes the count
    # index-only; if it's unavailable we fall back to a fixed cap (never block).
    if args.limit:
        _MAX_SENSES = args.limit
    elif _ensure_cover_index(coll, log):
        _MAX_SENSES = coll.count_documents(_COVER_FILTER)
    else:
        _MAX_SENSES = S04_MAX_SENSES_CAP
        log.warning("no cover index; pre-allocating cap %d", _MAX_SENSES)

    ids: list = []
    words: list[tuple[str, str, str]] = []
    mmap_path = Path(S04_MMAP_PATH)
    mmap_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = Path(S04_SIDECAR_PATH)

    # Crash-resume: a --force rerun over existing data can reuse the V file
    # already sitting on disk (identical row order — same query, same sort, no
    # sense-doc changes between runs). Only the rows past the last valid one
    # need a vector; opening mode="r+" avoids re-writing ~208 GB and the disk
    # contention that originally starved the MongoDB cursors.
    reuse_from = 0
    reuse_mode = "w+"
    if args.force and existing_be_count and mmap_path.exists():
        expected_bytes = _MAX_SENSES * settings.embed_dim * 4
        try:
            if mmap_path.stat().st_size == expected_bytes:
                reuse_mode = "r+"
                reuse_from = _count_filled_rows(settings.embed_dim)
                log.info(
                    "--force resume: reusing %s; %d/%d vector rows already valid",
                    mmap_path, reuse_from, _MAX_SENSES,
                )
        except (OSError, ValueError):
            log.warning("could not reuse %s; falling back to full rebuild", mmap_path)

    # Sidecar: restore ids/words without the ~3.5 h Mongo re-stream. Only valid
    # when the mmap is reusable too (the sidecar implies vectors were written).
    reuse_sidecar = False
    if args.force and existing_be_count and reuse_mode == "r+" and sidecar_path.exists():
        try:
            with open(sidecar_path, "rb") as fh:
                sc = pickle.load(fh)
            if len(sc["ids"]) == _MAX_SENSES and len(sc["words"]) == _MAX_SENSES:
                ids, words = sc["ids"], sc["words"]
                reuse_sidecar = True
                log.info(
                    "--force resume: ids/words restored from %s (%d rows)",
                    sidecar_path, len(ids),
                )
            else:
                log.warning("sidecar row-count mismatch; re-streaming ids/words")
        except Exception as e:  # noqa: BLE001 — resume must degrade to full stream
            log.warning("sidecar unreadable (%s); re-streaming ids/words", e)

    V = np.memmap(
        mmap_path, mode=reuse_mode, dtype=np.float32,
        shape=(_MAX_SENSES, settings.embed_dim),
    )

    if not reuse_sidecar:
        # Skip the heavy vector payload when the rows are being filled below by
        # backfill; otherwise stream vectors as before.
        stream_vectors = reuse_from == 0
        proj = {"_id": 1, "properties.word": 1, "properties.pos": 1,
                "properties.lang": 1}
        if stream_vectors:
            proj["vector"] = 1
        log.info("streaming L15 senses from MongoDB (pre-alloc %d)", _MAX_SENSES)
        # Bound the _id scan at the s03 cutover (see _max_sense_id): avoids
        # fetching the 5.3M+ BE tail on the final getMore.
        stream_filter = dict(_COVER_FILTER)
        stream_filter["_id"] = {"$lte": _max_sense_id(coll, log)}
        cur = coll.find(stream_filter, proj).sort("_id", 1)

        for i, doc in enumerate(cur):
            if i >= _MAX_SENSES:
                # Only reachable via the fallback cap — never silently truncate.
                if not args.limit and _MAX_SENSES == S04_MAX_SENSES_CAP:
                    raise RuntimeError(
                        f"hit pre-alloc cap ({S04_MAX_SENSES_CAP}) — senses exceed it; "
                        "raise S04_MAX_SENSES_CAP or fix the covering index"
                    )
                break
            ids.append(doc["_id"])
            p = doc["properties"]
            words.append((p["word"], p["pos"], p.get("lang", "en")))
            if stream_vectors:
                V[i] = unpack_vec(doc["vector"])
            if i % 100_000 == 0 and i > 0:
                log.info("  loaded %d senses", i)
        n = len(ids)
        if not args.limit and not args.dry_run:
            try:
                with open(sidecar_path, "wb") as fh:
                    pickle.dump({"ids": ids, "words": words}, fh, protocol=pickle.HIGHEST_PROTOCOL)
                log.info("wrote ids/words sidecar %s (%d rows)", sidecar_path, n)
            except Exception as e:  # noqa: BLE001 — non-fatal; next run re-streams
                log.warning("could not write ids/words sidecar (%s)", e)
    else:
        n = len(ids)

    V = V[:n]

    if reuse_from > 0 and n > reuse_from:
        tail_ids = ids[reuse_from:]
        log.info("  backfilling %d tail vectors by _id", len(tail_ids))
        by_id = {
            d["_id"]: unpack_vec(d["vector"])
            for d in coll.find({"_id": {"$in": tail_ids}}, {"vector": 1})
        }
        for k in range(reuse_from, n):
            V[k] = by_id[ids[k]]

    if n < 2:
        log.warning("fewer than 2 L15 senses (%d) — nothing to pair", n)
        cp.total = n
        try:
            del V
        except Exception:
            pass
        _cleanup_mmap(log)
        if not args.dry_run:
            finish(cp, settings, log)
        return

    embedder = get_embedder(settings)
    batch_n = args.batch_size or settings.embed_batch_size

    nb_cache: dict[tuple[str, str, str], tuple[list[str], list[str]]] = {}

    def nb(wp: tuple[str, str, str]) -> tuple[list[str], list[str]]:
        if wp not in nb_cache:
            nb_cache[wp] = _word_neighborhood(coll, wp[0], wp[1], wp[2])
        return nb_cache[wp]

    be_ids: list = []
    be_q: list[float] = []
    paired: set[int] = set()
    n_pairs = 0
    orphan_BEVp: np.ndarray | None = None
    ann_bundle: dict | None = None
    # Greedy-match output: filled by the resume/match branches below, consumed
    # by the shared 4c/4d embed+insert loop. pending = pairs still owed here.
    pairs: list[tuple[int, int, float]] = []
    pending: list[tuple[int, int, float]] = []

    def _match_pairs(V: np.ndarray) -> list[tuple[int, int, float]]:
        """4a/4b greedy match, persisting the bundle before any embed call."""
        by_word: dict[tuple[str, str, str], list[int]] = {}
        for i, wp in enumerate(words):
            by_word.setdefault(wp, []).append(i)
        same_word: set[frozenset[int]] = set()
        for idxs in by_word.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    same_word.add(frozenset((idxs[a], idxs[b])))
        m = greedy_unique_match(
            top_k_pairs(V),
            threshold=settings.q_min_l15,
            same_word=same_word,
            polysemy_floor=settings.polysemy_q_floor,
        )
        log.info("greedy match: %d pairs from %d senses", len(m), n)
        if not args.dry_run:
            # Persist before the first embed call: a crash in 4c/4d (e.g.
            # a dead embed endpoint) must not cost another HNSW pass.
            _save_pairs_bundle(m, n, settings, log)
        return m

    # --- --force with existing BEs: skip main pairing, resume orphan re-entry ---
    if args.force and existing_be_count:
        log.info(
            "--force: %d L15 BEs already present — loading state, skipping to orphan re-entry",
            coll.count_documents({"doc_type": "baryedge", "level": 15}),
        )
        # Guard against re-entry products (source="reentry") polluting the parent
        # pool: best_bi indices are positions into the pool as it was at sweep
        # time, so the count must only see non-reentry edges.
        ann_bundle = None
        be_count_now = coll.count_documents(
            {"doc_type": "baryedge", "level": 15, "source": "inferred"}
        )
        ann_bundle = _load_ann_bundle(len(ids), be_count_now, log)
        if ann_bundle is not None:
            be_ids, be_q = _fetch_be_ids_q(coll, log)
            orphan_BEVp = None
            paired = set()
            n_pairs = len(be_ids)
            log.info(
                "loaded %d existing BEs, %d swept orphans reusable "
                "(skipping scan/sweep)", n_pairs, len(ann_bundle["best_bi"]))
        else:
            be_ids, be_q, orphan_BEVp = _load_bes_projected(coll, log)
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
            ann_bundle = None
            log.info(
                "loaded %d existing BEs, %d paired senses, %d orphans to process",
                n_pairs, len(paired), n - len(paired),
            )
        # A persisted greedy match means the main phase never finished — the
        # "skip to orphan re-entry" shortcut would silently drop every pair
        # that was never embedded. Resume those first.
        pairs = _load_pairs_bundle(len(ids), settings, log)
        if pairs is not None:
            pending = _filter_pending(
                pairs, coll, ids, paired, be_ids, be_q, adopt=False, log=log
            )
            log.info("--force: resuming main pairing (%d pairs left)", len(pending))
        else:
            # No usable bundle (stale guard / deleted): silently skipping the
            # main phase would strand every never-embedded pair as an orphan
            # in 4e. Re-derive the match; _filter_pending drops the done ones.
            log.warning("--force: no usable pairs bundle — re-running greedy match")
            pairs = _match_pairs(V)
            pending = _filter_pending(
                pairs, coll, ids, paired, be_ids, be_q, adopt=False, log=log
            )
            log.info("--force: main pairing re-derived (%d pairs left)", len(pending))
    else:
        pairs = _load_pairs_bundle(n, settings, log)
        if pairs is None:
            pairs = _match_pairs(V)
        n_pairs = len(pairs)
        pending = _filter_pending(
            pairs, coll, ids, paired, be_ids, be_q, adopt=True, log=log
        )

    # --- 4c/4d: build type_text per pair, batch-embed, compute bary_vec ---
    # Shared by the fresh-match and both resume paths. Progress is durable
    # per chunk: parents are stamped immediately after the insert, and the
    # checkpoint is saved, so a crash loses at most one chunk of work.
    # Embed calls may run concurrently (--embed-concurrency); everything from
    # bary_vec computation downwards stays sequential in chunk order.
    embed_workers = max(1, args.embed_concurrency)
    inserted_any = False
    consumed = 0  # pending pairs already through the loop (for cp.total)

    def _embed_pair_chunk(chunk: list) -> np.ndarray:
        texts = []
        for i, j, _q in chunk:
            ant_a, syn_a = nb(words[i])
            ant_b, syn_b = nb(words[j])
            texts.append(
                build_l15_type_text(words[i][0], ant_a, syn_a, words[j][0], ant_b, syn_b)
            )
        return embedder.embed(texts)

    pair_batches = (
        pending[start : start + batch_n]
        for start in range(0, len(pending), batch_n)
    )
    for chunk, type_vecs in _parallel_embed_batches(
        pair_batches, _embed_pair_chunk, embed_workers
    ):
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
        inserted_any = True
        parent_updates: list[UpdateOne] = []
        for (i, j, q), eid, _doc in zip(chunk, res.inserted_ids, edge_docs, strict=True):
            be_ids.append(eid)
            be_q.append(q)
            now = datetime.now(timezone.utc)
            parent_updates.append(
                UpdateOne({"_id": ids[i]}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
            )
            parent_updates.append(
                UpdateOne({"_id": ids[j]}, {"$set": {"parent_edge_id": eid, "updated_at": now}})
            )
            doi_bridge.propagate(bridge_coll, eid, [ids[i], ids[j]])
        coll.bulk_write(parent_updates, ordered=False)
        cp.processed = len(be_ids)
        cp.total = len(be_ids) + len(pending) - (consumed + len(chunk))
        cp_mod.save(cp, settings)
        consumed += len(chunk)

    if inserted_any:
        # BEVp was built from the pre-existing BEs only; any BE added above is
        # missing from its rows. Force a fresh projection in 4e rather than
        # handing _run_orphan_reentry a matrix shorter than its index space.
        if orphan_BEVp is not None:
            log.info("new BEs inserted — dropping stale BE projection for 4e")
            orphan_BEVp = None
        n_pairs = len(be_ids)

    # --- 4e: L15 orphan re-entry ---
    n_reentry = 0
    if not args.dry_run:
        n_reentry = _run_orphan_reentry(
            coll, bridge_coll, ids, words, V, be_ids, be_q, orphan_BEVp, paired, batch_n,
            embedder, nb, log, workers=embed_workers, ann_bundle=ann_bundle
        )

    cp.processed = n_pairs + n_reentry
    cp.total = n_pairs + n_reentry
    # Release the disk-backed V (flush + unmap) and remove the temp file.
    try:
        del V
    except Exception:
        pass
    _cleanup_mmap(log)
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
