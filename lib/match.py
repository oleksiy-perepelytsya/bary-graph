"""Pair-matching primitives shared by stages 04, 06, 07, 08.

``top_k_pairs`` uses brute-force O(n²) for small inputs and switches to an
hnswlib HNSW ANN index above ANN_THRESHOLD. The rest of the pipeline only
consumes the ranked-pair iterable and is backend-agnostic.

ANN parameters are tunable via environment variables:
  ANN_THRESHOLD        switch point (default 20 000)
  ANN_K                neighbours queried per vector (default 20)
  ANN_EF               hnswlib ef query parameter (default 100)
  ANN_M                hnswlib M graph parameter (default 16)
  ANN_EF_CONSTRUCTION  hnswlib ef_construction (default 100)
  ANN_BUILD_CHUNK      nodes per add_items batch for progress logging (default 50 000)
"""

from __future__ import annotations

import gc
import logging
import operator
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FermionTier:
    """One row of the L14 fermion-order matching table (v0.4 §7.1)."""

    priority: int
    edge_type: str
    kaikki_fields: tuple[str, ...]
    q_seed_key: str


FERMION_TIERS: tuple[FermionTier, ...] = (
    FermionTier(1, "contradicts", ("antonyms",), "contradicts"),
    FermionTier(2, "applies_to", ("meronyms", "holonyms"), "applies_to"),
    FermionTier(3, "is_instance_of", ("hypernyms", "hyponyms"), "is_instance_of"),
    FermionTier(4, "extends", ("derived", "related"), "extends"),
    FermionTier(5, "same_phenomenon", ("coordinate_terms",), "coordinate_terms"),
    FermionTier(6, "same_phenomenon", ("synonyms",), "synonyms"),
)

# Above this many vectors, use hnswlib ANN instead of brute-force O(n²).
ANN_THRESHOLD: int = int(os.environ.get("ANN_THRESHOLD", 20_000))
ANN_K: int = int(os.environ.get("ANN_K", 20))
ANN_EF: int = int(os.environ.get("ANN_EF", 100))
ANN_M: int = int(os.environ.get("ANN_M", 16))
ANN_EF_CONSTRUCTION: int = int(os.environ.get("ANN_EF_CONSTRUCTION", 100))
ANN_BUILD_CHUNK: int = int(os.environ.get("ANN_BUILD_CHUNK", 50_000))

# ANN search runs on a projected subspace, not the full embedding dimension.
# The stored/graph embeddings stay at EMBED_DIM (e.g. 4096 for qwen), but the
# HNSW index itself would need ~16 KB/point at 4096-dim (≈210 GB for 12.7M
# senses) — far more than this box's RAM. Projecting the *match* down to
# MATCH_DIM (default 1024 = 4096/4) shrinks the HNSW index to a viable ~55 GB.
# Neighbor rows map back to the full-dim vectors unchanged; only the cosine
# *scores* used for ranking are measured in the reduced space (approximate,
# consistent with HNSW being approximate anyway).
MATCH_DIM: int = int(os.environ.get("MATCH_DIM", 1024))
# Fixed seed so projection is deterministic across runs.
_PROJECT_SEED = 0xC0FFEE

_proj_cache: dict[int, np.ndarray] = {}


def _gaussian_projector(dim: int) -> np.ndarray:
    """Seeded Gaussian random projection matrix (MATCH_DIM, dim), cached.

    Johnson–Lindenstrauss-style: a fixed random matrix roughly preserves
    pairwise cosine structure, so nearest-neighbour discovery in the reduced
    space tracks the full space well enough for ranking.
    """
    proj = _proj_cache.get(dim)
    if proj is None:
        rng = np.random.default_rng(_PROJECT_SEED)
        # Normalise columns approx. so projection ~ rotation+scale (keeps
        # cosine structure near-identity for orthonormal-ish + small noise).
        m = rng.standard_normal((MATCH_DIM, dim)).astype(np.float32)
        proj = m / np.sqrt(dim)  # scale so E[||xP||²] ≈ ||x||²
        _proj_cache[dim] = proj
    return proj


def _project_match(vectors: np.ndarray) -> np.ndarray:
    """Reduce ``vectors`` to MATCH_DIM for ANN search (row indices unchanged)."""
    dim = vectors.shape[1]
    if dim <= MATCH_DIM:
        return vectors
    proj = _gaussian_projector(dim)
    red = vectors @ proj.T
    norms = np.linalg.norm(red, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return red / norms


def top_k_pairs(
    vectors: np.ndarray,
    k: int | None = None,
    min_score: float = 0.0,
) -> Iterable[tuple[int, int, float]]:
    """Yield (i, j, cos) with i<j, sorted by descending cosine.

    Rows of ``vectors`` must be L2-normalised. Uses brute-force for n ≤
    ANN_THRESHOLD and hnswlib HNSW otherwise.

    ``min_score`` pre-filters pairs below the threshold before sorting,
    avoiding an O(n log n) sort on tens of millions of low-score pairs.
    """
    n = vectors.shape[0]
    if n < 2:
        return
    if n <= ANN_THRESHOLD:
        yield from _brute_force_pairs(vectors, k)
    else:
        yield from _ann_pairs(vectors, k, min_score=min_score)


def _brute_force_pairs(
    vectors: np.ndarray, k: int | None
) -> Iterable[tuple[int, int, float]]:
    sims = vectors @ vectors.T
    iu = np.triu_indices(vectors.shape[0], k=1)
    scores = sims[iu]
    order = np.argsort(-scores)
    if k is not None:
        order = order[:k]
    for idx in order:
        yield int(iu[0][idx]), int(iu[1][idx]), float(scores[idx])


def _ann_pairs(
    vectors: np.ndarray, k: int | None, min_score: float = 0.0
) -> Iterable[tuple[int, int, float]]:
    """ANN pair search via hnswlib HNSW index.

    hnswlib cosine space uses distance = 1 − cosine for normalised vectors,
    so cosine_similarity = 1 − distance.
    """
    import hnswlib  # optional at import time; always present in production

    n, dim = vectors.shape
    reduced = _project_match(vectors) if dim > MATCH_DIM else vectors
    match_dim = reduced.shape[1]
    if match_dim != dim:
        _log.info("ANN match: projecting %d -> %d dims", dim, match_dim)
    nn_k = min(n - 1, ANN_K)
    ef = max(ANN_EF, (nn_k + 1) * 2)

    _log.info("HNSW init: n=%d dim=%d ef_construction=%d M=%d ANN_K=%d",
              n, match_dim, ANN_EF_CONSTRUCTION, ANN_M, nn_k)
    index = hnswlib.Index(space="cosine", dim=match_dim)
    index.init_index(max_elements=n, ef_construction=ANN_EF_CONSTRUCTION, M=ANN_M)

    # Add in chunks so we get periodic progress log lines instead of one
    # silent multi-hour call.
    chunk = ANN_BUILD_CHUNK
    t_build_start = time.monotonic()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        index.add_items(reduced[start:end], np.arange(start, end))
        elapsed = time.monotonic() - t_build_start
        rate = end / elapsed if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else float("inf")
        _log.info("HNSW build: %d/%d (%.0f%%) elapsed=%.0fs rate=%.0f/s eta=%.0fs",
                  end, n, 100 * end / n, elapsed, rate, eta)

    _log.info("HNSW build done in %.0fs", time.monotonic() - t_build_start)
    index.set_ef(ef)

    _log.info("HNSW knn_query: n=%d k=%d ef=%d", n, nn_k + 1, ef)
    t_query = time.monotonic()
    labels, distances = index.knn_query(reduced, k=nn_k + 1)  # +1: self is returned
    _log.info("HNSW knn_query done in %.0fs", time.monotonic() - t_query)

    _log.info("dedup pairs...")
    gc.disable()
    try:
        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int, float]] = []
        for i in range(n):
            for j, dist in zip(labels[i], distances[i], strict=True):
                j = int(j)
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                cos = 1.0 - float(dist)
                if cos >= min_score:
                    pairs.append((a, b, cos))
        _log.info("dedup done: %d unique pairs above %.3f", len(pairs), min_score)
        _log.info("sorting %d pairs...", len(pairs))
        pairs.sort(key=operator.itemgetter(2), reverse=True)
    finally:
        gc.enable()
    if k is not None:
        pairs = pairs[:k]
    yield from pairs


def project_match(vectors: np.ndarray) -> np.ndarray:
    """Public alias for reducing ``vectors`` to MATCH_DIM (see _project_match)."""
    return _project_match(vectors)


def ann_index(
    db: np.ndarray,
    *,
    preprojected: bool = False,
    log_build: bool = True,
    ef: int | None = None,
):
    """Build an hnswlib index over ``db`` for k-NN row searches.

    ``db`` rows may be full embedding dim; unless ``preprojected`` they are
    projected to MATCH_DIM first (callers that already hold the projected
    matrix pass ``preprojected=True`` so the projection isn't duplicated).
    Cosine space = rows are L2-normalised (project_match normalises). Returns
    the index; query with ``index.knn_query(q, k=k)`` → (labels, dist).
    """
    import hnswlib  # optional at import time; always present in production

    b = db if preprojected else _project_match(db)
    n = b.shape[0]
    index = hnswlib.Index(space="cosine", dim=b.shape[1])
    index.init_index(max_elements=n, ef_construction=ANN_EF_CONSTRUCTION, M=ANN_M)
    t_build_start = time.monotonic()
    chunk = ANN_BUILD_CHUNK
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        index.add_items(b[start:end], np.arange(start, end))
        if log_build:
            elapsed = time.monotonic() - t_build_start
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n - end) / rate if rate > 0 else float("inf")
            _log.info("HNSW build: %d/%d (%.0f%%) elapsed=%.0fs rate=%.0f/s eta=%.0fs",
                      end, n, 100 * end / n, elapsed, rate, eta)
    if log_build:
        _log.info("HNSW build done in %.0fs", time.monotonic() - t_build_start)
    index.set_ef(ef if ef is not None else max(ANN_EF, (ANN_K + 1) * 2))
    return index


def greedy_unique_match(
    pairs: Iterable[tuple[int, int, float]],
    threshold: float,
    *,
    same_word: set[frozenset[int]] | None = None,
    polysemy_floor: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Greedy highest-first selection honoring the unique-parent constraint.

    ``same_word`` lets the caller mark same-headword sense pairs whose
    effective q is floored at ``polysemy_floor`` (v0.4 §7.3); they are
    still subject to ``threshold`` after flooring.
    """
    same_word = same_word or set()
    taken: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for i, j, q in pairs:
        if i in taken or j in taken:
            continue
        eff_q = max(q, polysemy_floor) if frozenset((i, j)) in same_word else q
        if eff_q < threshold:
            continue
        taken.add(i)
        taken.add(j)
        out.append((i, j, eff_q))
    return out


def nearest_row(query: np.ndarray, matrix: np.ndarray) -> tuple[int, float]:
    """Index + cosine of the nearest row in ``matrix`` to ``query``."""
    sims = matrix @ np.asarray(query, dtype=np.float32)
    best = int(np.argmax(sims))
    return best, float(sims[best])
