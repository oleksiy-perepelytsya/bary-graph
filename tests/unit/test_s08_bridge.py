"""Bridge-assignment semantics for s08 ``_form_level`` (changed 2026-10-07).

No brute-force rescue: only unparented top-window ANNs qualify (k=50 in the
ANN branch with a k=1 retry; the top-``_K`` window in the small-pool branch).
Pairs whose window is exhausted are dropped — ``_form_level`` returns fewer
triads and the children stay unparented for s09's rescue round.

The tests monkeypatch ``_load_unparented_bes`` with synthetic normalized
matrices and run with ``dry_run=True`` so no MongoDB is touched.
"""

from __future__ import annotations

import numpy as np

import scripts.s08_metabary as s08

THR = 0.9


def _arc(n: int, spread_deg: float, offset_deg: float = 0.0) -> np.ndarray:
    """n L2-normalized 2-D vectors within an arc.

    Intra-arc mutual cosine >= cos(spread) (>= 0.9 for spread <= ~25 deg,
    so greedy pairing forms a complete matching); two arcs 90 deg apart
    never cross-match at the 0.9 threshold.
    """
    ang = np.deg2rad(
        np.linspace(offset_deg, offset_deg + spread_deg, n, dtype=np.float64)
    )
    return np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)


def _run(monkeypatch, children: np.ndarray, bridges: np.ndarray,
         ann_threshold: int | None = None) -> int:
    """Run _form_level dry on the synthetic pools; return triads formed."""
    if ann_threshold is not None:
        monkeypatch.setattr(s08, "ANN_THRESHOLD", ann_threshold)

    child_ids = list(range(len(children)))
    bridge_ids = list(range(10_000, 10_000 + len(bridges)))

    def _fake_load(coll, level, embed_dim, tag, cap=None):
        ids, V = ((child_ids, children) if tag.startswith("C")
                  else (bridge_ids, bridges))
        return ids, [{} for _ in ids], V, V, None, None

    monkeypatch.setattr(s08, "_load_unparented_bes", _fake_load)
    return s08._form_level(None, child_level=15, bridge_level=14,
                           threshold=THR, alpha=0.5, dry_run=True,
                           embed_dim=2)


# --------------------------------------------------------------------------
# Small-pool branch (n_bridges <= ANN_THRESHOLD): top-_K window, no rescue.
# --------------------------------------------------------------------------

def test_small_branch_forms_all_pairs(monkeypatch):
    children = np.vstack([_arc(2, 10.0), _arc(2, 10.0, offset_deg=90.0)])
    bridges = _arc(3, 15.0)
    # 2 cross-group-isolated pairs, 3 available bridges → 2 triads, 0 dropped.
    assert _run(monkeypatch, children, bridges) == 2


def test_small_branch_drops_pair_when_window_exhausted(monkeypatch):
    children = np.vstack([_arc(2, 10.0), _arc(2, 10.0, offset_deg=90.0)])
    bridges = _arc(1, 5.0)
    # 2 pairs compete for 1 bridge → exactly 1 triad; the loser is dropped
    # (old code rescued it with a full-matrix scan).
    assert _run(monkeypatch, children, bridges) == 1


def test_small_branch_zero_bridges_forms_nothing(monkeypatch):
    children = np.vstack([_arc(2, 10.0), _arc(2, 10.0, offset_deg=90.0)])
    assert _run(monkeypatch, children, np.zeros((0, 2), np.float32)) == 0


# --------------------------------------------------------------------------
# ANN branch (n_bridges > ANN_THRESHOLD): k=50 query, k=1 retry, drop.
# --------------------------------------------------------------------------

def test_ann_branch_accepts_every_pair_that_finds_a_bridge(monkeypatch):
    children = _arc(6, 15.0)   # 3 pairs (complete graph)
    bridges = _arc(6, 10.0)    # > ANN_THRESHOLD=4 → HNSW path
    assert _run(monkeypatch, children, bridges, ann_threshold=4) == 3


def test_ann_branch_drops_pair_when_no_unparented_left(monkeypatch):
    children = _arc(14, 20.0)  # 7 pairs
    bridges = _arc(6, 10.0)    # only 6 bridges
    # 6 pairs consume every bridge; the 7th exhausts k=50 AND the k=1
    # retry (index empty) → dropped. Old code full-scanned and rescued it.
    assert _run(monkeypatch, children, bridges, ann_threshold=4) == 6


def test_ann_branch_k1_retry_salvages_sparse_region(monkeypatch):
    children = _arc(112, 20.0)  # 56 pairs (all mutual cos >= cos(20°))
    bridges = _arc(60, 15.0)    # k = min(50, 60) = 50
    # Pairs 1..55 consume 55 bridges → only 5 valid remain, so the k=50
    # query fails (RuntimeError or partial labels); the k=1 retry must
    # salvage the nearest unparented bridge → all 56 pairs form triads.
    assert _run(monkeypatch, children, bridges, ann_threshold=4) == 56
