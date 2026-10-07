"""Crash-resume helpers for s04: the greedy-pairs bundle and pending filter.

The bundle is what lets a run that died mid-embed skip the ~43 min
HNSW build + knn sweep + greedy match; the filter is what keeps a rerun
from double-inserting pairs whose BE already landed.
"""

from __future__ import annotations

import logging

import pytest

import scripts.s04_l15_edges as s04


class _Settings:
    def __init__(self, q_min_l15: float = 0.42, polysemy_q_floor: float = 0.3):
        self.q_min_l15 = q_min_l15
        self.polysemy_q_floor = polysemy_q_floor


class _FakeColl:
    """Only the one query _filter_pending issues: existing inferred L15 BEs."""

    def __init__(self, bes: list[dict]):
        self._bes = bes

    def find(self, *_args, **_kwargs):
        return list(self._bes)


@pytest.fixture
def bundle_path(tmp_path, monkeypatch):
    path = tmp_path / "s04_pairs.npz"
    monkeypatch.setattr(s04, "S04_PAIRS_BUNDLE", str(path))
    return path


@pytest.fixture
def log():
    return logging.getLogger("test-s04")


def test_bundle_roundtrip(bundle_path, log):
    pairs = [(0, 1, 0.9), (2, 3, 0.8), (4, 5, 0.7)]
    s04._save_pairs_bundle(pairs, 10, _Settings(), log)
    assert bundle_path.exists()
    assert s04._load_pairs_bundle(10, _Settings(), log) == pairs


def test_bundle_missing_returns_none(tmp_path, monkeypatch, log):
    monkeypatch.setattr(s04, "S04_PAIRS_BUNDLE", str(tmp_path / "nope.npz"))
    assert s04._load_pairs_bundle(10, _Settings(), log) is None


def test_bundle_guards_sense_count(bundle_path, log):
    s04._save_pairs_bundle([(0, 1, 0.9)], 10, _Settings(), log)
    assert s04._load_pairs_bundle(11, _Settings(), log) is None


def test_bundle_guards_thresholds(bundle_path, log):
    s04._save_pairs_bundle([(0, 1, 0.9)], 10, _Settings(), log)
    assert s04._load_pairs_bundle(10, _Settings(q_min_l15=0.5), log) is None
    assert s04._load_pairs_bundle(10, _Settings(polysemy_q_floor=0.4), log) is None


def test_bundle_corrupt_file_degrades(bundle_path, log):
    bundle_path.write_bytes(b"not an npz")
    assert s04._load_pairs_bundle(10, _Settings(), log) is None


def test_filter_drops_embedded_pairs(bundle_path, log):
    ids = [f"id{i}" for i in range(10)]
    coll = _FakeColl([
        {"_id": "be1", "cm1_id": "id0", "cm2_id": "id1", "q": 0.9},
    ])
    paired: set[int] = set()
    be_ids: list = []
    be_q: list[float] = []
    pending = s04._filter_pending(
        [(0, 1, 0.9), (2, 3, 0.8)], coll, ids, paired, be_ids, be_q,
        adopt=True, log=log,
    )
    assert pending == [(2, 3, 0.8)]
    # done pair's senses must not be re-orphaned by 4e, and its BE joins the pool
    assert paired == {0, 1}
    assert be_ids == ["be1"] and be_q == [0.9]


def test_filter_adopt_false_leaves_pool_alone(log):
    ids = [f"id{i}" for i in range(10)]
    coll = _FakeColl([
        {"_id": "be1", "cm1_id": "id1", "cm2_id": "id0", "q": 0.9},  # reversed
    ])
    paired: set[int] = set()
    be_ids: list = ["already-there"]
    be_q: list[float] = [0.5]
    pending = s04._filter_pending(
        [(0, 1, 0.9), (2, 3, 0.8)], coll, ids, paired, be_ids, be_q,
        adopt=False, log=log,
    )
    assert pending == [(2, 3, 0.8)]
    assert paired == {0, 1}
    assert be_ids == ["already-there"]


def test_filter_empty_pairs_short_circuits(log):
    assert s04._filter_pending([], _FakeColl([]), [], set(), [], [],
                                adopt=True, log=log) == []


# --- _parallel_embed_batches -------------------------------------------------


def test_parallel_batch_order_matches_sequential():
    chunks = [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]]
    fn = lambda ch: [x * 2 for x in ch]  # noqa: E731
    seq = list(s04._parallel_embed_batches(chunks, fn, workers=1))
    par = list(s04._parallel_embed_batches(chunks, fn, workers=4))
    assert seq == par == [(c, fn(c)) for c in chunks]


def test_parallel_empty_input():
    assert list(s04._parallel_embed_batches([], lambda c: c, workers=4)) == []


def test_parallel_overlaps_workers():
    """With workers>1 more than one embed_fn must be in flight at once."""
    import threading
    import time

    lock = threading.Lock()
    in_flight = {"now": 0, "peak": 0}

    def slow(_chunk):
        with lock:
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        time.sleep(0.05)
        with lock:
            in_flight["now"] -= 1
        return _chunk

    out = list(s04._parallel_embed_batches(range(8), slow, workers=4))
    assert [c for c, _ in out] == list(range(8))
    assert in_flight["peak"] >= 2


def test_parallel_propagates_worker_error():
    def boom(chunk):
        if chunk == 3:
            raise RuntimeError("embed failed")
        return chunk

    gen = s04._parallel_embed_batches([1, 2, 3, 4], boom, workers=2)
    assert next(gen) == (1, 1)
    assert next(gen) == (2, 2)
    with pytest.raises(RuntimeError, match="embed failed"):
        next(gen)
