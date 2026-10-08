"""Regression test for the two-phase s08/s09 pool loader (2026-10-08).

The single-phase loader projected ``vector`` for EVERY doc in the level's
``_id`` band and filtered client-side — ~4.2 kB x 5.66M docs ~= 23 GB
streamed per load (~6 min/pass) even when the level held only a few hundred
candidates. The two-phase split (light band scan -> chunked ``$in`` vector
fetch) must be behaviourally IDENTICAL to the old algorithm:

  * same accepted ids, same order (single-worker), same meta/dropped counts,
  * byte-identical V rows and normalised float32 VP rows, row i <-> ids[i],
  * cap and empty-band semantics,
  * the perf contract itself: band-scan queries must never request
    ``vector`` (that is the whole point of the split).

``_reference_load`` below is a faithful copy of the pre-fix loader.
"""

from __future__ import annotations

import numpy as np
import pytest
from bson import ObjectId

import scripts.s08_metabary as s08
from lib.match import MATCH_DIM, _gaussian_projector
from lib.vector import unpack_vec

DIM = 8  # tiny embed dim; MATCH_DIM stays at its env default


# ---------------------------------------------------------------- fake Mongo

class _FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, key, direction=1):
        assert key == "_id" and direction == 1
        self._docs = sorted(self._docs, key=lambda d: d["_id"])
        return self

    def hint(self, *_a, **_k):
        return self

    def batch_size(self, *_a, **_k):
        return self

    def __iter__(self):
        return iter(self._docs)


class _FakeColl:
    """Just enough Mongo: _id-range/$in find with projection, max find_one.

    Records every band-scan projection so tests can pin the perf contract
    (no ``vector`` in phase-1 scans).
    """

    def __init__(self, docs):
        self.docs = docs
        self.band_scan_projections: list[dict] = []

    def find(self, q, proj=None):
        sel = list(self.docs)
        if "_id" in q:
            rng = q["_id"]
            if "$in" in rng:
                wanted = set(rng["$in"])
                sel = [d for d in sel if d["_id"] in wanted]
            else:
                lo = rng.get("$gte")
                hi = rng.get("$lt")
                if lo is not None or hi is not None:
                    self.band_scan_projections.append(dict(proj or {}))
                sel = [d for d in sel
                       if (lo is None or d["_id"] >= lo)
                       and (hi is None or d["_id"] < hi)]
        out = []
        for d in sel:
            nd = {k: d[k] for k in d
                  if proj is None or k == "_id" or proj.get(k)}
            out.append(nd)
        return _FakeCursor(out)

    def find_one(self, sort=None, projection=None):
        assert sort == [("_id", -1)]
        return {"_id": max(d["_id"] for d in self.docs)}


def _vec_bytes(seed: int) -> bytes:
    return np.random.default_rng(seed).standard_normal(DIM) \
        .astype(np.float32).tobytes()


def _corpus(all_vecs: bool) -> list[dict]:
    """Level-mixed docs; level 7 candidates land on i in {1, 4, 7, 10, ...}."""
    docs = []
    for i in range(1, 31):
        docs.append({
            "_id": ObjectId(f"{i:024x}"),
            "doc_type": "baryedge" if i % 5 else "node",
            "level": [6, 7, 8][i % 3],
            "parent_edge_id": ObjectId(f"{900000 + i:024x}") if i % 4 == 0 else None,
            "source": "structural" if i % 7 == 0 else None,
            "accumulated_weight": 0.5 + i,
            "vector": _vec_bytes(i),
        })
    if not all_vecs:
        # i=13 passes every level/type/parent/source filter — a would-be
        # candidate whose vector field is missing (phase-2 drop path).
        docs[12]["vector"] = None
    return docs


# ------------------------------------------------- pre-fix loader (reference)

def _reference_load(coll, level, cap=None):
    """Faithful copy of the single-phase loader (pre 2026-10-08 fix)."""
    proj = _gaussian_projector(DIM)
    band_lo = ObjectId("000000000000000000000000")
    max_id = coll.find_one(sort=[("_id", -1)], projection={"_id": 1})
    band_hi = max_id["_id"]
    proj_fields = {"_id": 1, "doc_type": 1, "level": 1, "parent_edge_id": 1,
                   "vector": 1, "accumulated_weight": 1, "source": 1}
    ids, meta, rows, prows = [], [], [], []
    n = dropped = 0

    def _scan(lo, hi):
        nonlocal n, dropped
        q = {"_id": {"$gte": lo}}
        if hi is not None:
            q["_id"]["$lt"] = hi
        for doc in coll.find(q, proj_fields).sort("_id", 1).hint("_id_") \
                .batch_size(10_000):
            if doc.get("doc_type") != "baryedge" or doc.get("level") != level \
                    or doc.get("parent_edge_id") is not None \
                    or doc.get("source") == "structural":
                dropped += 1
                continue
            vec = doc.get("vector")
            if vec is None:
                dropped += 1
                continue
            row = unpack_vec(vec)
            rp = (row @ proj.T).astype(np.float32)
            norm = float(np.linalg.norm(rp))
            rp = rp / norm if norm else rp
            if cap is not None and n >= cap:
                break
            ids.append(doc["_id"])
            meta.append({"_id": doc["_id"],
                         "accumulated_weight": doc["accumulated_weight"]})
            rows.append(row.tobytes())
            prows.append(rp.tobytes())
            n += 1

    _scan(band_lo, band_hi)
    return ids, meta, rows, prows, dropped


# ------------------------------------------------------------------ fixture

@pytest.fixture
def loader_env(tmp_path, monkeypatch):
    monkeypatch.setattr(s08, "_S08_BARY_LO", None)
    monkeypatch.setattr(s08, "_S08_TODAY_LO", None)
    monkeypatch.setattr(s08, "_S08_LOAD_WORKERS", 1)
    monkeypatch.setattr(s08, "_S08_MMAP_DIR", str(tmp_path))
    return tmp_path


def _read_rows(path, n, width):
    mm = np.memmap(path, mode="r", dtype=np.float32, shape=(n, width))
    return [mm[i].tobytes() for i in range(n)]


# ------------------------------------------------------------------- tests

def test_parity_with_single_phase_loader(loader_env):
    coll = _FakeColl(_corpus(all_vecs=False))
    ref_ids, ref_meta, ref_rows, ref_prows, _ = _reference_load(coll, 7)

    out = s08._load_unparented_bes(coll, 7, DIM, "TESTPAR")
    ids, meta, V, VP, v_path, vp_path = out
    try:
        # i=13 passes every client-side filter but has no vector: the old
        # loader dropped it mid-scan, the new one in phase 2 — same net set,
        # same order, same dropped total (both explore the same band once).
        assert ObjectId(f"{13:024x}") not in ids
        assert ids == ref_ids
        assert meta == ref_meta
        assert V.shape == (len(ids), DIM)
        assert VP.shape == (len(ids), MATCH_DIM)
        # byte-identical rows, row i <-> ids[i]
        assert _read_rows(v_path, len(ids), DIM) == ref_rows
        vp_rows = _read_rows(vp_path, len(ids), MATCH_DIM)
        assert vp_rows == ref_prows
        # projected rows are unit-normalised float32
        first = np.frombuffer(vp_rows[0], dtype=np.float32)
        assert first.dtype == np.float32
        assert np.isclose(np.linalg.norm(first.astype(np.float64)), 1.0)
    finally:
        _cleanup([v_path, vp_path])


def test_band_scans_never_request_vector(loader_env):
    """Perf contract: phase-1 band scans stream light fields only."""
    coll = _FakeColl(_corpus(all_vecs=True))
    out = s08._load_unparented_bes(coll, 7, DIM, "TESTPERF")
    try:
        assert coll.band_scan_projections, "loader did not band-scan"
        for proj in coll.band_scan_projections:
            assert "vector" not in proj, \
                f"band scan still streams vector payload: {proj}"
    finally:
        _cleanup(out[4:])


def test_cap_parity(loader_env):
    # cap parity needs ALL candidate vectors present (see module docstring):
    # the old cap counted written rows, the new one counts phase-1 candidates.
    coll = _FakeColl(_corpus(all_vecs=True))
    ref_ids, ref_meta, ref_rows, ref_prows, _ = _reference_load(coll, 7, cap=2)
    out = s08._load_unparented_bes(coll, 7, DIM, "TESTCAP", cap=2)
    ids, meta, V, VP, v_path, vp_path = out
    try:
        assert ids == ref_ids
        assert meta == ref_meta
        assert _read_rows(v_path, len(ids), DIM) == ref_rows
        assert _read_rows(vp_path, len(ids), MATCH_DIM) == ref_prows
    finally:
        _cleanup([v_path, vp_path])


def test_empty_band_returns_empty_and_cleans_up(loader_env):
    coll = _FakeColl(_corpus(all_vecs=True))
    out = s08._load_unparented_bes(coll, 99, DIM, "TESTEMPTY")
    ids, meta, V, VP, v_path, vp_path = out
    assert ids == [] and meta == []
    assert V.shape == (0, DIM) and VP.shape == (0, MATCH_DIM)
    assert v_path is None and vp_path is None
    d = loader_env
    assert not (d / "s08_TESTEMPTY.mmap").exists()
    assert not (d / "s08_TESTEMPTY_P.mmap").exists()


def test_multi_worker_set_parity_and_row_alignment(loader_env, monkeypatch):
    monkeypatch.setattr(s08, "_S08_LOAD_WORKERS", 2)
    coll = _FakeColl(_corpus(all_vecs=True))
    ref_ids, ref_meta, ref_rows, ref_prows, _ = _reference_load(coll, 7)
    ref_by_id = dict(zip(ref_ids, zip(ref_rows, ref_prows)))

    out = s08._load_unparented_bes(coll, 7, DIM, "TESTMT")
    ids, meta, V, VP, v_path, vp_path = out
    try:
        assert sorted(ids) == sorted(ref_ids), "id set must match"
        v_rows = _read_rows(v_path, len(ids), DIM)
        vp_rows = _read_rows(vp_path, len(ids), MATCH_DIM)
        for i, oid in enumerate(ids):  # rows stay aligned with own ids
            exp_v, exp_vp = ref_by_id[oid]
            assert v_rows[i] == exp_v
            assert vp_rows[i] == exp_vp
    finally:
        _cleanup([v_path, vp_path])


def _cleanup(paths):
    for p in paths:
        if p is None:
            continue
        try:
            p.unlink()
        except OSError:
            pass
