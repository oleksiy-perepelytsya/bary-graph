"""Regression test for the float64-projection write corruption (2026-10-08).

s07 absorb and s08 pool writers projected rows with ``row @ proj.T`` where
``proj`` (``_gaussian_projector``) is float64, then wrote ``rp.tobytes()`` to
a binary file that was reopened as a float32 memmap. float64 bytes reread as
float32 are garbage (every other 4-byte half of each 8-byte float) — that
silently produced an arbitrary-match absorb pass and a 0-pair s08 run.

The writers now cast to float32 before writing; this test pins the
round-trip invariant and documents the failure mode.
"""

import numpy as np
import pytest

from lib.match import _gaussian_projector

DIM = 1024


def _project(row: np.ndarray, proj: np.ndarray, cast: bool) -> np.ndarray:
    rp = row @ proj.T
    if cast:
        rp = rp.astype(np.float32)
    norm = float(np.linalg.norm(rp))
    return rp / norm if norm else rp


def test_projected_row_is_float32_and_roundtrips():
    rng = np.random.default_rng(7)
    row = rng.standard_normal(DIM).astype(np.float32)
    proj = _gaussian_projector(DIM)  # float64 by design
    assert proj.dtype == np.float64

    rp = _project(row, proj, cast=True)
    assert rp.dtype == np.float32

    # float32 bytes reread as float32 == the in-memory row (exact file layout
    # the memmaps reopen).
    back = np.frombuffer(rp.tobytes(), dtype=np.float32)
    assert len(back) == DIM
    assert np.array_equal(back, rp)


def test_uncast_float64_write_is_misread_as_float32():
    rng = np.random.default_rng(7)
    row = rng.standard_normal(DIM).astype(np.float32)
    proj = _gaussian_projector(DIM)

    old = _project(row, proj, cast=False)          # the bug: float64
    broken = np.frombuffer(old.tobytes(), dtype=np.float32)[: 8]
    intended = _project(row, proj, cast=True)[: 8]
    # Misreading float64 bytes as float32 yields unrelated garbage, not the
    # intended values. (Exact inequality holds essentially always; run
    # deterministically via fixed seed.)
    assert not np.allclose(broken, intended)
    # And the misread values are not normalized unit vectors either.
    assert not np.isclose(np.linalg.norm(broken), 1.0, atol=1e-3)


def test_gaussian_projector_is_seeded_deterministic():
    a = _gaussian_projector(DIM)
    b = _gaussian_projector(DIM)
    assert np.array_equal(a, b)