from __future__ import annotations

import numpy as np

from tensegrity_pipeline.tg_vec import (
    COMPLEMENT,
    CONTRADICT,
    ROLE_SENTENCES,
    compute_bary_vec,
    compute_metabary_vec,
    role_sign,
    title_vector,
)


def test_role_sign():
    assert role_sign(COMPLEMENT) == 1.0
    assert role_sign(CONTRADICT) == -1.0
    assert role_sign(None) == 1.0  # unknown falls back to attractive


def test_role_sentences_present_and_distinct():
    assert set(ROLE_SENTENCES) == {COMPLEMENT, CONTRADICT}
    assert ROLE_SENTENCES[COMPLEMENT] != ROLE_SENTENCES[CONTRADICT]


def test_compute_bary_vec_complement_vs_contradict_differ():
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    v_role = np.array([1.0, 0.0], dtype=np.float32)
    q = 0.5

    comp = compute_bary_vec(v1, v2, v_role, q, COMPLEMENT)
    contra = compute_bary_vec(v1, v2, v_role, q, CONTRADICT)

    # complement: 0.5·v1 + 0.5·v2 + 0.5·v_role = [1.0, 0.5]
    # contradict: 0.5·v1 + 0.5·v2 − 0.5·v_role = [0.0, 0.5]
    assert comp[0] > 0
    assert abs(contra[0]) < 1e-6
    assert contra[1] > 0
    assert not np.allclose(comp, contra)


def test_compute_bary_vec_is_unit_norm():
    rng = np.random.default_rng(0)
    v1 = rng.standard_normal(16).astype(np.float32)
    v2 = rng.standard_normal(16).astype(np.float32)
    v_role = rng.standard_normal(16).astype(np.float32)
    for role in (COMPLEMENT, CONTRADICT):
        out = compute_bary_vec(v1, v2, v_role, 0.7, role)
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


def test_compute_metabary_vec_born_rule_q():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    bridge = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    vec, q_raw = compute_metabary_vec(a, b, bridge, 1.0, 1.0, 1.0, COMPLEMENT)
    # q_mb_raw = w3² / sqrt(w1⁴ + w2⁴ + w3⁴) = 1 / sqrt(3)
    assert abs(q_raw - 1.0 / np.sqrt(3.0)) < 1e-6
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_compute_metabary_vec_role_sign_flips_bridge_term():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    bridge = np.array([0.0, 1.0], dtype=np.float32)

    comp, _ = compute_metabary_vec(a, b, bridge, 1.0, 1.0, 1.0, COMPLEMENT)
    contra, _ = compute_metabary_vec(a, b, bridge, 1.0, 1.0, 1.0, CONTRADICT)
    # complement: [1,1] direction; contradict: [1,-1] direction.
    assert comp[1] > 0
    assert contra[1] < 0


def test_title_vector_normalizes_sum():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    out = title_vector([a], [b])
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6
    assert np.allclose(out, [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
