from __future__ import annotations

import numpy as np

from tensegrity_pipeline.tg_vec import (
    APPLIES_TO,
    CONTRADICTS,
    EXTENDS,
    IS_INSTANCE_OF,
    ROLE_SENTENCES,
    ROLES,
    SAME_PHENOMENON,
    compute_bary_vec,
    compute_metabary_vec,
    role_sign,
    title_vector,
)


def test_roles_are_the_kaikki_relation_types_in_fermion_order():
    assert ROLES == (
        CONTRADICTS,
        APPLIES_TO,
        IS_INSTANCE_OF,
        EXTENDS,
        SAME_PHENOMENON,
    )


def test_role_sentences_are_written_about_sentences():
    # The participants tensegrity pairs are sentences, so v_role is embedded
    # from sentence-oriented text, not the PoC's word-oriented sentences.
    assert set(ROLE_SENTENCES) == set(ROLES)
    assert len(set(ROLE_SENTENCES.values())) == len(ROLES)
    for text in ROLE_SENTENCES.values():
        assert "sentence" in text  # singular or plural
        assert "words" not in text


def test_role_sign_is_negative_only_for_contradicts():
    assert role_sign(CONTRADICTS) == -1.0
    assert role_sign(APPLIES_TO) == 1.0
    assert role_sign(IS_INSTANCE_OF) == 1.0
    assert role_sign(EXTENDS) == 1.0
    assert role_sign(SAME_PHENOMENON) == 1.0
    assert role_sign(None) == 1.0  # unknown falls back to attractive


def test_compute_bary_vec_contradicts_flips_the_role_term():
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    v_role = np.array([1.0, 0.0], dtype=np.float32)
    q = 0.5

    same = compute_bary_vec(v1, v2, v_role, q, SAME_PHENOMENON)
    contra = compute_bary_vec(v1, v2, v_role, q, CONTRADICTS)

    # same_phenomenon: 0.5·v1 + 0.5·v2 + 0.5·v_role = [1.0, 0.5]
    # contradicts:     0.5·v1 + 0.5·v2 − 0.5·v_role = [0.0, 0.5]
    assert same[0] > 0
    assert abs(contra[0]) < 1e-6
    assert contra[1] > 0
    assert not np.allclose(same, contra)


def test_compute_bary_vec_attractive_relations_share_a_sign():
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    v_role = np.array([1.0, 0.0], dtype=np.float32)
    q = 0.5

    vecs = [compute_bary_vec(v1, v2, v_role, q, r) for r in ROLES if r != CONTRADICTS]
    for v in vecs:
        assert np.allclose(v, vecs[0])


def test_compute_bary_vec_is_unit_norm():
    rng = np.random.default_rng(0)
    v1 = rng.standard_normal(16).astype(np.float32)
    v2 = rng.standard_normal(16).astype(np.float32)
    v_role = rng.standard_normal(16).astype(np.float32)
    for role in ROLES:
        out = compute_bary_vec(v1, v2, v_role, 0.7, role)
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


def test_compute_metabary_vec_born_rule_q():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    bridge = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    vec, q_raw = compute_metabary_vec(a, b, bridge, 1.0, 1.0, 1.0, SAME_PHENOMENON)
    # q_mb_raw = w3² / sqrt(w1⁴ + w2⁴ + w3⁴) = 1 / sqrt(3)
    assert abs(q_raw - 1.0 / np.sqrt(3.0)) < 1e-6
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_compute_metabary_vec_role_sign_flips_bridge_term():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    bridge = np.array([0.0, 1.0], dtype=np.float32)

    same, _ = compute_metabary_vec(a, b, bridge, 1.0, 1.0, 1.0, SAME_PHENOMENON)
    contra, _ = compute_metabary_vec(a, b, bridge, 1.0, 1.0, 1.0, CONTRADICTS)
    # same_phenomenon: [1,1] direction; contradicts: [1,-1] direction.
    assert same[1] > 0
    assert contra[1] < 0


def test_title_vector_normalizes_sum():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    out = title_vector([a], [b])
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6
    assert np.allclose(out, [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
