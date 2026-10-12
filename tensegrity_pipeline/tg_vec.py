"""Tensegrity-modified vector equations.

The PoC barycentric vector is::

    bary_vec = normalize( q·v_cm1 + q·v_cm2 + (1 − q)·v_type )

where ``v_type`` encodes the *edge type* and ``q`` the connection strength. The
tensegrity version replaces the edge-type term with a **signed role term** so
the vector itself encodes whether the relation is complementary (attraction /
"compression") or contradictory (repulsion / "tension"):

    complement:  normalize( q·v_cm1 + q·v_cm2 + (1 − q)·v_role )
    contradict:  normalize( q·v_cm1 + q·v_cm2 − (1 − q)·v_role )

``v_role`` is the embedding of a fixed per-role sentence (embedded once per
run). ``q`` remains the embedding cosine of the two participants; the role
itself — and its probability — come from the tev1 decision model.

MetaBary keeps the PoC "Born rule" for ``q_mb_raw`` but applies the same role
sign to the bridge term::

    dir  = normalize( w1·v1 + w2·v2 ± w3·v_bridge )
    q_mb_raw = w3² / sqrt(w1⁴ + w2⁴ + w3⁴)
"""

from __future__ import annotations

import numpy as np

from lib.bary_vec import level_factor, normalize  # noqa: F401  (re-exported)

COMPLEMENT = "complement"
CONTRADICT = "contradict"
ROLES: tuple[str, ...] = (COMPLEMENT, CONTRADICT)

# Fixed role sentences — embedded once, used as v_role. Kept deliberately
# parallel in structure so the only difference the embedder sees is the
# complementary/contradictory stance.
ROLE_SENTENCES: dict[str, str] = {
    COMPLEMENT: (
        "these two statements are complementary: they are consistent with each "
        "other and reinforce the same meaning"
    ),
    CONTRADICT: (
        "these two statements are contradictory: they are mutually exclusive "
        "and cannot both be true"
    ),
}


def role_sign(role: str | None) -> float:
    """+1 for complement (or unknown), −1 for contradict."""
    if role == CONTRADICT:
        return -1.0
    return 1.0


def compute_bary_vec(
    v_cm1: np.ndarray,
    v_cm2: np.ndarray,
    v_role: np.ndarray,
    q: float,
    role: str | None = COMPLEMENT,
) -> np.ndarray:
    """Signed-role BaryEdge vector (L14/L15)."""
    v_cm1 = np.asarray(v_cm1, dtype=np.float32)
    v_cm2 = np.asarray(v_cm2, dtype=np.float32)
    v_role = np.asarray(v_role, dtype=np.float32)
    raw = q * v_cm1 + q * v_cm2 + role_sign(role) * (1.0 - q) * v_role
    return normalize(raw)


def compute_metabary_vec(
    v_be1: np.ndarray,
    v_be2: np.ndarray,
    v_bridge: np.ndarray,
    w1: float,
    w2: float,
    w3: float,
    role: str | None = COMPLEMENT,
) -> tuple[np.ndarray, float]:
    """MetaBary (L≤13): role-signed direction + Born-rule ``q_mb_raw``.

    Returns ``(vector, q_mb_raw)``. The caller stores ``q_mb_raw`` as
    ``connection_strength`` and multiplies by :func:`level_factor` for
    ``accumulated_weight``.
    """
    v_be1 = np.asarray(v_be1, dtype=np.float32)
    v_be2 = np.asarray(v_be2, dtype=np.float32)
    v_bridge = np.asarray(v_bridge, dtype=np.float32)
    raw = w1 * v_be1 + w2 * v_be2 + role_sign(role) * w3 * v_bridge
    vec = normalize(raw)
    denom = float(np.sqrt(w1**4 + w2**4 + w3**4))
    q_mb_raw = (w3 * w3) / denom if denom > 0.0 else 0.0
    return vec, q_mb_raw


def title_vector(
    be_vecs: list[np.ndarray], orphan_vecs: list[np.ndarray]
) -> np.ndarray:
    """v(title) = normalize( Σ v(BE_i) + Σ v(orphan_sentence_j) ).

    Mirrors the PoC word-vector equation (lib.bary_vec.word_vector): no
    embedding call — L14 titles are derived from the L15 BaryEdges that parent
    their sentences plus their still-orphan sentences.
    """
    parts = [np.asarray(v, dtype=np.float32) for v in be_vecs]
    parts += [np.asarray(v, dtype=np.float32) for v in orphan_vecs]
    if not parts:
        raise ValueError("title_vector requires at least one component vector")
    return normalize(np.sum(np.stack(parts, axis=0), axis=0))
