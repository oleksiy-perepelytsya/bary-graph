"""Tensegrity-modified vector equations.

The PoC barycentric vector is::

    bary_vec = normalize( q·v_cm1 + q·v_cm2 + (1 − q)·v_type )

where ``v_type`` encodes the *edge type* and ``q`` the connection strength. The
tensegrity version keeps that shape, calls the relation term a **role** (decided
per edge by the tev1 decision model), and gives it a **sign** so the vector
itself encodes attraction ("compression") or repulsion ("tension"):

    any relation except ``contradicts``:
        normalize( q·v_cm1 + q·v_cm2 + (1 − q)·v_role )
    ``contradicts``:
        normalize( q·v_cm1 + q·v_cm2 − (1 − q)·v_role )

``v_role`` is the embedding of a fixed per-relation sentence, taken verbatim
from the PoC's :data:`lib.bary_vec.TYPE_SENTENCES` — so for a given relation the
tensegrity role term *is* the PoC's ``v_type`` and the two stay comparable. ``q``
remains the embedding cosine of the two participants; the relation itself — and
its probability — come from the tev1 decision model.

The vocabulary is the PoC's kaikki-derived edge-type set (fermion order):
``contradicts``, ``applies_to``, ``is_instance_of``, ``extends``,
``same_phenomenon``. ``contradicts`` is the only relation that inverts the
relation term; the other four are attractive, as in the PoC's unsigned form.

MetaBary keeps the PoC "Born rule" for ``q_mb_raw`` but applies the same role
sign to the bridge term::

    dir  = normalize( w1·v1 + w2·v2 ± w3·v_bridge )
    q_mb_raw = w3² / sqrt(w1⁴ + w2⁴ + w3⁴)
"""

from __future__ import annotations

import numpy as np

from lib.bary_vec import TYPE_SENTENCES, level_factor, normalize  # noqa: F401  (re-exported)

# Relation vocabulary — the PoC's kaikki-derived edge types, in fermion priority
# order (spec §7.1: rarer, more informative relations matched first). Every
# tensegrity BaryEdge's ``role`` is one of these. There is deliberately no
# "complement" role: it was tensegrity's own invention, and ``same_phenomenon``
# now plays the affirmative/default part it used to.
CONTRADICTS = "contradicts"
APPLIES_TO = "applies_to"
IS_INSTANCE_OF = "is_instance_of"
EXTENDS = "extends"
SAME_PHENOMENON = "same_phenomenon"

ROLES: tuple[str, ...] = (
    CONTRADICTS,
    APPLIES_TO,
    IS_INSTANCE_OF,
    EXTENDS,
    SAME_PHENOMENON,
)

# ``v_role`` per relation: the PoC's own L14 type sentences, imported verbatim
# rather than re-worded, so a tensegrity role vector is the same vector the PoC
# stores as ``v_type`` for that relation.
ROLE_SENTENCES: dict[str, str] = {role: TYPE_SENTENCES[role] for role in ROLES}


def role_sign(role: str | None) -> float:
    """+1 for every relation except ``contradicts`` (−1); +1 for an unknown or
    absent role, so an unclassified edge stays attractive."""
    return -1.0 if role == CONTRADICTS else 1.0


def compute_bary_vec(
    v_cm1: np.ndarray,
    v_cm2: np.ndarray,
    v_role: np.ndarray,
    q: float,
    role: str | None = SAME_PHENOMENON,
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
    role: str | None = SAME_PHENOMENON,
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
