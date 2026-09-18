from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

# Nominal embedding dimension for the default nomic model. Runtime dimension
# is authoritative via Settings.embed_dim (nomic→768, qwen3-embedding:8b→4096);
# bary_vec math is dimension-agnostic.
EMBED_DIM = 768

# L14 edge-type sentences — embedded once, used as v(type). See CLAUDE.md.
TYPE_SENTENCES: dict[str, str] = {
    "same_phenomenon": "these two words describe the same concept",
    "contradicts": "these two words have opposite meanings",
    "extends": "one word is derived from or extends the other",
    "applies_to": "these two words share a common origin or root",
    "is_instance_of": "this relationship is a specific instance of the broader relationship",
    # s07b default pairing (general, complementary): no synonymy/antonymy claim.
    "same_lang": "these two words are related and occur in the same language",
    "cross_lang": "these two words name related concepts in different languages",
}


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def compute_bary_vec(
    v_cm1: np.ndarray, v_cm2: np.ndarray, v_type: np.ndarray, q: float
) -> np.ndarray:
    """bary_vec = normalize( q·v_cm1 + q·v_cm2 + (1−q)·v_type )  — L14/L15."""
    v_cm1 = np.asarray(v_cm1, dtype=np.float32)
    v_cm2 = np.asarray(v_cm2, dtype=np.float32)
    v_type = np.asarray(v_type, dtype=np.float32)
    raw = q * v_cm1 + q * v_cm2 + (1.0 - q) * v_type
    return normalize(raw)


def level_factor(level: int, alpha: float = 0.5) -> float:
    """Amplification factor for MetaBary accumulated_weight: 1 + α·(14 − L) / 13.

    L13 → 1.0 (no boost), L1 → 1 + α (maximum). Default α = 0.5.
    """
    return 1.0 + alpha * (14 - level) / 13


def compute_metabary_vec(
    v_be1: np.ndarray,
    v_be2: np.ndarray,
    v_bridge: np.ndarray,
    w1: float,
    w2: float,
    w3: float,
) -> tuple[np.ndarray, float]:
    """MetaBary (L13+): direction uses accumulated_weights; returns (vec, q_mb_raw).

    Step 1 — vector direction: normalize( w1·v_be1 + w2·v_be2 + w3·v_bridge )
    Step 2 — q_mb_raw (Born rule): w3² / sqrt(w1⁴ + w2⁴ + w3⁴)

    w1/w2/w3 are accumulated_weight values of the children. The caller stores
    q_mb_raw as connection_strength and multiplies by level_factor(L) for
    accumulated_weight.
    """
    v_be1 = np.asarray(v_be1, dtype=np.float32)
    v_be2 = np.asarray(v_be2, dtype=np.float32)
    v_bridge = np.asarray(v_bridge, dtype=np.float32)
    raw = w1 * v_be1 + w2 * v_be2 + w3 * v_bridge
    vec = normalize(raw)
    denom = float(np.sqrt(w1**4 + w2**4 + w3**4))
    q_mb_raw = (w3 * w3) / denom if denom > 0.0 else 0.0
    return vec, q_mb_raw


def word_vector(
    be_vecs: Iterable[np.ndarray],
    orphan_sense_vecs: Iterable[np.ndarray],
) -> np.ndarray:
    """v(word) = normalize( Σ v(BE_i) + Σ v(orphan_sense_j) ).

    No embedding call. BE-centroid + orphan senses only — see v0.5 §2.4.
    """
    parts = [np.asarray(v, dtype=np.float32) for v in be_vecs]
    parts += [np.asarray(v, dtype=np.float32) for v in orphan_sense_vecs]
    if not parts:
        raise ValueError("word_vector requires at least one component vector")
    raw = np.sum(np.stack(parts, axis=0), axis=0)
    return normalize(raw)


def build_l15_type_text(
    word_a: str,
    ant_a: Sequence[str],
    syn_a: Sequence[str],
    word_b: str,
    ant_b: Sequence[str],
    syn_b: Sequence[str],
) -> str:
    """Per-pair lexical-neighborhood string used as v(type) source at L15."""

    def part(w: str, ants: Sequence[str], syns: Sequence[str]) -> str:
        segs = []
        if ants:
            segs.append("antonyms: " + ", ".join(ants))
        if syns:
            segs.append("synonyms: " + ", ".join(syns))
        inner = "; ".join(segs) if segs else ""
        return f"{w} ({inner})" if inner else w

    return f"{part(word_a, ant_a, syn_a)}; {part(word_b, ant_b, syn_b)}"
