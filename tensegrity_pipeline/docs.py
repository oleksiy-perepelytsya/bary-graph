"""MongoDB document constructors for the tensegrity collection.

Keeps every stage producing the same schema and makes the level invariants
(sentences=15, titles=14, MB≤13) easy to audit.

BaryEdge/MetaBary documents reuse the PoC physical shape
(``doc_type='baryedge'``, ``cm1_id``/``cm2_id``, BSON subtype-9 ``vector``,
``parent_edge_id``) and add the tensegrity-specific ``role``,
``role_probability`` and ``role_source`` fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from lib.vector import pack_vec
from tensegrity_pipeline.schema import ParsedSentence, ParsedTitle


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _vec(v: np.ndarray | list[float] | None) -> bytes | None:
    if v is None:
        return None
    return pack_vec(np.asarray(v, dtype=np.float32))


def sentence_node(
    s: ParsedSentence, vector: np.ndarray | list[float] | None
) -> dict[str, Any]:
    """L15 source-bearing sentence node."""
    ts = _now()
    return {
        "doc_type": "node",
        "node_type": "sentence",
        "level": 15,
        "label": s.sentence_id,
        "vector": _vec(vector),
        "surface": 1,
        "rotation": 0.0,
        "parent_edge_id": None,
        "properties": {
            "sentence_id": s.sentence_id,
            "text": s.text,
            "doc_id": s.doc_id,
            "paragraph_idx": s.paragraph_idx,
            "sentence_idx": s.sentence_idx,
            "section_idx": s.section_idx,
            "char_start": s.char_start,
            "char_end": s.char_end,
        },
        "created_at": ts,
        "updated_at": ts,
    }


def title_node(t: ParsedTitle) -> dict[str, Any]:
    """L14 title node with placeholder vector (filled by t05)."""
    ts = _now()
    return {
        "doc_type": "node",
        "node_type": "title",
        "level": 14,
        "label": t.title_id,
        "vector": None,
        "surface": 1,
        "rotation": 0.0,
        "parent_edge_id": None,
        "properties": {
            "title_id": t.title_id,
            "text": t.text,
            "doc_id": t.doc_id,
            "section_idx": t.section_idx,
            "title_level": t.title_level,
            "paragraph_idx": t.paragraph_idx,
        },
        "created_at": ts,
        "updated_at": ts,
    }


def tensegrity_baryedge(
    cm1_id: Any,
    cm2_id: Any,
    level: int,
    vector: np.ndarray,
    q: float,
    *,
    role: str | None,
    role_probability: float = 0.0,
    role_source: str = "unclassified",
    accumulated_weight: float | None = None,
    type_vector: np.ndarray | None = None,
    edge_type: str | None = None,
    source: str = "inferred",
    confidence: float = 1.0,
) -> dict[str, Any]:
    """L14/L15 BaryEdge carrying a signed tev1 role.

    ``connection_strength``/``q`` remain the embedding cosine; ``role`` is the
    tev1 argmax and ``role_probability`` its winning probability only.
    """
    ts = _now()
    acc_w = accumulated_weight if accumulated_weight is not None else q
    return {
        "doc_type": "baryedge",
        "cm1_id": cm1_id,
        "cm2_id": cm2_id,
        "level": level,
        "vector": _vec(vector),
        "parent_edge_id": None,
        "connection_strength": float(q),
        "accumulated_weight": float(acc_w),
        "edge_type": edge_type,
        "type_vector": _vec(type_vector),
        "q": float(q),
        "role": role,
        "role_probability": float(role_probability),
        "role_source": role_source,
        "source": source,
        "confidence": float(confidence),
        "created_at": ts,
        "updated_at": ts,
    }


def metabary(
    cm1_id: Any,
    cm2_id: Any,
    level: int,
    vector: np.ndarray,
    q_mb_raw: float,
    accumulated_weight: float,
    *,
    role: str | None,
    role_probability: float = 0.0,
    role_source: str = "unclassified",
) -> dict[str, Any]:
    """L≤13 MetaBary with a role on the bridge relation."""
    ts = _now()
    return {
        "doc_type": "baryedge",
        "cm1_id": cm1_id,
        "cm2_id": cm2_id,
        "level": level,
        "vector": _vec(vector),
        "parent_edge_id": None,
        "connection_strength": float(q_mb_raw),
        "accumulated_weight": float(accumulated_weight),
        "role": role,
        "role_probability": float(role_probability),
        "role_source": role_source,
        "created_at": ts,
        "updated_at": ts,
    }
