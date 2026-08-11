"""MongoDB document constructors for ``node`` and ``baryedge`` doc_types.

Centralizing these keeps every stage producing the same v0.5 schema and
makes the unique-parent / level invariants easy to audit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from lib.schema import ParsedSense, ParsedWord


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _vec(v: np.ndarray | list[float] | None) -> list[float] | None:
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32).tolist()


def sense_node(s: ParsedSense, vector: np.ndarray | list[float]) -> dict[str, Any]:
    ts = _now()
    return {
        "doc_type": "node",
        "node_type": "sense",
        "level": 15,
        "label": f"{s.word} ({s.pos}) [{s.sense_idx}]",
        "vector": _vec(vector),
        "surface": 1,
        "rotation": 0.0,
        "parent_edge_id": None,
        "properties": {
            "word": s.word,
            "pos": s.pos,
            "sense_id": s.sense_id,
            "sense_idx": s.sense_idx,
            "gloss": s.gloss,
            "examples": s.examples,
            "tags": s.tags,
            "topics": s.topics,
            "wikidata": s.wikidata,
            "doi": s.doi,
        },
        "created_at": ts,
        "updated_at": ts,
    }


def word_node(w: ParsedWord) -> dict[str, Any]:
    """L14 word node with placeholder vector (filled by s05_word_vectors)."""
    ts = _now()
    return {
        "doc_type": "node",
        "node_type": "word",
        "level": 14,
        "label": f"{w.word} ({w.pos})",
        "vector": None,
        "surface": max(1, len(w.sense_ids)),
        "rotation": 0.0,
        "parent_edge_id": None,
        "properties": {
            "word": w.word,
            "pos": w.pos,
            "etymology": w.etymology,
            "forms": w.forms,
            "ipa": w.ipa,
            "sense_ids": w.sense_ids,
            "relations": [{"kind": r.kind, "word": r.word} for r in w.relations],
        },
        "created_at": ts,
        "updated_at": ts,
    }


def baryedge(
    cm1_id: Any,
    cm2_id: Any,
    level: int,
    vector: np.ndarray,
    q: float,
    *,
    accumulated_weight: float | None = None,
    edge_type: str | None = None,
    type_vector: np.ndarray | None = None,
    source: str = "inferred",
    confidence: float = 1.0,
) -> dict[str, Any]:
    """L14/L15 BaryEdge. At L14/L15 accumulated_weight = q (base case)."""
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
) -> dict[str, Any]:
    """L≤13 MetaBary. connection_strength = q_mb_raw ∈ [0,1]; accumulated_weight may exceed 1."""
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
        "created_at": ts,
        "updated_at": ts,
    }
