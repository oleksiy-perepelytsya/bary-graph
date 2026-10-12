"""Shared pairing / BaryEdge-formation logic for the tensegrity pipeline.

Used by t04 (L15 sentence edges), t06 (L14 title edges), and t07 (orphan
re-entry). Mirrors the PoC's ``lib.match`` + s04/s06 flow — ANN/brute-force
``top_k_pairs`` → ``greedy_unique_match`` → signed-role barycentric vector —
but additionally asks the tev1 decision model for the complement/contradict
role of every pair.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne

from lib.bary_vec import normalize  # noqa: F401
from lib.match import greedy_unique_match, top_k_pairs
from lib.vector import unpack_vec
from tensegrity_pipeline.docs import tensegrity_baryedge
from tensegrity_pipeline.tev1 import DecisionModel, classify_role
from tensegrity_pipeline.tg_vec import (
    COMPLEMENT,
    ROLE_SENTENCES,
    ROLES,
    compute_bary_vec,
)

_log = logging.getLogger(__name__)


def node_filter(level: int) -> dict:
    if level == 15:
        return {"doc_type": "node", "node_type": "sentence", "level": 15}
    return {"doc_type": "node", "node_type": "title", "level": 14}


def node_parent_filter(level: int) -> dict:
    q = node_filter(level)
    q["parent_edge_id"] = None
    return q


def load_level_nodes(
    coll, level: int, *, limit: int | None = None, parents_only: bool = False
) -> tuple[list, list[str], np.ndarray]:
    """Load (ids, texts, V) for one level's nodes that carry a vector.

    Rows with a missing vector (e.g. dead zero-sense nodes) are skipped so
    ``ids[i] ↔ V[i]`` stay aligned.
    """
    q = node_parent_filter(level) if parents_only else node_filter(level)
    ids: list = []
    texts: list[str] = []
    vecs: list[np.ndarray] = []
    for doc in coll.find(q, {"_id": 1, "vector": 1, "properties.text": 1}):
        v = doc.get("vector")
        if v is None:
            continue
        ids.append(doc["_id"])
        texts.append(doc.get("properties", {}).get("text", ""))
        vecs.append(unpack_vec(v))
        if limit is not None and len(ids) >= limit:
            break
    if not vecs:
        return ids, texts, np.zeros((0, 0), dtype=np.float32)
    return ids, texts, np.stack(vecs).astype(np.float32)


def role_type_vectors(embedder) -> dict[str, np.ndarray]:
    """``v_role`` per role — embed the two fixed role sentences once."""
    out: dict[str, np.ndarray] = {}
    for role in ROLES:
        out[role] = np.asarray(embedder.embed([ROLE_SENTENCES[role]])[0], dtype=np.float32)
    return out


def form_level_edges(
    coll,
    settings,
    embedder,
    model: DecisionModel,
    level: int,
    *,
    limit: int | None = None,
    parents_only: bool = False,
    dry_run: bool = False,
    log: logging.Logger | None = None,
) -> int:
    """Pair nodes at ``level`` by cosine and write role-tagged BaryEdges.

    ``parents_only`` restricts the pool to unparented nodes (t07's orphan pair
    phase). Returns the number of BaryEdges written (or that *would* be written
    in dry-run mode).
    """
    log = log or _log
    batch_n = settings.batch_size
    ids, texts, V = load_level_nodes(coll, level, limit=limit, parents_only=parents_only)
    n_nodes = len(ids)
    if n_nodes < 2:
        log.info("L%d: fewer than 2 nodes (%d) — nothing to pair", level, n_nodes)
        return 0

    min_cos = settings.q_min_l15
    log.info("L%d: matching %d nodes (q_min=%.2f)", level, n_nodes, min_cos)
    raw = top_k_pairs(V, min_score=min_cos)
    pairs = greedy_unique_match(raw, threshold=min_cos)
    if limit is not None and len(pairs) > limit:
        pairs = pairs[:limit]
    log.info("L%d: %d pairs selected", level, len(pairs))
    if not pairs:
        return 0

    role_vecs = role_type_vectors(embedder)

    docs: list[dict] = []
    pair_idx: list[tuple[int, int]] = []
    n_written = 0
    now = datetime.now(timezone.utc)

    def _flush() -> None:
        nonlocal docs, pair_idx, n_written
        if not docs:
            return
        if not dry_run:
            res = coll.insert_many(docs)
            ups: list[UpdateOne] = []
            for (i, j), eid in zip(pair_idx, res.inserted_ids, strict=True):
                for node_id in (ids[i], ids[j]):
                    ups.append(
                        UpdateOne(
                            {"_id": node_id},
                            {"$set": {"parent_edge_id": eid, "updated_at": now}},
                        )
                    )
            coll.bulk_write(ups, ordered=False)
        n_written += len(docs)
        docs = []
        pair_idx = []

    for i, j, q in pairs:
        cls = classify_role(model, f"{texts[i]}\n{texts[j]}")
        role = cls["role"]
        v_role = role_vecs.get(role or COMPLEMENT, role_vecs[COMPLEMENT])
        bv = compute_bary_vec(V[i], V[j], v_role, q, role)
        docs.append(
            tensegrity_baryedge(
                ids[i], ids[j], level, bv, q,
                role=role,
                role_probability=cls["role_probability"],
                role_source=cls["role_source"],
                accumulated_weight=q,
                type_vector=v_role,
                source="inferred",
                confidence=q,
            )
        )
        pair_idx.append((i, j))
        if len(docs) >= batch_n:
            _flush()
    _flush()
    log.info("L%d: wrote %d BaryEdges", level, n_written)
    return n_written


def text_resolver(coll):
    """Return ``resolve(id) -> str`` over node texts and (recursively) BEs.

    Node documents expose ``properties.text``; a BaryEdge's text is the
    concatenation of its two participants, which is what t07/t08 hand to tev1
    when classifying an orphan against a BE or a MetaBary bridge.
    """
    node_text: dict = {}
    for d in coll.find({"doc_type": "node"}, {"_id": 1, "properties.text": 1}):
        node_text[d["_id"]] = d.get("properties", {}).get("text", "")
    be_cm: dict = {}
    for d in coll.find({"doc_type": "baryedge"}, {"_id": 1, "cm1_id": 1, "cm2_id": 1}):
        be_cm[d["_id"]] = (d.get("cm1_id"), d.get("cm2_id"))

    memo: dict = {}

    def resolve(doc_id, depth: int = 0) -> str:
        if doc_id in memo:
            return memo[doc_id]
        if doc_id in node_text:
            memo[doc_id] = node_text[doc_id]
            return memo[doc_id]
        if doc_id in be_cm and depth < 4:
            a, b = be_cm[doc_id]
            txt = " ".join(t for t in (resolve(a, depth + 1), resolve(b, depth + 1)) if t)
            memo[doc_id] = txt
            return txt
        return ""

    return resolve


def load_unparented_bes(
    coll, level: int, *, exclude_structural: bool = True
) -> tuple[list, list[float], np.ndarray]:
    """Load (ids, accumulated_weights, V) for unparented BEs/MBs at ``level``.

    In-memory variant of s08's memmap loader — adequate for document-sized
    corpora (a whole document set is orders of magnitude smaller than the
    kaikki L15 pool). ``source='structural'`` SMBs are excluded so the pipeline
    build never absorbs hand-admitted triads.
    """
    q: dict = {
        "doc_type": "baryedge",
        "level": level,
        "parent_edge_id": None,
    }
    if exclude_structural:
        q["source"] = {"$ne": "structural"}
    ids: list = []
    weights: list[float] = []
    vecs: list[np.ndarray] = []
    for doc in coll.find(q, {"_id": 1, "vector": 1, "accumulated_weight": 1}):
        v = doc.get("vector")
        if v is None:
            continue
        ids.append(doc["_id"])
        w = doc.get("accumulated_weight")
        weights.append(float(w) if w is not None else 1.0)
        vecs.append(unpack_vec(v))
    if not vecs:
        return ids, weights, np.zeros((0, 0), dtype=np.float32)
    return ids, weights, np.stack(vecs).astype(np.float32)


def nearest_be(
    coll, level: int, query_vec: np.ndarray
) -> tuple[ObjectId | None, float]:
    """Nearest existing BaryEdge at ``level`` to ``query_vec`` (cosine).

    Streams level BEs and keeps the best cosine; adequate for document-sized
    corpora and avoids an extra ANN index build.
    """
    best_id: ObjectId | None = None
    best_cos = -2.0
    qv = normalize(np.asarray(query_vec, dtype=np.float32))
    for doc in coll.find(
        {"doc_type": "baryedge", "level": level}, {"_id": 1, "vector": 1, "cm1_id": 1, "cm2_id": 1}
    ):
        v = doc.get("vector")
        if v is None:
            continue
        cos = float(np.dot(qv, normalize(unpack_vec(v))))
        if cos > best_cos:
            best_cos = cos
            best_id = doc["_id"]
    return best_id, best_cos
