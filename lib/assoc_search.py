"""Associative search over BaryGraph — "search through the graph, answer with
destinations".

Runs the full propagation / competition / convergence / abstraction pipeline
server-side and returns compact high-level associative coordinates with the
support paths that produced them. Pure search logic (no MCP dependency) so it
unit-tests without services.

Pipeline:

    cue -> seed resolution -> direct leaf/BE retrieval -> bounded multi-branch
    upward propagation (three channels: strong / divergent / convergent-low)
    -> high-level candidate accumulation -> path-grouping (convergence)
    -> relevance + convergence + novelty ranking -> top-k coordinates

Graph facts the engine relies on:

  * Upward motion is ``parent_edge_id`` (unique parent per doc):
        sense (L15) -> L15 BE (L15) -> L13 MB -> L12 MB -> L11 MB -> L10 MB
        word  (L14) -> L14 BE (L14) -> L12 MB -> L11 MB -> L10 MB
    Orphans (no parent) are dead ends that stop a branch.
  * A MetaBary's triad is ``cm1_id``/``cm2_id`` (+ the bridge, stored as
    ``bridge_id`` for SMBs or found by reverse ``parent_edge_id`` lookup).
  * Leaf words of any BE/MB come from ``cm_leaf_words`` (capped BFS).
  * The vector index filter fields are doc_type, level, edge_type, node_type
    — filters are $eq only, so seed resolution issues one $vectorSearch per
    (type, level) slice.
"""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from lib.bary_vec import cosine, normalize
from lib.db import cm_leaf_words, vector_search
from lib.vector import unpack_vec

_log = logging.getLogger(__name__)

_SEED_VEC_BUDGET = 40  # num_candidates floor for seed $vectorSearch calls
_PRIOR_SMB_SCAN = 100  # top-k candidates kept for the prior-SMB probe
_ALPHA_FLOOR = 0.0

# Leaf-level S = a·R + b·q + g·E; the non-q weight is split 60/40 between
# relevance and energy so the total is 1 for any q_weight_leaf.
_LEAF_RELEVANCE_FRAC = 0.6
_LEAF_ENERGY_FRAC = 0.4


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AssocConfig:
    """Tunable knobs for one associative-search run.

    The three top_k values mean different things (spec §1):
      seed_top_k    breadth at seed resolution
      bridge_top_k  intermediate paths allowed to continue (beam width)
      result_top_k  final high-level associative hits returned (compactness)
    """

    seed_top_k: int = 20
    bridge_top_k: int = 12
    result_top_k: int = 5
    max_hops: int = 4
    target_levels: tuple[int, ...] = (12, 11, 10)
    min_convergence: int = 1
    beam_decay: float = 0.75
    novelty_weight: float = 0.2
    convergence_weight: float = 0.4
    q_weight_leaf: float = 0.5
    q_weight_high: float = 0.05
    return_paths: bool = True
    include_dois: bool = False

    # Result-formatting caps (not scoring knobs).
    max_target_words: int = 40
    max_step_words: int = 12
    max_paths_per_result: int = 4
    branch_lambda: float = 0.15  # per-hop branch-query blending factor (§10)


_CONFIG_FIELDS = frozenset(AssocConfig.__dataclass_fields__)


def _config_from_kwargs(**kwargs: Any) -> AssocConfig:
    """Build an AssocConfig from caller kwargs, applying clamps + defaults.

    Unknown keys are ignored so callers can pass a superset safely.
    """
    cfg: dict[str, Any] = {k: v for k, v in kwargs.items() if k in _CONFIG_FIELDS and v is not None}
    levels = tuple(int(lv) for lv in cfg.get("target_levels", AssocConfig.target_levels))
    if not levels or min(levels) < 1 or max(levels) > 13:
        raise ValueError("target_levels must be non-empty integers in 1..13")
    cfg["target_levels"] = tuple(sorted(set(levels), reverse=True))

    cfg["seed_top_k"] = min(max(int(cfg.get("seed_top_k", 20)), 1), 200)
    cfg["bridge_top_k"] = min(max(int(cfg.get("bridge_top_k", 12)), 1), 100)
    cfg["result_top_k"] = min(max(int(cfg.get("result_top_k", 5)), 1), 30)
    cfg["max_hops"] = min(max(int(cfg.get("max_hops", 4)), 1), 8)
    cfg["min_convergence"] = min(max(int(cfg.get("min_convergence", 1)), 1), 20)
    for k in ("beam_decay", "novelty_weight", "convergence_weight",
              "q_weight_leaf", "q_weight_high", "branch_lambda"):
        cfg[k] = min(max(float(cfg.get(k, getattr(AssocConfig, k))), 0.0), 1.0)
    cfg["return_paths"] = bool(cfg.get("return_paths", True))
    cfg["include_dois"] = bool(cfg.get("include_dois", False))
    return AssocConfig(**{k: v for k, v in cfg.items() if k in _CONFIG_FIELDS})


def _clamp_sim(v: float) -> float:
    """Clamp cosine ([-1, 1]) into the [0, 1] scoring range."""
    if math.isnan(v):
        return 0.0
    return max(0.0, min(1.0, float(v)))


def _conn_strength(doc: dict[str, Any], fallback: float = 1.0) -> float:
    """Edge reliability to use as a transition factor during propagation."""
    v = doc.get("connection_strength")
    if v is None:
        v = doc.get("accumulated_weight")
    if v is None:
        return fallback
    return _clamp_sim(float(v))


# --------------------------------------------------------------------------
# Candidate / step state (spec §3)
# --------------------------------------------------------------------------


@dataclass
class Step:
    """One hop in a support path: the words visible at a level plus the edge q.

    ``words`` is materialized lazily (None until rendered) so propagation never
    pays the cm_leaf_words BFS cost for beam members that get pruned.
    """

    level: int
    q: float | None = None
    edge_type: str | None = None
    id: str | None = None
    words: list[str] | None = None


@dataclass
class Candidate:
    """One active branch of the beam (spec §3 candidate state)."""

    node_id: Any                      # ObjectId of the current BE/MB (or seed node)
    level: int
    vector: np.ndarray                # stored vector of the current node
    parent_id: Any | None             # parent_edge_id (None => dead end)
    conn_q: float                     # connection strength of this node (edge q)
    origin_word: str                  # root seed word (coarsens senses of one word)
    origin_id: Any                    # root seed doc id
    origin_lang: str                  # language group of the root seed
    first_baryedge_id: Any | None     # first BE on the path (None until hop 1 for node seeds)
    path_ids: list[Any]               # node ids seed..current (inclusive)
    path_steps: list[Step]            # one Step per hop (seed step first)
    local_score: float                # sim(branch query, current node vector)
    energy: float                     # surviving activation from the cue (E)
    branch_query: np.ndarray          # per-branch accumulated query state (§10)
    arrival_groups: int = 1           # distinct branch_keys reaching this node this hop
    novelty: float = 0.0              # 1 - max sim to seed neighborhood (§5 divergent channel)
    dead: bool = False

    # Convergence independence key (spec §7): paths that differ only by
    # different senses of the same word routed through the same first BE
    # collapse into one group; paths through genuinely different roots/BEs
    # stay separate.
    @property
    def branch_key(self) -> tuple[str, str, str]:
        return (self.origin_word, str(self.first_baryedge_id), self.origin_lang)


# A tiny process-local cache for per-step leaf words. ObjectIds are globally
# unique, so keying by their string form does not collide across DBs.
_WORD_CACHE: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_WORD_CACHE_MAX = 4096


def _as_oid(be_id: Any) -> Any:
    """Loose id coercion: strings may be hex ObjectIds (e.g. Step.id)."""
    if isinstance(be_id, str):
        try:
            from bson import ObjectId
            return ObjectId(be_id)
        except Exception:  # noqa: BLE001 — not a hex oid; leave as-is
            return be_id
    return be_id


def _step_words(coll: Any, be_id: Any, cap: int) -> list[str]:
    """Leaf words reachable from a BE/MB id, capped, with a bounded cache."""
    key = str(be_id)
    if key in _WORD_CACHE:
        words = _WORD_CACHE[key]
    else:
        words = tuple(sorted(cm_leaf_words(coll, _as_oid(be_id), max_words=cap)))
        _WORD_CACHE[key] = words
        _WORD_CACHE.move_to_end(key)
        while len(_WORD_CACHE) > _WORD_CACHE_MAX:
            _WORD_CACHE.popitem(last=False)
    return list(words[:cap])


# --------------------------------------------------------------------------
# Stage 1 — seed resolution
# --------------------------------------------------------------------------


def resolve_seeds(
    coll: Any,
    qv: np.ndarray,
    config: AssocConfig,
    query: str | None = None,
    seed_ids: list[Any] | None = None,
) -> list[Candidate]:
    """Seed resolution: nodes + leaf BEs via $vectorSearch, plus exact headwords.

    ``seed_ids`` bypasses vector search (deterministic tests / pipelines
    without mongot): candidates are built straight from those documents. The
    exact-headword pass still runs when ``query`` is given.

    Returns seeds sorted by local_score desc, capped at ``config.seed_top_k``.
    """
    out: list[Candidate] = []
    if seed_ids:
        docs = _fetch_seed_docs(coll, seed_ids)
        for d in docs:
            c = _seed_from_doc(coll, d, qv)
            if c is not None:
                out.append(c)
    else:
        budget_node = max(1, config.seed_top_k // 2)
        budget_be = config.seed_top_k - budget_node
        for d in _vector_seed_hits(coll, qv,
                                   budget_node, max(1, budget_be)):
            c = _seed_from_doc(coll, d, qv)
            if c is not None:
                out.append(c)

    if query:
        out.extend(_exact_word_matches(coll, qv, query, config.seed_top_k))

    # Dedupe by node_id (keep best local_score), trim to seed_top_k.
    seen: dict[Any, Candidate] = {}
    for c in out:
        prev = seen.get(c.node_id)
        if prev is None or c.local_score > prev.local_score:
            seen[c.node_id] = c
    seeds = sorted(seen.values(), key=lambda c: c.local_score, reverse=True)
    return seeds[: config.seed_top_k]


def _vector_seed_hits(
    coll: Any, qv: np.ndarray, budget_node: int, budget_be: int
) -> list[dict[str, Any]]:
    """One $vectorSearch per supported filter slice (see module docstring)."""
    hits: list[dict[str, Any]] = []
    for filt, limit in (
        ({"doc_type": "node"}, budget_node),
        ({"doc_type": "baryedge", "level": 14}, max(1, budget_be // 2)),
        ({"doc_type": "baryedge", "level": 15}, max(1, budget_be - budget_be // 2)),
    ):
        try:
            hits.extend(vector_search(
                coll, qv.tolist(), limit=limit,
                num_candidates=max(_SEED_VEC_BUDGET, limit * 10),
                filter=filt,
            ))
        except Exception as e:  # noqa: BLE001 — one slice failing shouldn't kill seeding
            _log.warning("assoc: vector_search slice %s failed: %s", filt, e)
    return hits


def _exact_word_matches(
    coll: Any, qv: np.ndarray, query: str, top_n: int
) -> list[Candidate]:
    """Exact headword matches for the query string and its alpha tokens."""
    tokens: list[str] = []
    if query.strip():
        tokens.append(query.strip().lower())
    tokens += [t for t in re.findall(r"[a-z\u00e0-\u024f'-]{3,}", query.lower())
               if t not in tokens]

    cands: list[Candidate] = []
    seen_ids: set[Any] = set()
    for tok in tokens:
        docs = list(coll.find(
            {"doc_type": "node", "node_type": "word", "properties.word": tok},
            {"_id": 1, "doc_type": 1, "node_type": 1, "level": 1, "vector": 1,
             "parent_edge_id": 1, "properties.word": 1, "properties.lang": 1},
        ))
        fresh = [d for d in docs if d["_id"] not in seen_ids]
        for d in docs:
            seen_ids.add(d["_id"])
        for d in fresh:
            v = unpack_vec(d["vector"]) if d.get("vector") else np.array([], dtype=np.float32)
            if v.size == 0:
                continue
            score = _clamp_sim(cosine(qv, v))
            if score < 0.01:
                continue
            cands.append(_node_candidate(d, score, qv))
        if len(cands) >= top_n:
            break
    return cands


def _fetch_seed_docs(coll: Any, seed_ids: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(0, len(seed_ids), 200):
        chunk = seed_ids[i : i + 200]
        out.extend(coll.find(
            {"_id": {"$in": chunk}},
            {"_id": 1, "doc_type": 1, "node_type": 1, "level": 1, "vector": 1,
             "parent_edge_id": 1, "connection_strength": 1, "accumulated_weight": 1,
             "edge_type": 1, "properties.word": 1, "properties.lang": 1},
        ))
    return out


def _seed_from_doc(coll: Any, d: dict[str, Any], qv: np.ndarray) -> Candidate | None:
    v = d.get("vector")
    if v is None:
        return None
    vec = unpack_vec(v)
    if vec.size == 0:
        return None
    score = _clamp_sim(float(d.get("_score", cosine(qv, vec))))
    if d.get("doc_type") == "node":
        return _node_candidate(d, score, qv)
    # baryedge seed (L15/L14 BE): its identity is the words it pairs.
    conn = _conn_strength(d)
    label = " / ".join(_step_words(coll, d["_id"], 4)) or str(d["_id"])
    return Candidate(
        node_id=d["_id"],
        level=int(d.get("level", 0)),
        vector=vec,
        parent_id=d.get("parent_edge_id"),
        conn_q=conn,
        origin_word=label,
        origin_id=d["_id"],
        origin_lang="",
        first_baryedge_id=d["_id"],
        path_ids=[d["_id"]],
        path_steps=[Step(level=int(d.get("level", 0)), q=conn,
                         edge_type=d.get("edge_type"), id=str(d["_id"]))],
        local_score=score,
        energy=1.0,
        branch_query=qv.copy(),
    )


def _node_candidate(
    d: dict[str, Any], score: float, qv: np.ndarray
) -> Candidate:
    vec = unpack_vec(d["vector"])
    props = d.get("properties") or {}
    word = props.get("word") or ""
    level = int(d.get("level", 0)) or (15 if d.get("node_type") == "sense" else 14)
    return Candidate(
        node_id=d["_id"],
        level=level,
        vector=vec,
        parent_id=d.get("parent_edge_id"),
        conn_q=score,
        origin_word=word,
        origin_id=d["_id"],
        origin_lang=props.get("lang") or "en",
        first_baryedge_id=None,
        path_ids=[d["_id"]],
        path_steps=[Step(level=level, words=[word] if word else None)],
        local_score=score,
        energy=1.0,
        branch_query=qv.copy(),
    )


# --------------------------------------------------------------------------
# Stage 2 — upward propagation (one hop)
# --------------------------------------------------------------------------


def expand_upward(
    coll: Any,
    beam: list[Candidate],
    config: AssocConfig,
    direct_nbrs: list[np.ndarray],
    qv: np.ndarray,
) -> list[Candidate]:
    """Move each live branch one hop up ``parent_edge_id``.

    Creates the expanded candidate (new node, extended path, decayed energy)
    and computes per-branch novelty against the direct neighborhood D(q).
    """
    live = [c for c in beam if not c.dead]
    if not live:
        return []
    parent_ids = [c.parent_id for c in live if c.parent_id]
    if not parent_ids:
        for c in live:
            c.dead = True
        return []

    parents = {
        d["_id"]: d
        for d in coll.find(
            {"_id": {"$in": parent_ids}},
            {"_id": 1, "doc_type": 1, "node_type": 1, "level": 1, "vector": 1,
             "connection_strength": 1, "accumulated_weight": 1, "parent_edge_id": 1,
             "edge_type": 1},
        )
    }

    expanded: list[Candidate] = []
    for c in live:
        pd = parents.get(c.parent_id)
        if pd is None or not pd.get("vector"):
            c.dead = True
            continue
        vec = unpack_vec(pd["vector"])
        conn = _conn_strength(pd)
        level = int(pd.get("level", 0))
        bq = normalize(c.branch_query + config.branch_lambda * vec)
        local = _clamp_sim(cosine(bq, vec))
        energy = c.energy * conn * config.beam_decay
        novelty = _clamp_sim(
            1.0 - max(cosine(vec, n) for n in direct_nbrs)
        ) if direct_nbrs else 0.0
        fb = c.first_baryedge_id if c.first_baryedge_id is not None else pd["_id"]
        expanded.append(Candidate(
            node_id=pd["_id"],
            level=level,
            vector=vec,
            parent_id=pd.get("parent_edge_id"),
            conn_q=conn,
            origin_word=c.origin_word,
            origin_id=c.origin_id,
            origin_lang=c.origin_lang,
            first_baryedge_id=fb,
            path_ids=c.path_ids + [pd["_id"]],
            path_steps=c.path_steps + [Step(level=level, q=conn,
                                            edge_type=pd.get("edge_type"),
                                            id=str(pd["_id"]))],
            local_score=local,
            energy=energy,
            branch_query=bq,
            novelty=novelty,
        ))
    return expanded


# --------------------------------------------------------------------------
# Stage 3 — beam retention (three channels, spec §5)
# --------------------------------------------------------------------------


def _leaf_score(c: Candidate, config: AssocConfig) -> float:
    """Leaf-level activation: S_leaf = a·R + b·q + g·E (spec §6)."""
    b = config.q_weight_leaf
    a = _LEAF_RELEVANCE_FRAC * (1.0 - b)
    g = _LEAF_ENERGY_FRAC * (1.0 - b)
    return a * c.local_score + b * c.conn_q + g * c.energy


def _farness(c: Candidate, chosen: list[Candidate]) -> float:
    """Distance to the nearest already-chosen vector, in [0, 1]."""
    if not chosen:
        return 1.0
    return min(1.0 - _clamp_sim(cosine(c.vector, o.vector)) for o in chosen)


def _beam_select(expanded: list[Candidate], config: AssocConfig) -> list[Candidate]:
    """Retain up to bridge_top_k branches with three channels + diversity fill.

    Channel A — strong path:     high local·energy (reliable structure).
    Channel B — divergent:       high novelty (away from the obvious
                                 neighborhood; cross-domain bridges).
    Channel C — convergent-low:  high arrival_groups even at low energy (weak
                                 individual paths, several independent routes).
    """
    k = min(config.bridge_top_k, len(expanded))
    if k <= 0:
        return []

    selected: list[Candidate] = []
    selected_ids: set[Any] = set()

    budget_a = max(1, k * 2 // 5)
    budget_b = max(1, (k - budget_a) * 2 // 3)
    budget_c = max(0, k - budget_a - budget_b)
    counts = {"a": 0, "b": 0, "c": 0}

    def _add(c: Candidate, channel: str) -> None:
        if counts[channel] >= {"a": budget_a, "b": budget_b, "c": budget_c}[channel]:
            return
        if c.node_id in selected_ids:
            return
        counts[channel] += 1
        selected_ids.add(c.node_id)
        selected.append(c)

    # Channel A — strong path (budget a), Channel B — divergent (budget b).
    # Duplicate candidates (two branches expanding into the same parent) must
    # not let one channel overshoot its share and starve the others.
    for c in sorted(expanded, key=lambda c: _leaf_score(c, config),
                    reverse=True)[: budget_a * 3]:
        _add(c, "a")
    for c in sorted(expanded, key=lambda c: c.novelty, reverse=True)[: budget_b * 3]:
        _add(c, "b")
    for c in sorted(expanded, key=lambda c: (c.arrival_groups, -c.energy),
                    reverse=True)[: k * 2]:
        _add(c, "c")

    # Diversity fill: greedily add the farthest-from-selected remaining branch,
    # promoted by leaf score so "strong path" intent still biases the fill.
    remaining = [c for c in expanded if c.node_id not in selected_ids]
    while remaining and len(selected) < k:
        best = max(remaining, key=lambda c: 0.5 * _leaf_score(c, config)
                   + 0.5 * _farness(c, selected))
        selected_ids.add(best.node_id)
        selected.append(best)
        remaining = [c for c in remaining if c.node_id not in selected_ids]
    return selected


# --------------------------------------------------------------------------
# Stage 4 — convergence + ranking (spec §6, §7)
# --------------------------------------------------------------------------


def _rank_targets(
    qv: np.ndarray,
    seeds: list[Candidate],
    support_map: dict[Any, dict[tuple[str, str, str], Candidate]],
    config: AssocConfig,
) -> list[dict[str, Any]]:
    """High-level scoring: S = a·R + b·C + g·N + d·E, filtered by min_convergence."""
    alpha = max(_ALPHA_FLOOR, 1.0 - config.convergence_weight
                - config.novelty_weight - config.q_weight_high)
    seed_vecs = [s.vector for s in seeds]
    scored: list[dict[str, Any]] = []

    for node_id, bucket in support_map.items():
        paths = sorted(bucket.values(), key=lambda c: c.energy, reverse=True)
        groups = len(paths)
        if groups < config.min_convergence:
            continue
        weights = [float(p.energy) for p in paths]
        convergence = 1.0 - math.prod(1.0 - w for w in weights)
        best = paths[0]
        relevance = _clamp_sim(cosine(qv, best.vector))
        novelty = _clamp_sim(
            1.0 - max(cosine(best.vector, n) for n in seed_vecs)
        ) if seed_vecs else 0.0
        energy = float(best.energy)
        score = (alpha * relevance + config.convergence_weight * convergence
                 + config.novelty_weight * novelty + config.q_weight_high * energy)
        scored.append({
            "node_id": node_id,
            "target": best,
            "paths": paths,
            "score": score,
            "relevance": relevance,
            "convergence": convergence,
            "novelty": novelty,
            "energy": energy,
            "support_paths": groups,
        })
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored


# --------------------------------------------------------------------------
# Stage 5 — rendering (spec §8, §11)
# --------------------------------------------------------------------------


def _materialize_steps(coll: Any, steps: list[Step], cap: int) -> list[Step]:
    """Fill in lazy Step.words (cached leaf words per level)."""
    for s in steps:
        if s.words is None:
            if s.id is not None:
                s.words = _step_words(coll, s.id, cap)
            else:
                s.words = []
    return steps


def _triad_branches(coll: Any, mb_id: Any) -> dict[str, list[str]]:
    """child1 / child2 / bridge word sets for an MB, each capped."""
    doc = coll.find_one({"_id": mb_id}, {"cm1_id": 1, "cm2_id": 1, "bridge_id": 1})
    if not doc:
        return {"child1": [], "child2": [], "bridge": []}
    cm1, cm2 = doc.get("cm1_id"), doc.get("cm2_id")
    bridge = doc.get("bridge_id")
    if bridge is None:
        bdoc = coll.find_one(
            {"parent_edge_id": mb_id, "_id": {"$nin": [cm1, cm2]}}, {"_id": 1}
        )
        bridge = bdoc["_id"] if bdoc else None
    out: dict[str, list[str]] = {}
    for label, oid in (("child1", cm1), ("child2", cm2), ("bridge", bridge)):
        if oid:
            out[label] = _step_words(coll, oid, 20)
        else:
            out[label] = []
    return out


def _why_tree(coll: Any, steps_: list[Step], target_words: set[str]) -> dict[str, Any]:
    """Compressed explanation of the route (§11), not the activated area."""
    steps = _materialize_steps(coll, steps_, 12)
    seed_words: set[str] = set()
    for s in steps[:-1]:
        for w in s.words or []:
            seed_words.add(w)
    mb_steps = [s for s in steps if s.level <= 13]
    tr_step = next((s for s in steps if s.level in (14, 15) and s.q is not None), None)
    return {
        "trigger": (tr_step.words or []) if tr_step else (steps[0].words or []),
        "intermediary": [
            {"level": s.level, "words": s.words or []} for s in mb_steps[:-1]
        ],
        "support": sorted(w for w in target_words if w in seed_words),
        "alternate": sorted(w for w in target_words if w not in seed_words),
    }


def _path_word_chain(coll: Any, cand: Candidate, cap: int = 8) -> list[str]:
    """Flatten a candidate's path steps into a deduped word chain."""
    out: list[str] = []
    for s in _materialize_steps(coll, cand.path_steps, cap):
        for w in s.words or []:
            if w not in out:
                out.append(w)
    return out


def render_result(
    coll: Any,
    record: dict[str, Any],
    config: AssocConfig,
    rank: int,
    bridge_coll: Any = None,
) -> dict[str, Any]:
    """Render one ranked target into the compact result contract (§8)."""
    target = record["target"]
    node_id = target.node_id
    words = _step_words(coll, node_id, config.max_target_words)

    result: dict[str, Any] = {
        "rank": rank,
        "id": str(node_id),
        "level": target.level,
        "words": words[: config.max_target_words],
        "words_truncated": len(words) >= config.max_target_words,
        "score": round(float(record["score"]), 3),
        "scores": {
            "relevance": round(float(record["relevance"]), 3),
            "convergence": round(float(record["convergence"]), 3),
            "novelty": round(float(record["novelty"]), 3),
            "energy": round(float(record["energy"]), 3),
        },
        "convergence": round(float(record["convergence"]), 3),
        "novelty": round(float(record["novelty"]), 3),
        "support_paths": record["support_paths"],
    }

    steps = _materialize_steps(coll, target.path_steps, config.max_step_words)
    be_steps = [s for s in steps if s.level in (14, 15) and s.q is not None]
    if be_steps:
        tr = be_steps[0]
        result["trigger"] = {
            "level": tr.level, "words": tr.words or [],
            "q": round(float(tr.q or 0.0), 3),
            "edge_type": tr.edge_type, "id": tr.id,
        }

    result["why"] = _why_tree(coll, target.path_steps, set(words))

    if config.return_paths:
        chains: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for p in record["paths"][: config.max_paths_per_result]:
            chain = _path_word_chain(coll, p)
            key = tuple(chain)
            if key in seen or not chain:
                continue
            seen.add(key)
            chains.append(chain)
        result["paths"] = chains
        result["path_steps"] = [
            {
                "level": s.level,
                "words": (s.words or [])[: config.max_step_words],
                "q": s.q,
                "edge_type": s.edge_type,
                "id": s.id,
            }
            for s in steps
        ]

    if config.include_dois and bridge_coll is not None:
        from lib import doi_bridge

        result["dois"] = sorted(doi_bridge.dois_for_node(bridge_coll, node_id))
    return result


def _summary_text(query: str, payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for i, r in enumerate(payload.get("results", [])[:3]):
        words = " / ".join(r["words"][:8])
        lines.append(f"ASSOCIATIVE COORDINATE ({i + 1}) — L{r['level']}: {words}")
        lines.append("Reached through:")
        chains = r.get("paths") or []
        if chains:
            lines.append("  " + " -> ".join(chains[0]))
        else:
            segs = [
                " -> ".join(s["words"][:4]) or f"L{s['level']}"
                for s in r.get("path_steps", [])
            ]
            lines.append("  " + " -> ".join(segs))
        lines.append(f"support_paths={r.get('support_paths')} "
                     f"convergence={r.get('convergence')} novelty={r.get('novelty')}")
    lines.append("Interpretation status:")
    lines.append("candidate coordinate; not a verified claim")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Stage 6 — core loop shared by one-shot + progressive discover
# --------------------------------------------------------------------------


def _core_search(
    coll: Any,
    embedder: Any,
    query: str,
    config: AssocConfig,
    *,
    bridge_coll: Any = None,
    seed_ids: list[Any] | None = None,
    qv: np.ndarray | None = None,
) -> dict[str, Any]:
    """Seed -> propagate -> accumulate -> rank.

    Returns either a status payload (error / no_seed / no_target) or a dict
    with the ranked records under ``_records`` plus the payload fields.
    """
    try:
        qv_ = (qv if qv is not None
               else np.asarray(embedder.embed([query])[0], dtype=np.float32))
    except Exception as e:  # noqa: BLE001
        _log.exception("assoc: embedding failed for query=%r", query)
        return {"status": "error", "query": query,
                "message": f"Embedding failed: {type(e).__name__}: {e}"}

    seeds = resolve_seeds(coll, qv_, config, query=query, seed_ids=seed_ids)
    if not seeds:
        return {
            "status": "no_seed", "query": query, "seed_count": 0,
            "target_levels": list(config.target_levels),
            "message": "No exact graph seed was found.",
        }

    direct_nbrs = [s.vector for s in seeds]
    beam = seeds
    support_map: dict[Any, dict[tuple[str, str, str], Candidate]] = {}
    highest = max((s.level for s in seeds), default=0)  # numeric level = abstractness
    hops_done = 0

    for hop in range(1, config.max_hops + 1):
        live = [c for c in beam if not c.dead]
        if not live:
            break
        expanded = expand_upward(coll, live, config, direct_nbrs, qv_)
        if not expanded:
            break
        hops_done = hop

        arrivals: dict[Any, set[tuple[str, str, str]]] = {}
        for c in expanded:
            arrivals.setdefault(c.node_id, set()).add(c.branch_key)
        for c in expanded:
            c.arrival_groups = len(arrivals[c.node_id])
            highest = min(highest, c.level)
            if c.level in config.target_levels:
                bucket = support_map.setdefault(c.node_id, {})
                prev = bucket.get(c.branch_key)
                if prev is None or c.energy > prev.energy:
                    bucket[c.branch_key] = c

        beam = _beam_select(expanded, config)

    ranked = _rank_targets(qv_, seeds, support_map, config)
    if not ranked:
        return {
            "status": "no_target", "query": query, "seed_count": len(seeds),
            "highest_reached_level": highest,
            "target_levels": list(config.target_levels),
            "message": "Seed structures found, but no candidate reached the "
                       "requested target levels.",
        }

    return {
        "status": "ok", "query": query, "seed_count": len(seeds),
        "highest_reached_level": highest, "hops": hops_done,
        "max_hops": config.max_hops, "target_levels": list(config.target_levels),
        "seeds": seeds, "support_map": support_map, "ranked": ranked,
        "_qv": qv_,
    }


def run_search(
    coll: Any,
    embedder: Any,
    query: str,
    config: AssocConfig,
    *,
    bridge_coll: Any = None,
    seed_ids: list[Any] | None = None,
    qv: np.ndarray | None = None,
) -> dict[str, Any]:
    """One-shot associative search. Returns the JSON-ready payload (§13)."""
    core = _core_search(coll, embedder, query, config, bridge_coll=bridge_coll,
                        seed_ids=seed_ids, qv=qv)
    if core["status"] != "ok":
        core.pop("_qv", None)
        return core

    results = [
        render_result(coll, rec, config, i, bridge_coll)
        for i, rec in enumerate(core["ranked"][: config.result_top_k], start=1)
    ]
    payload: dict[str, Any] = {
        "status": "ok", "query": query, "seed_count": core["seed_count"],
        "hops": core["hops"], "max_hops": core["max_hops"],
        "highest_reached_level": core["highest_reached_level"],
        "target_levels": core["target_levels"], "results": results,
    }
    payload["summary"] = _summary_text(query, payload)
    return payload


# --------------------------------------------------------------------------
# Progressive sessions (spec §9): discover -> expand -> compare -> propose
# --------------------------------------------------------------------------


@dataclass
class Session:
    session_id: str
    query: str
    config: AssocConfig
    qv: np.ndarray
    created_at: datetime
    updated_at: datetime
    seeds: list[Candidate] = field(default_factory=list)
    ranked: list[dict[str, Any]] = field(default_factory=list)
    rendered: list[dict[str, Any]] = field(default_factory=list)
    support_map: dict[Any, dict[tuple[str, str, str], Candidate]] = field(
        default_factory=dict
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


_SESSIONS: dict[str, Session] = {}
_MAX_SESSIONS = 32


def _session_get(session_id: str) -> Session | None:
    return _SESSIONS.get(session_id)


def _session_put(s: Session) -> None:
    s.touch()
    _SESSIONS[s.session_id] = s
    if len(_SESSIONS) > _MAX_SESSIONS:
        oldest = sorted(_SESSIONS.values(), key=lambda x: x.updated_at)
        for old in oldest[: len(_SESSIONS) - _MAX_SESSIONS]:
            _SESSIONS.pop(old.session_id, None)


def _coerce_oid(s: str) -> Any:
    from bson import ObjectId

    try:
        return ObjectId(s)
    except Exception:  # noqa: BLE001
        return s


def _stage_discover(
    coll, embedder, session_id: str, query: str, config: AssocConfig,
    *, bridge_coll, qv=None,
) -> dict[str, Any]:
    """Run the full search and store the activated area for later stages."""
    core = _core_search(coll, embedder, query, config, bridge_coll=bridge_coll, qv=qv)
    if core["status"] != "ok":
        core.pop("_qv", None)
        return core

    sess = Session(
        session_id=session_id, query=query, config=config, qv=core["_qv"],
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        seeds=core["seeds"], ranked=core["ranked"], support_map=core["support_map"],
    )
    _session_put(sess)

    payload: dict[str, Any] = {
        "status": "ok", "session_id": session_id, "stage": "discover",
        "query": query, "seed_count": core["seed_count"], "hops": core["hops"],
        "highest_reached_level": core["highest_reached_level"],
        "target_levels": core["target_levels"],
        "session": {
            "session_id": session_id, "query": query,
            "created_at": sess.created_at.isoformat(),
            "updated_at": sess.updated_at.isoformat(),
            "seed_count": len(sess.seeds),
            "target_levels": list(config.target_levels),
            "max_hops": config.max_hops,
            "bridge_top_k": config.bridge_top_k,
        },
        # Every ranked candidate, compact — the caller picks which to expand.
        "candidates": [
            {
                "rank": i, "id": str(r["target"].node_id),
                "level": r["target"].level,
                "score": round(float(r["score"]), 3),
                "convergence": round(float(r["convergence"]), 3),
                "novelty": round(float(r["novelty"]), 3),
                "support_paths": r["support_paths"],
            }
            for i, r in enumerate(core["ranked"], start=1)
        ],
    }
    payload["summary"] = _summary_text(query, payload)
    return payload


def _stage_expand(coll, sess: Session, selected_ids: list[str],
                  config: AssocConfig, bridge_coll) -> dict[str, Any]:
    by_id = {str(r["target"].node_id): r for r in sess.ranked}
    expanded: list[dict[str, Any]] = []
    for sid in selected_ids:
        rec = by_id.get(sid)
        if rec is None:
            expanded.append({"selected_id": sid,
                             "error": "not a discovered target"})
            continue
        t = rec["target"]
        leaf = _step_words(coll, t.node_id, config.max_target_words)
        triad = _triad_branches(coll, t.node_id)
        siblings = [
            {
                "id": str(o["target"].node_id), "level": o["target"].level,
                "cosine": round(_clamp_sim(cosine(t.vector, o["target"].vector)), 3),
                "words": _step_words(coll, o["target"].node_id, 8),
            }
            for o in sess.ranked
            if o is not rec and o["target"].level == t.level
        ][:5]
        seed_step = t.path_steps[0].words or []
        expanded.append({
            "id": sid, "level": t.level,
            "words": leaf[: config.max_target_words],
            "words_truncated": len(leaf) >= config.max_target_words,
            "triad": triad,
            "score": round(float(rec["score"]), 3),
            "convergence": round(float(rec["convergence"]), 3),
            "support_paths": rec["support_paths"],
            "strongest_leaf_support": seed_step,
            "competing_siblings": siblings,
            "alternate_cross_links": (triad.get("bridge") or triad.get("child2") or [])[:10],
        })
    sess.touch()
    return {"status": "ok", "session_id": sess.session_id, "stage": "expand",
            "expanded": expanded}


def _stage_compare(coll, sess: Session, selected_ids: list[str],
                   config: AssocConfig) -> dict[str, Any]:
    by_id = {str(r["target"].node_id): r for r in sess.ranked}
    picks = [by_id[i] for i in selected_ids if i in by_id]
    if len(picks) < 2:
        return {"status": "error", "session_id": sess.session_id, "stage": "compare",
                "message": "compare needs at least 2 discovered target ids."}

    pairs = []
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            a, b = picks[i], picks[j]
            ta, tb = a["target"], b["target"]
            wa = set(_step_words(coll, ta.node_id, 30))
            wb = set(_step_words(coll, tb.node_id, 30))
            inter = wa & wb
            union = wa | wb
            trig_a = a["paths"][0].path_steps[1].id if len(a["paths"][0].path_steps) > 1 else None
            trig_b = b["paths"][0].path_steps[1].id if len(b["paths"][0].path_steps) > 1 else None
            pairs.append({
                "a_id": str(ta.node_id), "b_id": str(tb.node_id),
                "a_level": ta.level, "b_level": tb.level,
                "cosine": round(_clamp_sim(cosine(ta.vector, tb.vector)), 3),
                "word_overlap": round(len(inter) / len(union), 3) if union else 0.0,
                "overlapping_words": sorted(inter)[:10],
                "shared_origins": sorted({
                    p.origin_word for p in a["paths"]} & {p.origin_word for p in b["paths"]})[:10],
                "shared_trigger": trig_a is not None and trig_a == trig_b,
            })
    sess.touch()
    return {"status": "ok", "session_id": sess.session_id, "stage": "compare",
            "pairs": pairs}


def _stage_propose(coll, sess: Session, selected_ids: list[str],
                   config: AssocConfig, bridge_coll) -> dict[str, Any]:
    by_id = {str(r["target"].node_id): r for r in sess.ranked}
    picks = [by_id[i] for i in selected_ids if i in by_id]
    if len(picks) < 2:
        return {"status": "error", "session_id": sess.session_id, "stage": "propose",
                "message": "propose needs at least 2 discovered target ids."}
    a, b = picks[0], picks[1]
    ta, tb = a["target"], b["target"]
    if ta.level != tb.level:
        return {"status": "error", "session_id": sess.session_id, "stage": "propose",
                "message": f"selected targets must share a level "
                           f"(got L{ta.level} vs L{tb.level})."}
    child_level = ta.level
    mb_level = child_level - 2
    if mb_level < 1:
        return {"status": "error", "session_id": sess.session_id, "stage": "propose",
                "message": f"child_level {child_level} would produce an SMB at "
                           f"level {mb_level} (min 1)."}

    centroid = normalize(ta.vector + tb.vector)
    child_cosine = round(_clamp_sim(cosine(ta.vector, tb.vector)), 4)
    bridge_level = child_level - 1

    bridges: list[dict[str, Any]] = []
    try:
        for h in vector_search(
            coll, centroid.tolist(), limit=3,
            num_candidates=max(_SEED_VEC_BUDGET, 50),
            filter={"doc_type": "baryedge", "level": bridge_level},
        ):
            bridges.append({
                "id": str(h["_id"]), "level": h.get("level"),
                "cosine": round(float(h.get("_score", 0)), 3),
                "connection_strength": h.get("connection_strength"),
                "words": _step_words(coll, h["_id"], 10),
            })
    except Exception as e:  # noqa: BLE001
        _log.warning("assoc propose: bridge search failed: %s", e)

    prior_smbs: list[dict[str, Any]] = []
    try:
        wa = set(_step_words(coll, ta.node_id, 20))
        wb = set(_step_words(coll, tb.node_id, 20))
        region = wa | wb
        for match in _prior_smbs_near(coll, centroid, region):
            prior_smbs.append(match)
    except Exception as e:  # noqa: BLE001
        _log.warning("assoc propose: prior SMB scan failed: %s", e)

    dois: list[str] = []
    if bridge_coll is not None:
        from lib import doi_bridge

        dois = sorted(doi_bridge.dois_for_nodes(
            bridge_coll, [ta.node_id, tb.node_id]
        ).get(ta.node_id, []))
        for br in bridges:
            try:
                dois.extend(doi_bridge.dois_for_node(
                    bridge_coll, _coerce_oid(br["id"])
                ))
            except Exception:  # noqa: BLE001
                pass
        dois = sorted(set(dois))

    sess.touch()
    return {
        "status": "ok", "session_id": sess.session_id, "stage": "propose",
        "packet": {
            "cm1_id": str(ta.node_id), "cm2_id": str(tb.node_id),
            "cm1_words": _step_words(coll, ta.node_id, 20),
            "cm2_words": _step_words(coll, tb.node_id, 20),
            "child_level": child_level, "smb_level": mb_level,
            "expected_child_cosine": child_cosine,
            "bridge_candidates": bridges,
            "convergence": {
                "cm1": round(float(a["convergence"]), 3),
                "cm2": round(float(b["convergence"]), 3),
            },
            "academic_provenance": dois,
            "prior_smbs": prior_smbs,
            "note": "This packet is a proposal — call create_structure_meta_bary "
                    "with (cm1_id, cm2_id, bridge_id) to build it.",
        },
    }


def _prior_smbs_near(
    coll: Any, centroid: np.ndarray, region_words: set[str]
) -> list[dict[str, Any]]:
    """Structural-SMB docs near the centroid, probed via a bounded $vectorSearch.

    The vector index has no ``source`` filter path and ``source`` has no
    collection index, so a document query like
    ``{doc_type: baryedge, source: structural, level: {$lte: 13}}`` is a
    non-selective scan (Mongo discards every pipeline MetaBary until the
    whole range is exhausted — the 30s+ ``propose`` hang). Instead, keep the
    HNSW candidates that are level-filtered through the index's ``level``
    path, then select ``source='structural'`` client-side.
    """
    matches: list[dict[str, Any]] = []
    try:
        hits = vector_search(
            coll, centroid.tolist(), limit=_PRIOR_SMB_SCAN,
            num_candidates=_SEED_VEC_BUDGET,
            filter={"doc_type": "baryedge", "level": {"$in": [10, 11, 12, 13]}},
        )
    except Exception as e:  # noqa: BLE001
        _log.warning("assoc propose: prior SMB scan failed: %s", e)
        return []
    for d in hits:
        if d.get("source") != "structural":
            continue
        v = unpack_vec(d["vector"]) if d.get("vector") else np.array([], dtype=np.float32)
        if v.size == 0:
            continue
        c = _clamp_sim(cosine(centroid, v))
        if c < 0.5:
            continue
        overlap = sorted(set(_step_words(coll, d["_id"], 20)) & region_words)[:10]
        matches.append({
            "id": str(d["_id"]), "level": d.get("level"),
            "cosine": round(c, 3), "overlapping_words": overlap,
        })
    matches.sort(key=lambda m: m["cosine"], reverse=True)
    return matches[:5]


def progressive(
    coll: Any,
    embedder: Any,
    session_id: str,
    stage: str,
    query: str = "",
    selected_ids: list[str] | None = None,
    config: AssocConfig | None = None,
    *,
    bridge_coll: Any = None,
    qv: np.ndarray | None = None,
) -> dict[str, Any]:
    """Session-based progressive associative search (spec §9)."""
    config = config or AssocConfig()

    if stage == "discover":
        if not query:
            return {"status": "error", "stage": "discover",
                    "message": "discover requires a query."}
        return _stage_discover(coll, embedder, session_id, query, config,
                               bridge_coll=bridge_coll, qv=qv)

    sess = _session_get(session_id)
    if sess is None:
        return {"status": "error", "session_id": session_id, "stage": stage,
                "message": f"No session '{session_id}' — call stage=discover first."}
    if not selected_ids:
        return {"status": "error", "session_id": session_id, "stage": stage,
                "message": f"{stage} requires selected_ids."}

    if stage == "expand":
        out = _stage_expand(coll, sess, selected_ids, config, bridge_coll)
    elif stage == "compare":
        out = _stage_compare(coll, sess, selected_ids, config)
    elif stage == "propose":
        out = _stage_propose(coll, sess, selected_ids, config, bridge_coll)
    else:
        return {"status": "error", "session_id": session_id, "stage": stage,
                "message": f"unknown stage '{stage}' (want discover|expand|compare|propose)."}
    out.setdefault("query", sess.query)
    out.setdefault("target_levels", list(config.target_levels))
    return out
