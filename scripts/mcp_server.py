"""BaryGraph MCP server — exposes the barygraph collection as Claude tools.

Provides sixteen tools:
  context_search   — MAIN entry point: $vectorSearch + full leaf content +
                      full ancestor chain to root, merged into a single call
  human_readable_search — plain search across ALL vectors; top 3 hits rendered
                      in human words only (glosses, relations, hierarchy) —
                      no ids, no parameters, no scores
  find_word        — look up word nodes (all POS variants)
  word_senses      — list all L15 sense glosses for a word
  word_edges       — L14 BaryEdges where the word is a CM
  edge_info        — details + CM structure for any BE/MB by id
  traverse_up      — walk parent_edge_id chain upward; shows triad at each MB level
  sample_metabary  — sample random MetaBary docs with triad + optional parent triad
  leaf_nodes       — all L15 sense / L14 word docs reachable from a BE or MB
  semantic_search  — $vectorSearch (requires mongot index from s10_index)
  create_sense     — insert a new L15 sense node (same schema as pipeline)
  create_word      — insert a new L14 word node; vector computed from sense/BE ids (s05 formula)
  create_edge      — insert a BaryEdge between two same-level nodes
  create_structure_meta_bary — form an SMB triad; allows already-parented CMs/bridge
  associative_search     — search THROUGH the graph, answer WITH destinations:
                           bounded multi-branch upward propagation, convergence
                           grouping, novelty, and top-k high-level coordinates
  associative_progressive — session-based discover -> expand -> compare -> propose
                           for BG Progressive / SMB construction workflows

Transports:
  stdio (default) — Claude Code / Claude Desktop:
      python -m scripts.mcp_server

  SSE — Claude mobile / Gemini / any HTTP MCP client:
      python -m scripts.mcp_server --transport sse [--host 0.0.0.0] [--port 8000]

  The SSE endpoint Claude mobile should point to:
      http://<host>:<port>/sse
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import anyio.to_thread
import numpy as np
from bson import ObjectId
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pymongo.errors import PyMongoError

from lib.assoc_search import (
    AssocConfig,
    _config_from_kwargs,
    progressive,
    run_search,
)
from lib.bary_vec import (
    TYPE_SENTENCES,
    compute_bary_vec,
    compute_metabary_vec,
    cosine,
    level_factor,
)
from lib.bary_vec import normalize as _norm_vec
from lib.config import Settings
from lib.db import cm_leaf_words, ensure_indexes, get_collection, vector_search
from lib.docs import baryedge as _make_baryedge
from lib.docs import metabary as _make_metabary
from lib.doi_bridge import dois_for_node, dois_for_nodes, get_bridge_collection
from lib.embed import get_embedder
from lib.log import setup_logging
from lib.vector import unpack_vec

# -- helpers ------------------------------------------------------------------


def _run_thr(fn, *args):
    """Run a blocking tool body in the worker pool (default thread limiter)."""
    return anyio.to_thread.run_sync(fn, *args)


def _leaf_nodes(be_id: Any, cap: int | None = None) -> dict[str, Any]:
    """BFS from a BE/MB down to all reachable L14/L15 node docs.

    Returns {"senses": [...], "words": [...], "truncated": bool} where senses
    are L15 sense nodes (gloss, tags, topics) and words are L14 word nodes
    (pos, ipa, etymology).

    ``cap`` bounds the traversal itself, not just the returned lists — a BE/MB
    near the root of the graph can fan out to a huge fraction of the corpus,
    so callers on a latency budget (e.g. context_search expanding several
    hits per call) should pass one. When the cap stops the walk early,
    "truncated" is True and the returned lists (and their counts) are a
    lower bound, not the full subtree.
    """
    frontier: set[Any] = {be_id}
    visited: set[Any] = set()
    senses: list[dict] = []
    words: list[dict] = []
    truncated = False

    for _ in range(15):
        to_fetch = frontier - visited
        if not to_fetch:
            break
        visited |= to_fetch
        next_frontier: set[Any] = set()
        for doc in _coll.find(
            {"_id": {"$in": list(to_fetch)}},
            {"doc_type": 1, "cm1_id": 1, "cm2_id": 1,
             "node_type": 1, "properties": 1},
        ):
            if doc.get("doc_type") == "node":
                props = doc.get("properties", {})
                nt = doc.get("node_type")
                if nt == "sense":
                    senses.append({
                        "id": str(doc["_id"]),
                        "word": props.get("word"),
                        "pos": props.get("pos"),
                        "gloss": props.get("gloss", ""),
                        "tags": props.get("tags", []),
                        "topics": props.get("topics", []),
                    })
                elif nt == "word":
                    words.append({
                        "id": str(doc["_id"]),
                        "word": props.get("word"),
                        "pos": props.get("pos"),
                        "ipa": props.get("ipa"),
                        "etymology": (props.get("etymology") or "")[:150] or None,
                    })
            else:
                if doc.get("cm1_id"):
                    next_frontier.add(doc["cm1_id"])
                if doc.get("cm2_id"):
                    next_frontier.add(doc["cm2_id"])
        if cap is not None and len(senses) + len(words) >= cap:
            truncated = True
            break
        frontier = next_frontier

    return {"senses": senses, "words": words, "truncated": truncated}

_settings = Settings.load()
setup_logging(_settings.log_level)
_log = logging.getLogger(__name__)
_coll = get_collection(_settings)
_bridge_coll = get_bridge_collection(_settings)

_public_host = os.environ.get("MCP_PUBLIC_HOST", "")
_allowed_hosts = ["localhost:*", "127.0.0.1:*"]
_allowed_origins = ["http://localhost:*", "http://127.0.0.1:*"]
if _public_host:
    _allowed_hosts.append(_public_host)
    _allowed_origins.append(f"https://{_public_host}")

_BG_INSTRUCTIONS = """BaryGraph (BG) is a semantic graph+vector index built from the kaikki.org
English Wiktionary (~6.7M docs: 1.44M words, 1.74M senses, 2.5M BaryEdges,
~989k MetaBary triads — the live instance is mid-build, so current counts may
run below these until the pipeline completes). Its core idea: **relationships
are first-class, searchable objects**, not thin edges between nodes. Every
relation is a *BaryEdge* (its own stored vector), and BaryEdges pair up into
*MetaBary triads* (child1 ↔ child2 via a **bridge**) that recurse upward into
increasingly abstract relation-clusters (L15 senses → L14 words → L13…L10
MetaBary levels).

Think of it as an **associative atlas / coordinate system**, not an answer
engine. A BG result is a *prompt for investigation*, never a proof.

## What you CAN get
- **Concepts near a query**: word nodes (IPA, etymology, forms, glosses) and
  sense nodes (gloss, tags, topics).
- **Relations as objects**: BaryEdges (edge_type: synonym/antonym/hypernym/…,
  strength *q*), MetaBary triads with the bridge — the concept-set connecting
  two otherwise-distant branches. This surfaces *productive unfamiliar
  adjacency*: structure that flat document search does not provide.
- **Everything in one call**: `context_search` returns each hit pre-expanded
  with its full leaf content **and** full ancestor chain to the root — the
  default starting point for any meaning-based question. No follow-up calls
  for most cases.
- **Exact-string lookups** when you need dictionary precision instead of
  semantics: `find_word`, `word_senses`, `word_edges`.
- **Deep-dives on a specific id**: `leaf_nodes` (full subtree),
  `traverse_up` (ancestry), `edge_info` (edge/triad details),
  `sample_metabary` (browse randomly).

## What you CANNOT get
- **Verified facts.** Results are coordinates to *interpret, test, reject, or
  develop* — not answers, citations, causal claims, or recommendations. The
  graph is a supervisor to no one and an answer key to no one.
- **Numbers / instance-level specifics.** It encodes methodology granularity
  ("bounded design windows", route types, unit families) — not filled values
  ("200 °C, 12 h") or full dimensional chains (those may require your own
  inference, e.g. Jy = kg/s²).
- **Confidence.** Noisy edges are expected; sometimes a "bad" edge is the most
  productive prompt. No edge is labeled reliable.
- **Coherent top levels.** Ancestry is a single linear parent chain, and
  L10/L11 clusters can degrade into unrelated residues (e.g. church/synapse
  words surfacing under a materials cluster) — discount them as background
  noise.

## How to use
1. Start with **`context_search`** for any meaning-based question: one call =
   hit + full subtree + full lineage.
2. Treat results as **coordinates worth investigating**: use them to form a
   better question, find a bridge/tension/contradiction, or reject them. The
   graph retrieves; you interpret. Two-phase workflow: *retrieve a coordinate,
   then do the real work* (domain knowledge, sources, reasoning).
3. Reach for `find_word`/`word_senses`/`word_edges` only for exact-string
   lookups, and for `semantic_search`/`leaf_nodes`/`traverse_up` only when
   `context_search` caps or raw dumps are needed.

## Links
- Repo (README + PoC spec):
  https://github.com/oleksiy-perepelytsya/bary-graph
- Zenodo pilot: https://doi.org/10.5281/zenodo.20186500
"""


mcp = FastMCP(
    "barygraph",
    instructions=_BG_INSTRUCTIONS,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    ),
)

_public_only = os.environ.get("MCP_PUBLIC", "0").lower() in ("1", "true", "yes")
_read_only = _public_only or os.environ.get("MCP_READ_ONLY", "0").lower() in ("1", "true", "yes")
_write_tool = (lambda f: f) if _read_only else mcp.tool()
_assoc_tool = (lambda f: f) if _public_only else mcp.tool()


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


_MAX_TEXT_LEN = 512


def _validate_text(value: str, name: str) -> str | None:
    """Return an error message if ``value`` is empty/whitespace-only or over-long, else None.

    Guards only absurd inputs (blank or oversized payloads) — no symbol/character
    filtering: the multilingual corpus legitimately contains words with dots,
    apostrophes, hyphens and diacritics, and every user string is bound as a
    parameterized filter value, never interpolated into a query.
    """
    if not value.strip():
        return f"{name} must be a non-empty string."
    if len(value) > _MAX_TEXT_LEN:
        return f"{name} is too long (max {_MAX_TEXT_LEN} characters)."
    return None


def _triad_of(
    mb_id: ObjectId,
    cm1_id: ObjectId,
    cm2_id: ObjectId,
    bridge_id: ObjectId | None = None,
) -> dict[str, Any]:
    """Fetch the bridge doc and return triad structure with leaf words for all three.

    Uses a single BFS across all three branches simultaneously instead of three
    separate cm_leaf_words traversals — cuts MongoDB round trips by ~3x.

    bridge_id: pre-resolved for SMBs (source='structural') which store it
    explicitly because they do not set parent_edge_id on their children.
    Falls back to the standard parent_edge_id reverse lookup for pipeline MBs.
    """
    if bridge_id is None:
        # Pipeline MB: bridge is the third child whose parent_edge_id = mb_id
        bridge_doc = _coll.find_one(
            {"parent_edge_id": mb_id, "_id": {"$nin": [cm1_id, cm2_id]}},
            {"_id": 1},
        )
        bridge_id = bridge_doc["_id"] if bridge_doc else None

    # origin maps each frontier id to its branch label; subtrees are disjoint
    # in a forest so there are no conflicts when propagating to children.
    origin: dict[Any, str] = {cm1_id: "child1", cm2_id: "child2"}
    if bridge_id:
        origin[bridge_id] = "bridge"

    words: dict[str, set[str]] = {"child1": set(), "child2": set(), "bridge": set()}
    visited: set[Any] = set()
    frontier: set[Any] = set(origin)

    for _ in range(15):
        to_fetch = frontier - visited
        if not to_fetch:
            break
        visited |= to_fetch
        next_frontier: set[Any] = set()
        for doc in _coll.find(
            {"_id": {"$in": list(to_fetch)}},
            {"doc_type": 1, "cm1_id": 1, "cm2_id": 1, "properties.word": 1},
        ):
            branch = origin[doc["_id"]]
            if doc.get("doc_type") == "node":
                w = doc.get("properties", {}).get("word")
                if w:
                    words[branch].add(w)
            else:
                for child_id in (doc.get("cm1_id"), doc.get("cm2_id")):
                    if child_id and child_id not in visited:
                        origin[child_id] = branch
                        next_frontier.add(child_id)
        frontier = next_frontier

    return {
        "child1": {"id": str(cm1_id), "words": sorted(words["child1"])},
        "child2": {"id": str(cm2_id), "words": sorted(words["child2"])},
        "bridge": {
            "id": str(bridge_id) if bridge_id else None,
            "words": sorted(words["bridge"]),
        },
    }


@mcp.tool()
def find_word(word: str) -> str:
    """Find a word in the graph. Returns all POS variants with edge counts and etymology."""
    err = _validate_text(word, "word")
    if err:
        return err
    docs = list(_coll.find(
        {"doc_type": "node", "node_type": "word", "properties.word": word},
        {"properties": 1, "parent_edge_id": 1},
    ))
    if not docs:
        return f"Word '{word}' not found. Try graph_stats to check if the graph is populated."

    word_ids = [d["_id"] for d in docs]
    # Batch edge counts with a single aggregation instead of one count per doc.
    edge_counts: dict[Any, int] = {d["_id"]: 0 for d in docs}
    for row in _coll.aggregate([
        {"$match": {"doc_type": "baryedge",
                    "$or": [{"cm1_id": {"$in": word_ids}},
                             {"cm2_id": {"$in": word_ids}}]}},
        {"$project": {"cm1_id": 1, "cm2_id": 1}},
    ]):
        for field in ("cm1_id", "cm2_id"):
            wid = row.get(field)
            if wid in edge_counts:
                edge_counts[wid] = edge_counts.get(wid, 0) + 1

    results = []
    for d in docs:
        p = d["properties"]
        results.append({
            "id": str(d["_id"]),
            "word": p["word"],
            "pos": p["pos"],
            "ipa": p.get("ipa"),
            "etymology": (p.get("etymology") or "")[:150] or None,
            "forms": (p.get("forms") or [])[:6],
            "sense_count": len(p.get("sense_ids") or []),
            "baryedge_count": edge_counts.get(d["_id"], 0),
            "has_parent_edge": d.get("parent_edge_id") is not None,
        })
    return _fmt(results)


@mcp.tool()
def word_senses(word: str, include_dois: bool = False) -> str:
    """List all L15 sense nodes for a word — glosses, tags, and whether each sense is paired.

    include_dois: when True, include each sense's source DOI(s) (academic-batch
    provenance) — stored directly on the sense node, so this adds no extra
    lookup. Most senses (plain kaikki dictionary data) have an empty list.
    """
    err = _validate_text(word, "word")
    if err:
        return err
    projection = {
        "properties.sense_idx": 1, "properties.pos": 1, "properties.gloss": 1,
        "properties.tags": 1, "properties.topics": 1, "parent_edge_id": 1,
    }
    if include_dois:
        projection["properties.doi"] = 1
    docs = list(_coll.find(
        {"doc_type": "node", "node_type": "sense", "properties.word": word},
        projection,
    ).sort("properties.sense_idx", 1))
    if not docs:
        return f"No senses found for '{word}' (word may not be in the graph)."
    results = [
        {
            "id": str(d["_id"]),
            "sense_idx": d["properties"].get("sense_idx"),
            "pos": d["properties"].get("pos"),
            "gloss": d["properties"].get("gloss", ""),
            "tags": d["properties"].get("tags", []),
            "topics": d["properties"].get("topics", []),
            "paired": d.get("parent_edge_id") is not None,
        }
        for d in docs
    ]
    if include_dois:
        for r, d in zip(results, docs, strict=True):
            r["dois"] = d["properties"].get("doi", [])
    return _fmt(results)


@mcp.tool()
def word_edges(word: str, pos: str = "") -> str:
    """Get L14 BaryEdges where this word is a CM (direct kaikki relations).

    Optionally filter by POS (noun, verb, adj, …).
    Returns edge_type, partner word, q, and accumulated_weight.
    """
    err = _validate_text(word, "word")
    if err:
        return err
    query: dict[str, Any] = {
        "doc_type": "node", "node_type": "word", "properties.word": word,
    }
    if pos:
        query["properties.pos"] = pos

    word_docs = list(_coll.find(query, {"_id": 1, "properties.pos": 1}))
    if not word_docs:
        return f"Word '{word}'" + (f" ({pos})" if pos else "") + " not found."

    word_ids = [d["_id"] for d in word_docs]
    edges = list(_coll.find(
        {"doc_type": "baryedge", "level": 14,
         "$or": [{"cm1_id": {"$in": word_ids}}, {"cm2_id": {"$in": word_ids}}]},
        {"cm1_id": 1, "cm2_id": 1, "edge_type": 1, "q": 1, "accumulated_weight": 1},
    ))
    if not edges:
        return f"No L14 edges found for '{word}'. It may be an orphan — check word_senses."

    all_cm_ids = list({e["cm1_id"] for e in edges} | {e["cm2_id"] for e in edges})
    id_to_label: dict[Any, str] = {}
    for d in _coll.find({"_id": {"$in": all_cm_ids}},
                        {"doc_type": 1, "properties.word": 1, "properties.pos": 1}):
        props = d.get("properties") or {}
        label = props.get("word") or d.get("doc_type", "?")
        if props.get("pos"):
            label += f" ({props['pos']})"
        id_to_label[d["_id"]] = label

    return _fmt([
        {
            "edge_id": str(e["_id"]),
            "edge_type": e.get("edge_type"),
            "cm1": id_to_label.get(e["cm1_id"], str(e["cm1_id"])),
            "cm2": id_to_label.get(e["cm2_id"], str(e["cm2_id"])),
            "q": e.get("q"),
            "accumulated_weight": e.get("accumulated_weight"),
        }
        for e in edges
    ])


@mcp.tool()
def edge_info(edge_id: str, include_dois: bool = False) -> str:
    """Get full details about a BaryEdge or MetaBary by id.

    For L14/L15 BaryEdges: shows edge_type, q, and flat CM leaf words.
    For MetaBary (L13–L10): shows the triad structure — child1, child2, and
    bridge — each with their own word sets, so you can see what concepts
    each branch represents and how they are connected.

    include_dois: when True, include the DOI(s) of every academic-batch
    source whose provenance chain passes through this edge/triad (union of
    its constituents' DOIs, via the doi_bridges reverse index) — empty for
    edges built entirely from plain kaikki dictionary data.
    """
    try:
        oid = ObjectId(edge_id)
    except Exception:
        return f"Invalid edge_id '{edge_id}' — must be a 24-char hex ObjectId string."

    doc = _coll.find_one({"_id": oid})
    if not doc:
        return f"No document with id {edge_id}."

    level = doc.get("level")
    result: dict[str, Any] = {
        "id": edge_id,
        "level": level,
        "connection_strength": doc.get("connection_strength"),
        "accumulated_weight": doc.get("accumulated_weight"),
        "has_parent": doc.get("parent_edge_id") is not None,
        "parent_id": str(doc["parent_edge_id"]) if doc.get("parent_edge_id") else None,
    }

    if level is not None and level <= 13:
        # MetaBary: show triad (child1, child2, bridge) with leaf words per branch.
        result["triad"] = _triad_of(oid, doc["cm1_id"], doc["cm2_id"], doc.get("bridge_id"))
    else:
        # L14/L15 BaryEdge: flat leaf words + relation details.
        result["edge_type"] = doc.get("edge_type")
        result["q"] = doc.get("q")
        result["cm1_id"] = str(doc.get("cm1_id"))
        result["cm2_id"] = str(doc.get("cm2_id"))
        result["cm_leaf_words"] = sorted(cm_leaf_words(_coll, oid))

    if include_dois:
        result["dois"] = dois_for_node(_bridge_coll, oid)

    return _fmt(result)


@mcp.tool()
async def traverse_up(edge_id: str, max_levels: int = 6) -> str:
    """Walk the parent_edge_id chain upward from any BE or MB.

    Returns the ancestry chain from the starting edge to the root (or until
    max_levels is reached / parent is null). For MetaBary levels (≤13) each
    step includes the full triad structure (child1, child2, bridge with their
    word sets); for L14/L15 BaryEdges shows flat leaf words and edge_type.

    max_levels: how many steps up to walk (max 20; clamped to [1, 20]).
    """
    return await _run_thr(_traverse_up_body, edge_id, max_levels)


def _traverse_up_body(edge_id: str, max_levels: int) -> str:
    max_levels = min(max(max_levels, 1), 20)
    try:
        current_id: Any = ObjectId(edge_id)
    except Exception:
        return f"Invalid edge_id '{edge_id}'."

    chain = []
    for _ in range(max_levels):
        doc = _coll.find_one(
            {"_id": current_id},
            {"level": 1, "parent_edge_id": 1, "edge_type": 1,
             "connection_strength": 1, "cm1_id": 1, "cm2_id": 1},
        )
        if not doc:
            break
        level = doc.get("level")
        step: dict[str, Any] = {
            "id": str(doc["_id"]),
            "level": level,
            "connection_strength": doc.get("connection_strength"),
        }
        if level is not None and level <= 13:
            step["triad"] = _triad_of(
                doc["_id"], doc["cm1_id"], doc["cm2_id"], doc.get("bridge_id")
            )
        else:
            step["edge_type"] = doc.get("edge_type")
            step["leaf_words"] = sorted(cm_leaf_words(_coll, doc["_id"]))
        chain.append(step)
        parent_id = doc.get("parent_edge_id")
        if not parent_id:
            break
        current_id = parent_id

    return _fmt({"starting_id": edge_id, "chain_length": len(chain), "chain": chain})


@mcp.tool()
async def sample_metabary(level: int, n: int = 5, with_parent: bool = True) -> str:
    """Sample N random MetaBary docs at the given level with full triad structure.

    level: 10–13 (13 = closest to individual senses, 10 = most abstract).
    n: number to sample, max 1000 (default 5 — triads are verbose).
    with_parent: when True (default), include the parent MB's triad structure
      if one exists, so you see both this level and the level above in one call.

    Each result shows the three constituents of the MetaBary triad:
    - child1 and child2: the two BEs/MBs being bridged
    - bridge: the BE/MB that connects them
    Each branch is explained as the set of words reachable through it, so you
    can read a MetaBary as "child1-words ↔ child2-words via bridge-words".
    """
    return await _run_thr(
        _sample_metabary_body, level, n, with_parent
    )


def _sample_metabary_body(level: int, n: int, with_parent: bool) -> str:
    if not (10 <= level <= 13):
        return "level must be between 10 and 13 (MetaBary range)."
    n = min(max(n, 1), 1000)

    docs = list(_coll.aggregate([
        {"$match": {"doc_type": "baryedge", "level": level}},
        {"$sample": {"size": n}},
        {"$project": {"cm1_id": 1, "cm2_id": 1,
                      "connection_strength": 1, "accumulated_weight": 1,
                      "parent_edge_id": 1}},
    ]))
    if not docs:
        return f"No MetaBary docs found at level {level}. Run graph_stats to check pipeline state."

    results = []
    for doc in docs:
        mb_id = doc["_id"]
        entry: dict[str, Any] = {
            "id": str(mb_id),
            "level": level,
            "connection_strength": doc.get("connection_strength"),
            "accumulated_weight": doc.get("accumulated_weight"),
            "triad": _triad_of(mb_id, doc["cm1_id"], doc["cm2_id"], doc.get("bridge_id")),
            "parent": None,
        }
        parent_oid = doc.get("parent_edge_id")
        if with_parent and parent_oid:
            pdoc = _coll.find_one(
                {"_id": parent_oid},
                {"level": 1, "cm1_id": 1, "cm2_id": 1,
                 "connection_strength": 1, "parent_edge_id": 1},
            )
            if pdoc:
                entry["parent"] = {
                    "id": str(pdoc["_id"]),
                    "level": pdoc.get("level"),
                    "connection_strength": pdoc.get("connection_strength"),
                    "has_grandparent": pdoc.get("parent_edge_id") is not None,
                    "triad": _triad_of(
                        pdoc["_id"], pdoc["cm1_id"], pdoc["cm2_id"],
                        pdoc.get("bridge_id"),
                    ),
                }
        results.append(entry)
    return _fmt(results)


@mcp.tool()
async def leaf_nodes(edge_id: str, max_leaves: int | None = None) -> str:
    """Get all L15 sense nodes and L14 word nodes reachable from a BE or MB.

    Traverses the full CM lineage downward to leaf nodes and returns the
    actual documents with semantic content — not just word strings:
    - senses: L15 sense nodes with gloss, tags, and topics
    - words:  L14 word nodes with pos, ipa, and etymology snippet

    Use this to build full search context for a MB or BE found via
    semantic_search or sample_metabary — see every sense and word the
    edge encodes.

    max_leaves: optional cap on the number of senses+words returned (bounds the
      traversal itself, not just the returned lists — useful for near-root MBs
      that fan out to a huge fraction of the corpus). Default: no cap (full
      subtree dump).
    """
    return await _run_thr(_leaf_nodes_body, edge_id, max_leaves)


def _leaf_nodes_body(edge_id: str, max_leaves: int | None) -> str:
    try:
        oid = ObjectId(edge_id)
    except Exception:
        return f"Invalid edge_id '{edge_id}' — must be a 24-char hex ObjectId string."
    if not _coll.find_one({"_id": oid}, {"_id": 1}):
        return f"No document with id {edge_id}."
    cap = max(max_leaves, 1) if max_leaves is not None else None
    result = _leaf_nodes(oid, cap=cap)
    result["edge_id"] = edge_id
    result["sense_count"] = len(result["senses"])
    result["word_count"] = len(result["words"])
    return _fmt(result)


@mcp.tool()
async def semantic_search(
    query: str, doc_type: str = "baryedge", top_k: int = 10, include_dois: bool = False
) -> str:
    """Semantic similarity search against the BaryGraph vector index (mongot).

    doc_type: 'baryedge' searches relationship vectors (default);
              'node' searches word/sense vectors.
    top_k: number of hits to return (max 20).
    Requires s10_index to have completed. The HNSW index may take several
    minutes to build after creation.
    include_dois: when True, attach each hit's DOI(s) — academic-batch source
    provenance. Sense hits read this straight off their own properties.doi;
    word/edge/MB hits use one batched doi_bridges lookup across all hits
    (union of their constituents' DOIs). Empty for hits built entirely from
    plain kaikki dictionary data.
    """
    try:
        return await _run_thr(
            _semantic_search_body, query, doc_type, top_k, include_dois
        )
    except Exception as e:  # pragma: no cover - defensive
        _log.exception("semantic_search failed for query=%r", query)
        return _fmt({"status": "error", "query": query, "message": str(e)})


def _semantic_search_body(
    query: str, doc_type: str, top_k: int, include_dois: bool
) -> str:
    err = _validate_text(query, "query")
    if err:
        return err
    if doc_type not in ("node", "baryedge"):
        return "doc_type must be 'node' or 'baryedge'."
    top_k = min(max(top_k, 1), 20)

    try:
        embedder = get_embedder(_settings)
        qv = embedder.embed([query])[0].tolist()
    except Exception as e:
        _log.exception("semantic_search: embedding failed for query=%r", query)
        return f"Embedding failed — is Ollama running at {_settings.ollama_url}?\nError: {e}"

    try:
        docs = vector_search(
            _coll, qv,
            limit=top_k,
            num_candidates=max(top_k * 10, 200),
            filter={"doc_type": doc_type},
        )
    except PyMongoError as e:
        # Covers both a missing/still-building mongot index (OperationFailure)
        # and a slow/overloaded one (ExecutionTimeout, ServerSelectionTimeoutError,
        # ...) — narrowly catching OperationFailure let the latter escape uncaught.
        _log.exception("semantic_search: vector_search failed for query=%r", query)
        return (
            "Vector search failed — the mongot index may still be building, or "
            f"the query timed out under load. Error: {type(e).__name__}: {e}"
        )

    if not docs:
        return "No results returned. Index may still be building or corpus is empty."

    # Hits needing the aggregated reverse lookup (everything above sense level);
    # sense hits carry their own DOIs already, so they're excluded from the batch.
    agg_lookup_ids = [
        d["_id"] for d in docs
        if not (d["doc_type"] == "node" and d.get("node_type") == "sense")
    ] if include_dois else []
    dois_by_id = dois_for_nodes(_bridge_coll, agg_lookup_ids) if agg_lookup_ids else {}

    results = []
    for d in docs:
        r: dict[str, Any] = {
            "id": str(d["_id"]),
            "score": round(float(d.get("_score", 0)), 4),
            "level": d.get("level"),
        }
        if d["doc_type"] == "node":
            r["node_type"] = d.get("node_type")
            r["word"] = d.get("properties", {}).get("word")
            r["gloss"] = (d.get("properties", {}).get("gloss") or "")[:100]
        else:
            r["edge_type"] = d.get("edge_type")
            r["accumulated_weight"] = d.get("accumulated_weight")
            if d.get("level") is not None and d["level"] <= 13:
                r["triad_words"] = _triad_of(
                    d["_id"], d["cm1_id"], d["cm2_id"], d.get("bridge_id")
                )
            else:
                r["cm_words"] = sorted(cm_leaf_words(_coll, d["_id"]))
        if include_dois:
            if d["doc_type"] == "node" and d.get("node_type") == "sense":
                r["dois"] = d.get("properties", {}).get("doi", [])
            else:
                r["dois"] = dois_by_id.get(d["_id"], [])
        results.append(r)

    return _fmt(results)


def _node_context(doc: dict[str, Any], max_leaves: int) -> dict[str, Any]:
    """Expand a node hit (word or sense) with everything context_search promises inline."""
    props = doc.get("properties", {})
    nt = doc.get("node_type")
    ctx: dict[str, Any] = {"node_type": nt, "word": props.get("word"), "pos": props.get("pos")}
    if nt == "sense":
        ctx.update({
            "gloss": props.get("gloss", ""),
            "tags": props.get("tags", []),
            "topics": props.get("topics", []),
        })
    elif nt == "word":
        ctx.update({
            "ipa": props.get("ipa"),
            "etymology": (props.get("etymology") or "")[:150] or None,
            "forms": (props.get("forms") or [])[:6],
        })
        sense_docs = list(_coll.find(
            {"doc_type": "node", "node_type": "sense", "properties.word": props.get("word")},
            {"properties.pos": 1, "properties.gloss": 1, "properties.sense_idx": 1},
        ).sort("properties.sense_idx", 1).limit(max_leaves))
        ctx["senses"] = [
            {
                "id": str(s["_id"]), "pos": s["properties"].get("pos"),
                "gloss": s["properties"].get("gloss", ""),
            }
            for s in sense_docs
        ]
    return ctx


def _edge_context(doc: dict[str, Any], max_leaves: int) -> dict[str, Any]:
    """Expand a BE/MB hit with its triad (if MB) and full leaf senses/words."""
    level = doc.get("level")
    ctx: dict[str, Any] = {"level": level, "connection_strength": doc.get("connection_strength")}
    # Cap traversal, not just the returned list: a BE/MB near the root can fan out
    # to a huge fraction of the corpus, and an oblique query is more likely to
    # match one. Without this, one broad hit can blow the tool's response budget.
    leaves = _leaf_nodes(doc["_id"], cap=max(max_leaves * 5, 300))
    if level is not None and level <= 13:
        ctx["triad"] = _triad_of(doc["_id"], doc["cm1_id"], doc["cm2_id"], doc.get("bridge_id"))
    else:
        ctx["edge_type"] = doc.get("edge_type")
        ctx["q"] = doc.get("q")
    ctx["senses"] = leaves["senses"][:max_leaves]
    ctx["words"] = leaves["words"][:max_leaves]
    # "at least N" when the traversal itself was cut short — don't report a
    # partial count as if it were the true subtree size.
    n_senses, n_words = len(leaves["senses"]), len(leaves["words"])
    if leaves["truncated"]:
        ctx["sense_count"] = f"at least {n_senses}"
        ctx["word_count"] = f"at least {n_words}"
    else:
        ctx["sense_count"] = n_senses
        ctx["word_count"] = n_words
    return ctx


def _ancestry_chain(
    parent_id: Any, expanded_ids: set[str], max_levels: int = 20
) -> list[dict[str, Any]]:
    """Walk parent_edge_id all the way to the root — unlike leaf traversal this is a
    single linear chain (one parent per doc), never wide, so going the full distance
    is cheap. The one exception is the L14/L15 BE steps near the bottom of the
    chain: each of those still fans out downward via cm_leaf_words, so it's capped
    the same way _edge_context caps _leaf_nodes. Branches off the chain itself are
    what leaf_nodes/context_search on a specific id are for.

    ``expanded_ids`` is shared across every hit in one context_search call. Hits
    close together in the graph routinely share lineage — one hit's ancestor is
    often another hit itself, or two hits climb through the same lower-level MB.
    Without this, each occurrence re-fetches and re-serializes an identical
    triad/leaf-word structure, multiplying response size for zero new
    information; a top_k of 6 on a tight cluster can trigger this several
    times over and blow past a client-side payload limit. Once an id has been
    fully rendered anywhere in the response (as a hit or as another chain's
    step), later encounters stop and leave a pointer instead.
    """
    chain: list[dict[str, Any]] = []
    current_id = parent_id
    for _ in range(max_levels):
        pdoc = _coll.find_one(
            {"_id": current_id},
            {"level": 1, "edge_type": 1, "parent_edge_id": 1,
             "cm1_id": 1, "cm2_id": 1, "bridge_id": 1, "connection_strength": 1},
        )
        if not pdoc:
            break
        id_str = str(pdoc["_id"])
        if id_str in expanded_ids:
            chain.append({
                "id": id_str,
                "level": pdoc.get("level"),
                "already_shown_elsewhere_in_response": True,
            })
            break
        expanded_ids.add(id_str)
        step: dict[str, Any] = {
            "id": id_str,
            "level": pdoc.get("level"),
            "connection_strength": pdoc.get("connection_strength"),
        }
        if pdoc.get("level") is not None and pdoc["level"] <= 13:
            step["triad_words"] = _triad_of(
                pdoc["_id"], pdoc["cm1_id"], pdoc["cm2_id"],
                pdoc.get("bridge_id"),
            )
        else:
            step["edge_type"] = pdoc.get("edge_type")
            leaf_words_cap = 200
            leaf_words = cm_leaf_words(_coll, pdoc["_id"], max_words=leaf_words_cap)
            step["leaf_words"] = sorted(leaf_words)
            # cm_leaf_words returns a bare set with no truncation signal of its own —
            # hitting the cap exactly is the only way to tell "capped" from "this
            # subtree genuinely has ~200 words" apart, so treat it as truncated.
            if len(leaf_words) >= leaf_words_cap:
                step["leaf_words_truncated"] = True
        chain.append(step)
        next_id = pdoc.get("parent_edge_id")
        if not next_id:
            break
        current_id = next_id
    return chain


@mcp.tool()
async def context_search(
    query: str,
    doc_type: str = "any",
    top_k: int = 5,
    max_leaves: int = 40,
    include_ancestry: bool = True,
) -> str:
    """Main entry point for meaning-based lookup — semantic search with full context inline.

    Runs the same $vectorSearch as semantic_search, but each hit is returned
    already expanded with everything you'd otherwise need leaf_nodes and
    traverse_up follow-up calls to get:
      - word node   → properties (ipa/etymology/forms) + all its senses (gloss list)
      - sense node  → gloss/tags/topics
      - L14/L15 BE  → edge_type + every sense/word reachable underneath (leaf_nodes)
      - MetaBary (L10-L13) → triad (child1/child2/bridge word sets) +
        every sense/word reachable underneath (leaf_nodes)
    Use this first for any "find things related to X" question; reach for
    semantic_search/leaf_nodes/traverse_up directly only when you need a raw
    dump, a full multi-level ancestry chain, or to page through more hits than
    context_search's caps allow.

    doc_type: 'node' (words+senses only), 'baryedge' (BE/MB relations only),
      or 'any' (default — no filter, so results are ranked across both
      spaces together by the same HNSW index; the best default when you don't
      know whether the answer is a word or a relation/cluster).
    top_k: number of hits to return (max 20).
    max_leaves: cap on senses/words listed per hit for BE/MB results, so a
      broad L10 MetaBary doesn't dump its whole subtree (max 200). Totals are
      still reported via sense_count/word_count even when truncated.
    include_ancestry: when True (default), also includes the full ancestor_chain
      from each hit up to the root — triad for MB ancestors,
      edge_type/leaf_words for BE ancestors — so you see everything the hit
      belongs to, all the way up, without a separate traverse_up call. This
      is cheap because parent_edge_id is a single linear chain (one parent
      per doc), unlike the fan-out you get traversing down — so there's no
      partial "one level" option; you get the whole chain or none. If you
      instead need to explore sideways/downward from a specific ancestor id,
      call context_search or leaf_nodes on that id directly. Set False to
      skip the chain for a faster, terser response. Hits close together in
      the graph often share lineage (one hit is literally another hit's
      ancestor, or two hits climb through the same lower MB) — when an id
      has already been fully shown elsewhere in the response, later
      occurrences collapse to {"id", "level", "already_shown_elsewhere_in_response":
      true} instead of repeating the same triad/leaf-word content.
    """
    try:
        return await _run_thr(
            _context_search_body, query, doc_type, top_k, max_leaves,
            include_ancestry,
        )
    except Exception as e:  # pragma: no cover - defensive
        _log.exception("context_search failed for query=%r", query)
        return _fmt({"status": "error", "query": query, "message": str(e)})


def _context_search_body(
    query: str, doc_type: str, top_k: int, max_leaves: int,
    include_ancestry: bool,
) -> str:
    err = _validate_text(query, "query")
    if err:
        return err
    if doc_type not in ("node", "baryedge", "any"):
        return "doc_type must be 'node', 'baryedge', or 'any'."
    top_k = min(max(top_k, 1), 20)
    max_leaves = min(max(max_leaves, 1), 200)

    try:
        embedder = get_embedder(_settings)
        qv = embedder.embed([query])[0].tolist()
    except Exception as e:
        _log.exception("context_search: embedding failed for query=%r", query)
        return f"Embedding failed — is Ollama running at {_settings.ollama_url}?\nError: {e}"

    filt = {"doc_type": doc_type} if doc_type in ("node", "baryedge") else None
    try:
        docs = vector_search(
            _coll, qv,
            limit=top_k,
            num_candidates=max(top_k * 10, 200),
            filter=filt,
        )
    except PyMongoError as e:
        # Covers both a missing/still-building mongot index (OperationFailure)
        # and a slow/overloaded one (ExecutionTimeout, ServerSelectionTimeoutError,
        # ...) — narrowly catching OperationFailure let the latter escape uncaught.
        _log.exception("context_search: vector_search failed for query=%r", query)
        return (
            "Vector search failed — the mongot index may still be building, or "
            f"the query timed out under load. Error: {type(e).__name__}: {e}"
        )
    if not docs:
        return "No results returned. Index may still be building or corpus is empty."

    # Outer safety net: expanding hits (leaf traversal, ancestry chains, mongo
    # round trips) touches enough moving parts that something not covered by
    # the per-hit try/except below could still slip through. Rather than let
    # that reach the caller as FastMCP's generic, undiagnosable "Error occurred
    # during tool execution", log the real cause and return whatever partial
    # results were already built.
    results: list[dict[str, Any]] = []
    # Hits close together in the graph routinely share lineage — see
    # _ancestry_chain's docstring. Seed with every hit's own id up front (not
    # as each hit is processed) so the dedup works regardless of which hit's
    # ancestor chain reaches a sibling hit first.
    expanded_ids: set[str] = {str(d["_id"]) for d in docs}
    try:
        for d in docs:
            r: dict[str, Any] = {
                "id": str(d["_id"]),
                "score": round(float(d.get("_score", 0)), 4),
                "doc_type": d.get("doc_type"),
            }
            # Per-hit try/except: one slow/broken hit (e.g. a mongo hiccup expanding
            # a huge subtree) shouldn't take down every other hit in the batch —
            # surface it as a partial result instead of a bare failed tool call.
            try:
                if d["doc_type"] == "node":
                    r.update(_node_context(d, max_leaves))
                else:
                    r.update(_edge_context(d, max_leaves))

                if include_ancestry and d.get("parent_edge_id"):
                    r["ancestor_chain"] = _ancestry_chain(d["parent_edge_id"], expanded_ids)
            except Exception as e:
                _log.exception("context_search: expansion failed for hit id=%s", d.get("_id"))
                r["expansion_error"] = (
                    f"{type(e).__name__}: {e}. Retry with include_ancestry=False "
                    "and/or a lower max_leaves/top_k to avoid expanding this hit's "
                    "subtree, or call leaf_nodes/traverse_up on this id directly."
                )

            results.append(r)
    except Exception as e:
        _log.exception("context_search: unhandled failure for query=%r", query)
        if results:
            return _fmt(results) + (
                f"\n\n[Stopped early after {len(results)} hit(s): "
                f"{type(e).__name__}: {e}. Retry with a lower top_k/max_leaves "
                "or include_ancestry=False.]"
            )
        return (
            f"context_search failed: {type(e).__name__}: {e}\n"
            "Retry with a lower top_k/max_leaves and/or include_ancestry=False."
        )

    return _fmt(results)


# --- human_readable_search: pure words + relations, hierarchy as text ---------


def _hl_words(words, cap: int = 6) -> str:
    """Compact human word list: 'a, b, c, …' — no ids, no parameters."""
    ws = sorted({w for w in words if w})
    shown = ws[:cap]
    s = ", ".join(shown)
    if len(ws) > cap:
        s += f", … (+{len(ws) - cap})"
    return s


def _hl_ancestry(parent_id: Any, cap_words: int = 5) -> list[str]:
    """Rendering of the upward chain (parent_edge_id) as readable one-liners.

    Returns lines from the parent up to the root, each line just words and the
    relation kind — levels L14/L15 BEs as 'type + words', MetaBary levels as
    'child1 ↔ child2 via bridge' (words only, capped). Pure indexed lookups.
    """
    steps: list[str] = []
    current = parent_id
    for _ in range(12):
        pdoc = _coll.find_one(
            {"_id": current},
            {"level": 1, "edge_type": 1, "parent_edge_id": 1,
             "cm1_id": 1, "cm2_id": 1, "bridge_id": 1},
        )
        if not pdoc:
            break
        lvl = pdoc.get("level")
        if lvl is not None and lvl <= 13:
            tr = _triad_of(pdoc["_id"], pdoc.get("cm1_id"), pdoc.get("cm2_id"),
                           pdoc.get("bridge_id"))
            steps.append(
                f"MetaBary L{lvl}: {{{_hl_words(tr['child1']['words'], cap_words)}}} "
                f"↔ {{{_hl_words(tr['child2']['words'], cap_words)}}} "
                f"via {{{_hl_words(tr['bridge']['words'], cap_words)}}}"
            )
        else:
            wset = cm_leaf_words(_coll, pdoc["_id"], max_words=cap_words * 4)
            steps.append(
                f"L{lvl} {pdoc.get('edge_type') or 'relation'}: "
                f"{{{_hl_words(wset, cap_words)}}}"
            )
        current = pdoc.get("parent_edge_id")
        if not current:
            break
    return steps


# When the query reads as a list of separate words ("science art music"), embed
# each term on its own and show one block per word — a single whole-phrase
# vector otherwise collapses every hit into whichever term dominates the blend.
_MULTIWORD_STOP = {
    "a", "an", "the", "and", "or", "nor", "but", "of", "to", "for", "in",
    "on", "at", "by", "with", "from", "as", "into", "via", "vs", "over",
    "under", "between",
}
_MULTIWORD_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def _search_terms(query: str) -> list[str]:
    """Split into lowercased search terms — stopwords and dupes removed."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _MULTIWORD_WORD_RE.finditer(query):
        t = m.group(0).lower()
        if t in _MULTIWORD_STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _word_hit_for(term: str, cap: int = 8) -> dict[str, Any] | None:
    """Nearest node hit for a single term, preferring a literal word node.

    Returns None if the embed or search fails — the whole-phrase fallback path
    reports the real error instead.
    """
    try:
        qv = get_embedder(_settings).embed([term])[0].tolist()
    except Exception:
        return None
    try:
        docs = vector_search(_coll, qv, limit=cap, num_candidates=200)
    except PyMongoError:
        return None
    for d in docs:
        if d.get("doc_type") == "node" and d.get("node_type") == "word":
            if ((d.get("properties") or {}).get("word") or "").lower() == term:
                return d
    for d in docs:
        if d.get("doc_type") == "node" and d.get("node_type") == "word":
            return d
    return docs[0] if docs else None


def _hl_render_hit(d: dict[str, Any]) -> str:
    """Render one vector-search hit as words + relations + hierarchy (no ids/params)."""
    props = d.get("properties") or {}
    lines: list[str] = []
    if d.get("doc_type") == "node":
        nt = d.get("node_type")
        word = props.get("word", "?")
        pos = props.get("pos")
        lines.append(f"{nt}: {word}" + (f" ({pos})" if pos else ""))
        if nt == "sense":
            # sense hits: only the full gloss (per tool contract)
            lines.append(f"gloss: {props.get('gloss', '')}")
        elif nt == "word":
            for i, s_ in enumerate(
                _coll.find(
                    {"doc_type": "node", "node_type": "sense",
                     "properties.word": word},
                    {"properties.gloss": 1},
                ).sort("properties.sense_idx", 1).limit(3), 1
            ):
                lines.append(f"gloss {i}: {s_.get('properties', {}).get('gloss', '')}")
        if d.get("parent_edge_id"):
            anc = _hl_ancestry(d["parent_edge_id"])
            if anc:
                lines.append("  in: " + "\n  in: ".join(anc))
        return "\n".join(lines)

    lvl = d.get("level")
    if lvl is not None and lvl <= 13:
        try:
            tr = _triad_of(d["_id"], d.get("cm1_id"), d.get("cm2_id"),
                           d.get("bridge_id"))
        except Exception:
            tr = None
        if tr:
            lines.append(f"MetaBary L{lvl}")
            lines.append(f"  child1: {{{_hl_words(tr['child1']['words'], 7)}}}")
            lines.append(f"  child2: {{{_hl_words(tr['child2']['words'], 7)}}}")
            lines.append(f"  bridge: {{{_hl_words(tr['bridge']['words'], 7)}}}")
        else:
            lines.append(f"MetaBary L{lvl}")
    else:
        wset = cm_leaf_words(_coll, d["_id"], max_words=32)
        lines.append(
            f"BaryEdge L{lvl}"
            + (f" · {d.get('edge_type')}" if d.get("edge_type") else "")
            + f" — {{{_hl_words(wset, 8)}}}"
        )
    if d.get("parent_edge_id"):
        anc = _hl_ancestry(d["parent_edge_id"])
        if anc:
            lines.append("  in: " + "\n  in: ".join(anc))
    return "\n".join(lines)


@mcp.tool()
async def human_readable_search(query: str) -> str:
    """Plain-search the whole graph and answer in human words, with hierarchy.

    Searches ALL indexed vectors at once — senses, words, BaryEdges and
    MetaBary — via a single fast $vectorSearch, and returns the top 3 hits
    rendered as words + relations + the ancestry chain they live in.

    Queries consisting of several separate words ("science art music") are
    embedded per term and rendered as one block per word, so each requested
    word gets its own relation neighborhood rather than the whole phrase
    collapsing into one dominant concept.

    No ids, no parameters, no technical fields:
      - sense hit   → just the full gloss
      - word hit    → word with a few glosses, plus the relation it hangs off
      - edge hit    → relation kind + the words it couples
      - MetaBary    → child1 ↔ child2 via bridge, as word sets
    The 'in: …' lines list the parent chain the hit is reachable in.
    """
    err = _validate_text(query, "query")
    if err:
        return err

    return await _run_thr(_human_readable_search_body, query)


def _human_readable_search_body(query: str) -> str:
    # Multi-word query ("science art music") → one rendered block per term, so
    # each requested word gets its own coordinates instead of the whole phrase
    # collapsing into the single dominant neighborhood.
    terms = _search_terms(query)
    if len(terms) >= 2:
        blocks: list[str] = []
        seen: set[str] = set()
        for term in terms:
            hit = _word_hit_for(term)
            if not hit or str(hit["_id"]) in seen:
                continue
            seen.add(str(hit["_id"]))
            try:
                body = _hl_render_hit(hit)
            except Exception as e:
                _log.exception("human_readable_search: render failed for id=%s", hit.get("_id"))
                body = f"(could not expand this hit: {type(e).__name__}: {e})"
            blocks.append(f"[{term}] {body.strip()}")
        if len(blocks) >= 2:
            return "\n\n".join(blocks)

    try:
        embedder = get_embedder(_settings)
        qv = embedder.embed([query])[0].tolist()
    except Exception as e:
        _log.exception("human_readable_search: embedding failed for query=%r", query)
        return f"Embedding failed — is Ollama running at {_settings.ollama_url}?\nError: {e}"

    top_k = 3
    try:
        docs = vector_search(_coll, qv, limit=top_k, num_candidates=max(top_k * 10, 100))
    except PyMongoError as e:
        _log.exception("human_readable_search: vector_search failed for query=%r", query)
        return (
            "Vector search failed — the mongot index may still be building, or "
            f"the query timed out under load. Error: {type(e).__name__}: {e}"
        )
    if not docs:
        return "No results returned. Index may still be building or corpus is empty."

    blocks: list[str] = []
    for i, d in enumerate(docs, 1):
        try:
            body = _hl_render_hit(d)
        except Exception as e:
            _log.exception("human_readable_search: render failed for id=%s", d.get("_id"))
            body = f"(could not expand this hit: {type(e).__name__}: {e})"
        blocks.append(f"[{i}] {body.strip()}")
    return "\n\n".join(blocks)


# q_seed lookup: edge_type → q_seeds key (same_phenomenon maps to synonyms tier)
_EDGE_TYPE_Q_KEY: dict[str, str] = {
    "contradicts":     "contradicts",
    "applies_to":      "applies_to",
    "is_instance_of":  "is_instance_of",
    "extends":         "extends",
    "same_phenomenon": "synonyms",
}


@_write_tool
async def create_sense(
    word: str,
    pos: str,
    gloss: str,
    examples: list[str] | None = None,
    tags: list[str] | None = None,
    topics: list[str] | None = None,
) -> str:
    """Create a new L15 sense node, exactly as the ingestion pipeline would.

    Embeds the gloss (+ up to 2 examples) via Ollama, then inserts a node
    document with the standard schema. parent_edge_id is None (orphan) until
    the node is paired via create_edge.

    word does not need to exist as an L14 word node yet — properties.word is
    a plain string, not a reference. Create senses first, then create_word.

    Returns the new document's _id.
    """
    return await _run_thr(
        _create_sense_body,
        word, pos, gloss, examples or [], tags or [], topics or [],
    )


def _create_sense_body(
    word: str, pos: str, gloss: str,
    examples: list[str], tags: list[str], topics: list[str],
) -> str:
    examples = examples or []
    tags = tags or []
    topics = topics or []

    try:
        embedder = get_embedder(_settings)
        embed_text = (gloss + " " + " ".join(examples[:2])).strip()
        vector = embedder.embed([embed_text])[0].tolist()
    except Exception as e:
        return f"Embedding failed — is Ollama running at {_settings.ollama_url}?\nError: {e}"

    ts = datetime.now(timezone.utc)
    doc: dict[str, Any] = {
        "doc_type": "node",          # always 'node' for sense/word docs
        "node_type": "sense",        # L15 = sense (individual gloss); L14 = word
        "level": 15,                 # bottom of the hierarchy; sense sits below word
        "label": f"{word} ({pos}) [0]",  # human-readable; sense_idx hardcoded 0 for new nodes
        "vector": vector,            # 768-dim nomic-embed-text of gloss + examples[:2]
        "surface": 1,                # number of surface forms; 1 for a single new sense
        "rotation": 0.0,             # reserved for future orientation encoding
        "parent_edge_id": None,      # orphan until paired by create_edge / pipeline
        "properties": {
            "word": word,
            "pos": pos,
            "sense_id": None,        # kaikki stable sense id; None for manually created nodes
            "sense_idx": 0,          # position within the word's sense list; 0 for new nodes
            "gloss": gloss,
            "examples": [{"text": e} for e in examples],
            "tags": tags,            # kaikki tags e.g. ["informal", "archaic"]
            "topics": topics,        # kaikki topics e.g. ["medicine", "computing"]
            "wikidata": [],          # Wikidata QIDs; empty for manually created nodes
        },
        "created_at": ts,
        "updated_at": ts,
    }
    result = _coll.insert_one(doc)
    return _fmt({"ok": True, "id": str(result.inserted_id), "label": doc["label"]})


@_write_tool
async def create_word(
    word: str,
    pos: str,
    source_ids: list[str],
    ipa: str = "",
    etymology: str = "",
) -> str:
    """Create a new L14 word node, exactly as s03 + s05 would produce.

    source_ids: mix of L15 sense node IDs and/or L15 BE IDs that cover this
    word's senses. The word vector is the normalized centroid of their vectors —
    the same formula s05_word_vectors.py uses:
      v(word) = normalize( Σ v(BE_i) + Σ v(orphan_sense_j) )

    Pass BE IDs for senses that were paired into a BE via create_edge, and
    sense node IDs for any senses still unpaired (orphans). Mixed lists are fine.

    ipa and etymology are optional metadata stored in properties.

    Returns the new word node's _id.
    """
    return await _run_thr(
        _create_word_body, word, pos, source_ids, ipa, etymology
    )


def _create_word_body(
    word: str, pos: str, source_ids: list[str], ipa: str, etymology: str,
) -> str:
    if not source_ids:
        return "source_ids must not be empty — provide at least one sense or BE id."

    try:
        oids = [ObjectId(sid) for sid in source_ids]
    except Exception:
        return "All source_ids must be 24-char hex ObjectId strings."

    docs = list(_coll.find({"_id": {"$in": oids}}, {"vector": 1, "level": 1, "doc_type": 1}))
    found = {d["_id"]: d for d in docs}
    missing = [str(o) for o in oids if o not in found]
    if missing:
        return f"No documents found for ids: {missing}."

    vecs = []
    for oid in oids:
        d = found[oid]
        if not d.get("vector"):
            return f"Document {d['_id']} has no vector."
        vecs.append(unpack_vec(d["vector"]))

    from lib.bary_vec import normalize
    word_vec = normalize(np.sum(vecs, axis=0)).tolist()  # normalized centroid — matches s05 formula

    ts = datetime.now(timezone.utc)
    doc: dict[str, Any] = {
        "doc_type": "node",       # always 'node'
        "node_type": "word",      # L14 = word (aggregates its senses); L15 = sense
        "level": 14,              # word level — above senses (L15), below L14 BEs
        "label": f"{word} ({pos})",  # no sense_idx — word nodes are not indexed by sense
        "vector": word_vec,       # normalized centroid of source_ids vectors (s05 formula)
        "surface": len(source_ids),  # number of sense/BE sources; proxy for polysemy breadth
        "rotation": 0.0,          # reserved for future orientation encoding
        "parent_edge_id": None,   # orphan until paired by create_edge
        "properties": {
            "word": word,
            "pos": pos,
            "etymology": etymology,  # optional; empty string if not provided
            "forms": [],             # kaikki inflected forms; empty for manually created words
            "ipa": ipa,              # optional pronunciation string
            "sense_ids": [],         # kaikki stable sense ids; empty for manually created words
            "relations": [],         # kaikki word-level relations; empty for manually created words
        },
        "created_at": ts,
        "updated_at": ts,
    }
    result = _coll.insert_one(doc)
    return _fmt({"ok": True, "id": str(result.inserted_id), "label": doc["label"],
                 "sources_used": len(vecs)})


@_write_tool
async def create_edge(
    cm1_id: str,
    cm2_id: str,
    edge_type: str | None = None,
    q: float = 0.0,
) -> str:
    """Create a new BaryEdge between two same-level nodes, exactly as the pipeline would.

    edge_type shapes the relational flavor of the BE vector via TYPE_SENTENCES embedding:
      - same_phenomenon  — these two words describe the same concept  (q≈0.90)
      - contradicts      — these two words have opposite meanings      (q≈0.85)
      - extends          — one word is derived from or extends other   (q≈0.60)
      - applies_to       — these two words share a common origin      (q≈0.55)
      - is_instance_of   — specific instance of a broader relation    (q≈0.65)

    Omit edge_type for L15 sense-to-sense BEs — bary_vec collapses to normalize(v1+v2).
    q defaults to the pipeline q_seed for the edge_type, or 1.0 when edge_type is None.
    Returns the new edge's _id.
    """
    return await _run_thr(
        _create_edge_body, cm1_id, cm2_id, edge_type, q
    )


def _create_edge_body(
    cm1_id: str, cm2_id: str, edge_type: str | None, q: float,
) -> str:
    if edge_type is not None and edge_type not in TYPE_SENTENCES:
        return (f"edge_type must be one of: {', '.join(TYPE_SENTENCES)}"
                " — or omit entirely for a type-neutral L15 BE.")

    try:
        oid1 = ObjectId(cm1_id)
        oid2 = ObjectId(cm2_id)
    except Exception:
        return "cm1_id and cm2_id must be 24-char hex ObjectId strings."

    cm1 = _coll.find_one({"_id": oid1}, {"vector": 1, "level": 1})
    cm2 = _coll.find_one({"_id": oid2}, {"vector": 1, "level": 1})
    if not cm1:
        return f"No document with cm1_id {cm1_id}."
    if not cm2:
        return f"No document with cm2_id {cm2_id}."
    if not cm1.get("vector") or not cm2.get("vector"):
        return "Both CMs must have a stored vector."
    if cm1.get("level") != cm2.get("level"):
        return f"Both CMs must be at the same level (got {cm1.get('level')} vs {cm2.get('level')})."

    v1 = unpack_vec(cm1["vector"])
    v2 = unpack_vec(cm2["vector"])

    if edge_type is not None:
        if not q:
            q = _settings.q_seeds.get(_EDGE_TYPE_Q_KEY[edge_type], 0.70)
        try:
            embedder = get_embedder(_settings)
            type_vec = embedder.embed([TYPE_SENTENCES[edge_type]])[0]
        except Exception as e:
            return f"Embedding failed — is Ollama running at {_settings.ollama_url}?\nError: {e}"
        bv = compute_bary_vec(v1, v2, type_vec, q)
    else:
        # L15 sense-to-sense BE: no relational type, pure CM centroid
        q = q if q else 1.0
        type_vec = None
        bv = _norm_vec(v1 + v2)

    doc = _make_baryedge(
        oid1, oid2,
        level=cm1.get("level", 14),  # inherit level from cm1; both CMs are same level
        vector=bv,                   # bary_vec: normalize(q·v1 + q·v2 + (1−q)·v_type)
        q=q,                         # connection_strength and base accumulated_weight
        edge_type=edge_type,         # None for L15; kaikki relation type for L14
        type_vector=type_vec,        # None for L15; embed(TYPE_SENTENCES[edge_type]) for L14
        source="ingested",           # matches pipeline — no staging distinction
        confidence=1.0,
    )
    result = _coll.insert_one(doc)
    return _fmt({
        "ok": True,
        "id": str(result.inserted_id),
        "edge_type": edge_type,
        "level": doc["level"],
        "q": q,
        "cm1_id": cm1_id,
        "cm2_id": cm2_id,
    })


@_write_tool
async def create_structure_meta_bary(
    cm1_id: str, cm2_id: str, bridge_id: str, author: str = ""
) -> str:
    """Create a Structure MetaBary (SMB) triad — cross-cutting, non-exclusive grouping.

    SMB follows the same vector and level rules as a pipeline MetaBary (s08)
    but relaxes the unique-parent constraint: cm1, cm2, and bridge may already
    be parented nodes. The SMB does not take hierarchical ownership of its
    children — parent_edge_id on the children is never modified.

    Distinguishable from pipeline MBs via source='structural' and the explicit
    bridge_id field (needed because the bridge is not re-parented, so the
    standard parent_edge_id reverse-lookup used by _triad_of cannot find it).

    Level rules (same as pipeline):
      - cm1 and cm2 must be at the same level
      - bridge must be at child_level - 1
      - SMB is inserted at child_level - 2

    author: optional provenance signature stored verbatim on the document.
      Convention: humans sign a nickname (e.g. "adseipsum"); models sign
      model name with version (e.g. "big-pickle@opencode-0.5"). Omitted or
      empty → the field is not stored and the SMB counts as anonymous
      testimony ({author: {$exists: false}} finds those).

    Returns the new SMB's _id, level, q_mb_raw, accumulated_weight,
    and the cosine between the two children (pipeline threshold is 0.90).
    """
    return await _run_thr(
        _create_structure_meta_bary_body, cm1_id, cm2_id, bridge_id, author
    )


def _create_structure_meta_bary_body(
    cm1_id: str, cm2_id: str, bridge_id: str, author: str = "",
) -> str:
    if len({cm1_id, cm2_id, bridge_id}) != 3:
        return "cm1_id, cm2_id, and bridge_id must all be distinct."
    try:
        oid1 = ObjectId(cm1_id)
        oid2 = ObjectId(cm2_id)
        oid3 = ObjectId(bridge_id)
    except Exception:
        return "cm1_id, cm2_id, bridge_id must be 24-char hex ObjectId strings."

    fields = {"vector": 1, "level": 1, "accumulated_weight": 1, "parent_edge_id": 1}
    cm1    = _coll.find_one({"_id": oid1}, fields)
    cm2    = _coll.find_one({"_id": oid2}, fields)
    bridge = _coll.find_one({"_id": oid3}, fields)

    if not cm1:
        return f"No document with cm1_id {cm1_id}."
    if not cm2:
        return f"No document with cm2_id {cm2_id}."
    if not bridge:
        return f"No document with bridge_id {bridge_id}."

    for label, doc in (("cm1", cm1), ("cm2", cm2), ("bridge", bridge)):
        if not doc.get("vector"):
            return f"{label} ({doc['_id']}) has no vector."

    # Level invariants — identical to pipeline
    child_level = cm1.get("level")
    if cm2.get("level") != child_level:
        return f"cm1 and cm2 must be the same level (got {child_level} vs {cm2.get('level')})."
    if bridge.get("level") != child_level - 1:
        return (
            f"bridge must be at child_level - 1 = {child_level - 1} "
            f"(got {bridge.get('level')})."
        )
    mb_level = child_level - 2
    if mb_level < 1:
        return f"child_level {child_level} would produce SMB at level {mb_level} — minimum is 1."

    v1 = unpack_vec(cm1["vector"])
    v2 = unpack_vec(cm2["vector"])
    vb = unpack_vec(bridge["vector"])
    w1 = float(cm1.get("accumulated_weight", 1.0))
    w2 = float(cm2.get("accumulated_weight", 1.0))
    w3 = float(bridge.get("accumulated_weight", 1.0))

    child_cosine = round(cosine(v1, v2), 4)  # pipeline threshold is 0.90; reported, not enforced

    vec, q_mb_raw = compute_metabary_vec(v1, v2, vb, w1, w2, w3)
    acc_w = q_mb_raw * level_factor(mb_level, _settings.level_factor_alpha)

    doc = _make_metabary(
        oid1, oid2,
        level=mb_level,           # child_level - 2; same rule as pipeline MB
        vector=vec,               # normalize(w1·v1 + w2·v2 + w3·v_bridge)
        q_mb_raw=q_mb_raw,  # Born rule: w3² / √(w1⁴+w2⁴+w3⁴); stored as connection_strength
        accumulated_weight=acc_w,  # q_mb_raw × level_factor; available for upward propagation
    )
    # SMB-specific fields layered on top of the standard metabary schema
    doc["source"] = "structural"  # distinguishes SMB from pipeline MBs ({source: 'structural'})
    doc["bridge_id"] = oid3  # explicit bridge ref — bridge not re-parented so reverse-lookup fails

    signature = author.strip()
    if signature:
        if len(signature) > 200:
            return "author must be at most 200 characters."
        doc["author"] = signature

    result = _coll.insert_one(doc)
    return _fmt({
        "ok": True,
        "id": str(result.inserted_id),
        "level": mb_level,
        "q_mb_raw": round(q_mb_raw, 6),
        "accumulated_weight": round(acc_w, 6),
        "child_cosine": child_cosine,
        "cm1_parented": cm1.get("parent_edge_id") is not None,
        "cm2_parented": cm2.get("parent_edge_id") is not None,
        "bridge_parented": bridge.get("parent_edge_id") is not None,
        "author": doc.get("author"),
        "cm1_id": cm1_id,
        "cm2_id": cm2_id,
        "bridge_id": bridge_id,
    })


def _validate_assoc_config(
    seed_top_k: int = 20,
    bridge_top_k: int = 12,
    result_top_k: int = 5,
    max_hops: int = 4,
    target_levels: list[int] | None = None,
    min_convergence: int = 1,
    beam_decay: float = 0.75,
    novelty_weight: float = 0.2,
    convergence_weight: float = 0.4,
    q_weight_leaf: float = 0.5,
    q_weight_high: float = 0.05,
    return_paths: bool = True,
    include_dois: bool = False,
) -> AssocConfig:
    """Build + validate an AssocConfig from tool params (shared by both tools)."""
    kwargs = dict(
        seed_top_k=seed_top_k, bridge_top_k=bridge_top_k,
        result_top_k=result_top_k, max_hops=max_hops,
        target_levels=tuple(target_levels or (12, 11, 10)),
        min_convergence=min_convergence, beam_decay=beam_decay,
        novelty_weight=novelty_weight, convergence_weight=convergence_weight,
        q_weight_leaf=q_weight_leaf, q_weight_high=q_weight_high,
        return_paths=return_paths, include_dois=include_dois,
    )
    return _config_from_kwargs(**kwargs)


@_assoc_tool
async def associative_search(
    query: str,
    seed_top_k: int = 20,
    bridge_top_k: int = 12,
    result_top_k: int = 5,
    max_hops: int = 4,
    target_levels: list[int] | None = None,
    min_convergence: int = 1,
    beam_decay: float = 0.75,
    novelty_weight: float = 0.2,
    convergence_weight: float = 0.4,
    q_weight_leaf: float = 0.5,
    q_weight_high: float = 0.05,
    return_paths: bool = True,
    include_dois: bool = False,
) -> str:
    """Associative search: search THROUGH the graph, answer WITH destinations.

    Given a cue, run bounded multi-branch propagation upward from resolved
    seeds, preserve strong / divergent / convergent-low-weight paths, accumulate
    high-level MetaBary coordinates, and return the strongest results with the
    support paths that produced them — instead of forcing the LLM to inspect
    the whole activated area.

    Parameters (default → what it controls):
      query (required)      The cue; embedded and used as the seed vector.
      seed_top_k = 20       Breadth at seed resolution (word/sense nodes + L15/L14
                            BaryEdges, split roughly half/half). Cap 200.
      bridge_top_k = 12     Beam width: how many paths may continue per hop,
                            split across three channels (strong / divergent /
                            convergent-low). Higher = wider recall, slower. Cap 100.
      result_top_k = 5      Number of final high-level associative coordinates
                            returned. Cap 30.
      max_hops = 4          Maximum upward steps from the seeds (allowed 1-8).
                            The graph's structure may exhaust earlier;
                            deepest reach is reported as highest_reached_level.
      target_levels=[12,11,10]  MetaBary levels to report (each 1-13; empty list =
                            any level). Levels with no coordinate reached are skipped.
      min_convergence = 1   Minimum number of independent support paths a
                            coordinate needs to be returned (a convergence filter).
      beam_decay = 0.75     Per-hop energy multiplier applied to q along a path
                            (path energy E).
      convergence_weight = 0.4  Weight of convergence C in the high-level score.
      novelty_weight = 0.2  Weight of novelty N (how unlike the direct
                            neighbourhood the coordinate is).
      q_weight_leaf = 0.5   Weight of edge q in the LEAF-level seed score:
                            S_leaf = 0.3*R + q_weight_leaf*q + 0.2*E.
      q_weight_high = 0.05  Weight of path energy in the HIGH-level score:
                            S = alpha*R + convergence_weight*C + novelty_weight*N
                            + q_weight_high*E, where alpha = 1 - the other weights.
      return_paths = true   Include compact support paths + path_steps per result.
      include_dois = false  Attach academic-batch source DOIs to each hit
                            (extra reverse-index lookup; skip unless you need
                            provenance).

    Returns status 'ok' with results, 'no_seed' (no graph seed), or
    'no_target' (seeds found but none reached the requested target levels).
    Each result includes the coordinate words, score/convergence/novelty,
    the triggering leaf structure, compact support paths, and a 'why' tree
    compressing the route. Summary text is included for direct LLM display.
    """
    try:
        config = _validate_assoc_config(
            seed_top_k=seed_top_k, bridge_top_k=bridge_top_k,
            result_top_k=result_top_k, max_hops=max_hops,
            target_levels=target_levels, min_convergence=min_convergence,
            beam_decay=beam_decay, novelty_weight=novelty_weight,
            convergence_weight=convergence_weight, q_weight_leaf=q_weight_leaf,
            q_weight_high=q_weight_high, return_paths=return_paths,
            include_dois=include_dois,
        )
    except ValueError as e:
        return _fmt({"status": "error", "query": query, "message": str(e)})

    payload = await _run_thr(
        lambda: run_search(
            _coll, get_embedder(_settings), query, config,
            bridge_coll=(_bridge_coll if config.include_dois else None),
        )
    )
    return _fmt(payload)


@_assoc_tool
async def associative_progressive(
    session_id: str,
    stage: str,
    query: str = "",
    selected_ids: list[str] | None = None,
    seed_top_k: int = 20,
    bridge_top_k: int = 12,
    result_top_k: int = 8,
    max_hops: int = 4,
    target_levels: list[int] | None = None,
    min_convergence: int = 1,
    beam_decay: float = 0.75,
    novelty_weight: float = 0.2,
    convergence_weight: float = 0.4,
    q_weight_leaf: float = 0.5,
    q_weight_high: float = 0.05,
) -> str:
    """Session-based progressive associative search: discover -> expand -> compare -> propose.

    Lets a workflow guide the search interactively instead of taking one
    shot blindly:

      discover  — run the full search for `query`; returns the activated area:
                  every ranked high-level coordinate (compact) so the caller
                  can pick which deserve deeper inspection.
      expand    — for selected_ids: triad structure (child1/child2/bridge word
                  sets), strongest leaf support, competing siblings, and
                  alternate cross-links for each chosen coordinate.
      compare   — for selected_ids (>=2): pairwise cosine, word overlap,
                  shared root origins, and shared triggering edge.
      propose   — for selected_ids (>=2, same level): an SMB-ready proposal
                  packet (cm1/cm2 at child_level, a level child_level-1 bridge
                  candidate via centroid vector search, expected child cosine,
                  convergence stats, academic provenance, prior structural SMBs
                  touching the region). The packet is NOT created — call
                  create_structure_meta_bary with (cm1_id, cm2_id, bridge_id).

    Sessions are held in memory (process-local, LRU-capped at 32); they are
    lost if the server restarts. Re-running discover on an existing session_id
    replaces it. Query is required for stage=discover only.

    Parameters (default → what it controls; same scoring/search knobs as
    associative_search — see its description for meanings):
      session_id (required)  Names the in-memory session; must be re-used
                             across stage calls. First call per session MUST
                             be discover (others error "no session").
      stage (required)       discover | expand | compare | propose (above).
      query = ""             The cue — required for discover only.
      selected_ids = null    Coordinate ids from the previous discover to
                             drill into (expand: >=1, compare/propose: >=2).
      result_top_k = 8       Coordinates kept from discover for later stages
                             (note: default 8 here, not 5).
      seed_top_k = 20, bridge_top_k = 12, max_hops = 4,
      target_levels=[12,11,10], min_convergence = 1, beam_decay = 0.75,
      convergence_weight = 0.4, novelty_weight = 0.2, q_weight_leaf = 0.5,
      q_weight_high = 0.05   Identical defaults and meaning to associative_search.
    """
    try:
        config = _validate_assoc_config(
            seed_top_k=seed_top_k, bridge_top_k=bridge_top_k,
            result_top_k=result_top_k, max_hops=max_hops,
            target_levels=target_levels, min_convergence=min_convergence,
            beam_decay=beam_decay, novelty_weight=novelty_weight,
            convergence_weight=convergence_weight, q_weight_leaf=q_weight_leaf,
            q_weight_high=q_weight_high,
        )
    except ValueError as e:
        return _fmt({"status": "error", "session_id": session_id,
                     "stage": stage, "message": str(e)})

    payload = await _run_thr(
        lambda: progressive(
            _coll, get_embedder(_settings), session_id, stage, query=query,
            selected_ids=selected_ids, config=config, bridge_coll=_bridge_coll,
        )
    )
    return _fmt(payload)


def _warmup_engine() -> None:
    """Best-effort boot warm-up for the cold-start stall.

    The very first Mongo query / $vectorSearch in a fresh server process
    can take 1-2 minutes (topology discovery + HNSW index pages paged in on
    a 6.7M-doc collection). Fire one cheap read and one small vector search
    from a daemon thread at startup so a real tool call never pays that
    cost. Never raises; failures are logged and non-fatal.
    """
    import time

    try:
        t0 = time.time()
        _coll.find_one({}, {"_id": 1})
        _log.info("warmup: find_one took %.1fs", time.time() - t0)
    except Exception as e:  # pragma: no cover - env-dependent
        _log.warning("warmup find_one failed: %s", e)

    try:
        import numpy as np

        t0 = time.time()
        rng = np.random.default_rng(0)
        v = rng.normal(size=_settings.embed_dim)
        v = (v / np.linalg.norm(v)).astype(np.float32).tolist()
        hits = vector_search(
            _coll, v, limit=3, num_candidates=50, filter={"doc_type": "node"}
        )
        _log.info("warmup: $vectorSearch got %d hits in %.1fs", len(hits), time.time() - t0)
    except Exception as e:  # pragma: no cover - env-dependent
        _log.warning("warmup $vectorSearch failed: %s", e)

    try:
        t0 = time.time()
        ensure_indexes(_coll)
        _log.info("warmup: ensure_indexes done in %.1fs", time.time() - t0)
    except Exception as e:  # pragma: no cover - env-dependent
        _log.warning("warmup ensure_indexes failed: %s", e)


if __name__ == "__main__":
    import argparse
    import threading

    threading.Thread(target=_warmup_engine, daemon=True, name="mcp-warmup").start()

    parser = argparse.ArgumentParser(description="BaryGraph MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="stdio (default, for Claude Code/Desktop), sse (legacy HTTP clients), "
        "or streamable-http (for Claude remote connector)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    args = parser.parse_args()

    if _public_only:
        mode = "PUBLIC (read-only, 9 tools)"
    elif _read_only:
        mode = "READ_ONLY (11 tools, no writes)"
    else:
        mode = "PRIVATE (full 17 tools)"
    _log.info("mode: %s", mode)

    if args.transport == "sse":
        import uvicorn
        uvicorn.run(mcp.sse_app(), host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        import uvicorn
        uvicorn.run(mcp.streamable_http_app(), host=args.host, port=args.port)
    else:
        mcp.run()
