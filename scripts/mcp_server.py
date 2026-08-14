"""BaryGraph MCP server — exposes the barygraph collection as Claude tools.

Provides fifteen tools:
  context_search   — MAIN entry point: $vectorSearch + full leaf content +
                      full ancestor chain to root, merged into a single call
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
  scan_unlabeled   — paginated MB scan for docs missing semantic_label
  set_label        — write semantic_label + label_vector back to an MB

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
import os
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pymongo.errors import OperationFailure

import numpy as np

from lib.bary_vec import TYPE_SENTENCES, compute_bary_vec, compute_metabary_vec, cosine, level_factor, normalize as _norm_vec
from lib.config import Settings
from lib.db import cm_leaf_words, get_collection, vector_search
from lib.docs import baryedge as _make_baryedge
from lib.docs import metabary as _make_metabary
from lib.embed import get_embedder
from lib.log import setup_logging

# -- helpers ------------------------------------------------------------------


def _leaf_nodes(be_id: Any) -> dict[str, list]:
    """BFS from a BE/MB down to all reachable L14/L15 node docs.

    Returns {"senses": [...], "words": [...]} where senses are L15 sense nodes
    (gloss, tags, topics) and words are L14 word nodes (pos, ipa, etymology).
    """
    frontier: set[Any] = {be_id}
    visited: set[Any] = set()
    senses: list[dict] = []
    words: list[dict] = []

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
        frontier = next_frontier

    return {"senses": senses, "words": words}

_settings = Settings.load()
setup_logging(_settings.log_level)
_coll = get_collection(_settings)

_public_host = os.environ.get("MCP_PUBLIC_HOST", "")
_allowed_hosts = ["localhost:*", "127.0.0.1:*"]
_allowed_origins = ["http://localhost:*", "http://127.0.0.1:*"]
if _public_host:
    _allowed_hosts.append(_public_host)
    _allowed_origins.append(f"https://{_public_host}")

mcp = FastMCP(
    "barygraph",
    instructions=(
        "BaryGraph knowledge graph built from the kaikki.org English dictionary. "
        "L14 nodes are words (node_type='word'), L15 nodes are individual senses "
        "(node_type='sense'). L15 BaryEdges pair sense nodes; L14 BaryEdges connect "
        "word nodes via kaikki relations (synonyms, antonyms, hypernyms…). "
        "L13–L10 are MetaBary triads: each MB has two children (cm1, cm2) and a "
        "bridge. "
        "START WITH context_search for any meaning/concept lookup — it runs "
        "$vectorSearch and returns each hit already expanded with its full leaf "
        "content (senses/words) and its full ancestor chain to the root, so no "
        "follow-up leaf_nodes or traverse_up call is needed. Reach for the other "
        "tools only when you need something context_search doesn't cover: "
        "sample_metabary to browse randomly, edge_info for a specific id, "
        "leaf_nodes for a raw full-subtree dump on a specific id, find_word/"
        "word_senses/word_edges for exact-string lookups rather than semantic ones."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    ),
)


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


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
def word_senses(word: str) -> str:
    """List all L15 sense nodes for a word — glosses, tags, and whether each sense is paired."""
    docs = list(_coll.find(
        {"doc_type": "node", "node_type": "sense", "properties.word": word},
        {"properties.sense_idx": 1, "properties.pos": 1, "properties.gloss": 1,
         "properties.tags": 1, "properties.topics": 1, "parent_edge_id": 1},
    ).sort("properties.sense_idx", 1))
    if not docs:
        return f"No senses found for '{word}' (word may not be in the graph)."
    return _fmt([
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
    ])


@mcp.tool()
def word_edges(word: str, pos: str = "") -> str:
    """Get L14 BaryEdges where this word is a CM (direct kaikki relations).

    Optionally filter by POS (noun, verb, adj, …).
    Returns edge_type, partner word, q, and accumulated_weight.
    """
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
def edge_info(edge_id: str) -> str:
    """Get full details about a BaryEdge or MetaBary by id.

    For L14/L15 BaryEdges: shows edge_type, q, and flat CM leaf words.
    For MetaBary (L13–L10): shows the triad structure — child1, child2, and
    bridge — each with their own word sets, so you can see what concepts
    each branch represents and how they are connected.
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

    return _fmt(result)


@mcp.tool()
def traverse_up(edge_id: str, max_levels: int = 6) -> str:
    """Walk the parent_edge_id chain upward from any BE or MB.

    Returns the ancestry chain from the starting edge to the root (or until
    max_levels is reached / parent is null). For MetaBary levels (≤13) each
    step includes the full triad structure (child1, child2, bridge with their
    word sets); for L14/L15 BaryEdges shows flat leaf words and edge_type.
    """
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
            step["triad"] = _triad_of(doc["_id"], doc["cm1_id"], doc["cm2_id"], doc.get("bridge_id"))
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
def sample_metabary(level: int, n: int = 5, with_parent: bool = True) -> str:
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
                    "triad": _triad_of(pdoc["_id"], pdoc["cm1_id"], pdoc["cm2_id"], pdoc.get("bridge_id")),
                }
        results.append(entry)
    return _fmt(results)


@mcp.tool()
def leaf_nodes(edge_id: str) -> str:
    """Get all L15 sense nodes and L14 word nodes reachable from a BE or MB.

    Traverses the full CM lineage downward to leaf nodes and returns the
    actual documents with semantic content — not just word strings:
    - senses: L15 sense nodes with gloss, tags, and topics
    - words:  L14 word nodes with pos, ipa, and etymology snippet

    Use this to build full search context for a MB or BE found via
    semantic_search or sample_metabary — see every sense and word the
    edge encodes.
    """
    try:
        oid = ObjectId(edge_id)
    except Exception:
        return f"Invalid edge_id '{edge_id}' — must be a 24-char hex ObjectId string."
    if not _coll.find_one({"_id": oid}, {"_id": 1}):
        return f"No document with id {edge_id}."
    result = _leaf_nodes(oid)
    result["edge_id"] = edge_id
    result["sense_count"] = len(result["senses"])
    result["word_count"] = len(result["words"])
    return _fmt(result)


@mcp.tool()
def semantic_search(query: str, doc_type: str = "baryedge", top_k: int = 10) -> str:
    """Semantic similarity search against the BaryGraph vector index (mongot).

    doc_type: 'baryedge' searches relationship vectors (default);
              'node' searches word/sense vectors.
    Requires s10_index to have completed. The HNSW index may take several
    minutes to build after creation.
    """
    try:
        embedder = get_embedder(_settings)
        qv = embedder.embed([query])[0].tolist()
    except Exception as e:
        return f"Embedding failed — is Ollama running at {_settings.ollama_url}?\nError: {e}"

    try:
        docs = vector_search(
            _coll, qv,
            limit=top_k,
            num_candidates=max(top_k * 10, 200),
            filter={"doc_type": doc_type},
        )
    except OperationFailure as e:
        return (
            "Vector search unavailable — the mongot index may still be building.\n"
            f"Error: {e}"
        )

    if not docs:
        return "No results returned. Index may still be building or corpus is empty."

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
            {"id": str(s["_id"]), "pos": s["properties"].get("pos"), "gloss": s["properties"].get("gloss", "")}
            for s in sense_docs
        ]
    return ctx


def _edge_context(doc: dict[str, Any], max_leaves: int) -> dict[str, Any]:
    """Expand a BE/MB hit with its triad (if MB) and full leaf senses/words."""
    level = doc.get("level")
    ctx: dict[str, Any] = {"level": level, "connection_strength": doc.get("connection_strength")}
    leaves = _leaf_nodes(doc["_id"])
    if level is not None and level <= 13:
        ctx["semantic_label"] = doc.get("semantic_label")
        ctx["triad"] = _triad_of(doc["_id"], doc["cm1_id"], doc["cm2_id"], doc.get("bridge_id"))
    else:
        ctx["edge_type"] = doc.get("edge_type")
        ctx["q"] = doc.get("q")
    ctx["senses"] = leaves["senses"][:max_leaves]
    ctx["words"] = leaves["words"][:max_leaves]
    ctx["sense_count"] = len(leaves["senses"])
    ctx["word_count"] = len(leaves["words"])
    return ctx


def _ancestry_chain(parent_id: Any, max_levels: int = 20) -> list[dict[str, Any]]:
    """Walk parent_edge_id all the way to the root — unlike leaf traversal this is a
    single linear chain (one parent per doc), never wide, so going the full distance
    is cheap. Branches off that chain are what leaf_nodes/context_search on a
    specific id are for."""
    chain: list[dict[str, Any]] = []
    current_id = parent_id
    for _ in range(max_levels):
        pdoc = _coll.find_one(
            {"_id": current_id},
            {"level": 1, "edge_type": 1, "semantic_label": 1, "parent_edge_id": 1,
             "cm1_id": 1, "cm2_id": 1, "bridge_id": 1, "connection_strength": 1},
        )
        if not pdoc:
            break
        step: dict[str, Any] = {
            "id": str(pdoc["_id"]),
            "level": pdoc.get("level"),
            "connection_strength": pdoc.get("connection_strength"),
        }
        if pdoc.get("level") is not None and pdoc["level"] <= 13:
            step["semantic_label"] = pdoc.get("semantic_label")
            step["triad_words"] = _triad_of(pdoc["_id"], pdoc["cm1_id"], pdoc["cm2_id"], pdoc.get("bridge_id"))
        else:
            step["edge_type"] = pdoc.get("edge_type")
            step["leaf_words"] = sorted(cm_leaf_words(_coll, pdoc["_id"]))
        chain.append(step)
        next_id = pdoc.get("parent_edge_id")
        if not next_id:
            break
        current_id = next_id
    return chain


@mcp.tool()
def context_search(
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
      - MetaBary (L10-L13) → semantic_label + triad (child1/child2/bridge word
        sets) + every sense/word reachable underneath (leaf_nodes)
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
      from each hit up to the root — semantic_label/triad for MB ancestors,
      edge_type/leaf_words for BE ancestors — so you see everything the hit
      belongs to, all the way up, without a separate traverse_up call. This
      is cheap because parent_edge_id is a single linear chain (one parent
      per doc), unlike the fan-out you get traversing down — so there's no
      partial "one level" option; you get the whole chain or none. If you
      instead need to explore sideways/downward from a specific ancestor id,
      call context_search or leaf_nodes on that id directly. Set False to
      skip the chain for a faster, terser response.
    """
    if doc_type not in ("node", "baryedge", "any"):
        return "doc_type must be 'node', 'baryedge', or 'any'."
    top_k = min(max(top_k, 1), 20)
    max_leaves = min(max(max_leaves, 1), 200)

    try:
        embedder = get_embedder(_settings)
        qv = embedder.embed([query])[0].tolist()
    except Exception as e:
        return f"Embedding failed — is Ollama running at {_settings.ollama_url}?\nError: {e}"

    filt = {"doc_type": doc_type} if doc_type in ("node", "baryedge") else None
    try:
        docs = vector_search(
            _coll, qv,
            limit=top_k,
            num_candidates=max(top_k * 10, 200),
            filter=filt,
        )
    except OperationFailure as e:
        return (
            "Vector search unavailable — the mongot index may still be building.\n"
            f"Error: {e}"
        )
    if not docs:
        return "No results returned. Index may still be building or corpus is empty."

    results = []
    for d in docs:
        r: dict[str, Any] = {
            "id": str(d["_id"]),
            "score": round(float(d.get("_score", 0)), 4),
            "doc_type": d.get("doc_type"),
        }
        if d["doc_type"] == "node":
            r.update(_node_context(d, max_leaves))
        else:
            r.update(_edge_context(d, max_leaves))

        if include_ancestry and d.get("parent_edge_id"):
            r["ancestor_chain"] = _ancestry_chain(d["parent_edge_id"])

        results.append(r)

    return _fmt(results)


# q_seed lookup: edge_type → q_seeds key (same_phenomenon maps to synonyms tier)
_EDGE_TYPE_Q_KEY: dict[str, str] = {
    "contradicts":     "contradicts",
    "applies_to":      "applies_to",
    "is_instance_of":  "is_instance_of",
    "extends":         "extends",
    "same_phenomenon": "synonyms",
}


@mcp.tool()
def create_sense(
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


@mcp.tool()
def create_word(
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
        vecs.append(np.asarray(d["vector"], dtype=np.float32))

    from lib.bary_vec import normalize
    word_vec = normalize(sum(vecs)).tolist()  # normalized centroid — matches s05 formula

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


@mcp.tool()
def create_edge(
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

    v1 = np.asarray(cm1["vector"], dtype=np.float32)
    v2 = np.asarray(cm2["vector"], dtype=np.float32)

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


@mcp.tool()
def create_structure_meta_bary(cm1_id: str, cm2_id: str, bridge_id: str) -> str:
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

    Returns the new SMB's _id, level, q_mb_raw, accumulated_weight,
    and the cosine between the two children (pipeline threshold is 0.90).
    """
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

    v1 = np.asarray(cm1["vector"],    dtype=np.float32)
    v2 = np.asarray(cm2["vector"],    dtype=np.float32)
    vb = np.asarray(bridge["vector"], dtype=np.float32)
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
        q_mb_raw=q_mb_raw,        # Born rule: w3² / √(w1⁴+w2⁴+w3⁴); stored as connection_strength
        accumulated_weight=acc_w, # q_mb_raw × level_factor; available for future upward propagation
    )
    # SMB-specific fields layered on top of the standard metabary schema
    doc["source"] = "structural"  # distinguishes SMB from pipeline MBs ({source: 'structural'})
    doc["bridge_id"] = oid3       # explicit bridge ref — bridge is not re-parented so reverse-lookup fails

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
        "cm1_id": cm1_id,
        "cm2_id": cm2_id,
        "bridge_id": bridge_id,
    })


@mcp.tool()
def scan_unlabeled(level: int, limit: int = 50, offset: int = 0) -> str:
    """Return MetaBary docs at the given level that have no semantic_label yet.

    Designed for batch labelling workflows: call repeatedly with increasing
    offset to page through all unlabeled MBs. Each doc includes its triad
    structure so the caller has everything needed to generate a label without
    a second round-trip.

    level:  10–13 (MetaBary range).
    limit:  docs per page, max 100 (default 50).
    offset: number of docs to skip (for pagination).

    Returns a summary header plus the list of docs:
      { "level": 13, "offset": 0, "returned": 50, "items": [...] }
    """
    if not (10 <= level <= 13):
        return "level must be between 10 and 13 (MetaBary range)."
    limit = min(max(limit, 1), 100)

    docs = list(_coll.find(
        {"doc_type": "baryedge", "level": level, "semantic_label": {"$exists": False}},
        {"cm1_id": 1, "cm2_id": 1, "connection_strength": 1,
         "accumulated_weight": 1, "parent_edge_id": 1},
    ).skip(offset).limit(limit))

    items = []
    for doc in docs:
        mb_id = doc["_id"]
        items.append({
            "id": str(mb_id),
            "connection_strength": doc.get("connection_strength"),
            "accumulated_weight": doc.get("accumulated_weight"),
            "has_parent": doc.get("parent_edge_id") is not None,
            "triad": _triad_of(mb_id, doc["cm1_id"], doc["cm2_id"], doc.get("bridge_id")),
        })

    return _fmt({
        "level": level,
        "offset": offset,
        "returned": len(items),
        "items": items,
    })


@mcp.tool()
def set_label(edge_id: str, label: str, label_vector: list[float], model: str) -> str:
    """Write a semantic label and its embedding vector back to a MetaBary document.

    edge_id:      24-char hex ObjectId of the MetaBary (level ≤ 13).
    label:        Short semantic phrase describing the MB cluster (e.g.
                  "Toxic plants that kill livestock").
    label_vector: 768-dim embedding of the label text, produced by the same
                  nomic-embed-text model used for all other vectors so it is
                  searchable via the existing $vectorSearch index.
    model:        Name of the model that generated the label (for provenance).

    Stores: semantic_label, label_vector, label_model, label_updated_at.
    Returns a confirmation with the stored values (vector omitted for brevity).
    """
    try:
        oid = ObjectId(edge_id)
    except Exception:
        return f"Invalid edge_id '{edge_id}' — must be a 24-char hex ObjectId string."

    doc = _coll.find_one({"_id": oid}, {"doc_type": 1, "level": 1})
    if not doc:
        return f"No document with id {edge_id}."
    if doc.get("doc_type") != "baryedge":
        return f"Document {edge_id} is not a baryedge (doc_type={doc.get('doc_type')!r})."
    if (doc.get("level") or 99) > 13:
        return f"Document {edge_id} is at level {doc.get('level')} — set_label is for MetaBary (level ≤ 13) only."
    if len(label_vector) != 768:
        return f"label_vector must be 768-dimensional, got {len(label_vector)}."

    now = datetime.now(timezone.utc)
    _coll.update_one(
        {"_id": oid},
        {"$set": {
            "semantic_label": label,
            "label_vector": label_vector,
            "label_model": model,
            "label_updated_at": now,
        }},
    )
    return _fmt({
        "ok": True,
        "edge_id": edge_id,
        "semantic_label": label,
        "label_model": model,
        "label_updated_at": now.isoformat(),
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BaryGraph MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="stdio (default, for Claude Code/Desktop) or sse (for mobile/HTTP clients)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="SSE bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="SSE port (default: 8000)")
    args = parser.parse_args()

    if args.transport == "sse":
        import uvicorn
        uvicorn.run(mcp.sse_app(), host=args.host, port=args.port)
    else:
        mcp.run()
