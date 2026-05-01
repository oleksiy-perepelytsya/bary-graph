"""BaryGraph MCP server — exposes the barygraph collection as Claude tools.

Provides nine tools:
  find_word        — look up word nodes (all POS variants)
  word_senses      — list all L15 sense glosses for a word
  word_edges       — L14 BaryEdges where the word is a CM
  edge_info        — details + CM structure for any BE/MB by id
  traverse_up      — walk parent_edge_id chain upward from any BE/MB
  sample_metabary  — sample random MetaBary docs at a level with triad structure
  semantic_search  — $vectorSearch (requires mongot index from s10_index)
  graph_stats      — document counts by level / type
  orphan_stats     — orphan rates per level (parent_edge_id=null counts)

Run via stdio transport (Claude Code + Claude Desktop both use stdio):
    python -m scripts.mcp_server
"""

from __future__ import annotations

import json
from typing import Any

from bson import ObjectId
from mcp.server.fastmcp import FastMCP
from pymongo.errors import OperationFailure

from lib.config import Settings
from lib.db import cm_leaf_words, get_collection, vector_search
from lib.embed import get_embedder
from lib.log import setup_logging

_settings = Settings.load()
setup_logging(_settings.log_level)
_coll = get_collection(_settings)

mcp = FastMCP(
    "barygraph",
    instructions=(
        "BaryGraph knowledge graph built from the kaikki.org English dictionary. "
        "L14 nodes are words (node_type='word'), L15 nodes are individual senses "
        "(node_type='sense'). L15 BaryEdges pair sense nodes; L14 BaryEdges connect "
        "word nodes via kaikki relations (synonyms, antonyms, hypernyms…). "
        "L13–L10 are MetaBary triads: each MB has two children (cm1, cm2) and a "
        "bridge — use sample_metabary to explore, edge_info for a specific MB's "
        "triad structure, traverse_up to see where it sits in the hierarchy. "
        "graph_stats and orphan_stats show pipeline progress."
    ),
)


def _fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _triad_of(mb_id: ObjectId, cm1_id: ObjectId, cm2_id: ObjectId) -> dict[str, Any]:
    """Fetch the bridge doc and return triad structure with leaf words for all three."""
    bridge_doc = _coll.find_one(
        {"parent_edge_id": mb_id, "_id": {"$nin": [cm1_id, cm2_id]}},
        {"_id": 1},
    )
    bridge_id = bridge_doc["_id"] if bridge_doc else None
    return {
        "child1": {"id": str(cm1_id), "words": sorted(cm_leaf_words(_coll, cm1_id))},
        "child2": {"id": str(cm2_id), "words": sorted(cm_leaf_words(_coll, cm2_id))},
        "bridge": {
            "id": str(bridge_id) if bridge_id else None,
            "words": sorted(cm_leaf_words(_coll, bridge_id)) if bridge_id else [],
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
                        {"properties.word": 1, "properties.pos": 1}):
        id_to_label[d["_id"]] = f"{d['properties']['word']} ({d['properties']['pos']})"

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
        result["triad"] = _triad_of(oid, doc["cm1_id"], doc["cm2_id"])
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
    max_levels is reached / parent is null). Each step shows its level, leaf
    words, and connection strength — useful for understanding where a specific
    relationship sits in the MetaBary hierarchy.
    """
    try:
        current_id: Any = ObjectId(edge_id)
    except Exception:
        return f"Invalid edge_id '{edge_id}'."

    chain = []
    for _ in range(max_levels):
        doc = _coll.find_one(
            {"_id": current_id},
            {"level": 1, "parent_edge_id": 1, "edge_type": 1, "connection_strength": 1},
        )
        if not doc:
            break
        chain.append({
            "id": str(doc["_id"]),
            "level": doc.get("level"),
            "edge_type": doc.get("edge_type"),
            "connection_strength": doc.get("connection_strength"),
            "leaf_words": sorted(cm_leaf_words(_coll, doc["_id"])),
        })
        parent_id = doc.get("parent_edge_id")
        if not parent_id:
            break
        current_id = parent_id

    return _fmt({"starting_id": edge_id, "chain_length": len(chain), "chain": chain})


@mcp.tool()
def sample_metabary(level: int, n: int = 10) -> str:
    """Sample N random MetaBary docs at the given level with full triad structure.

    level: 10–13 (13 = closest to individual senses, 10 = most abstract).
    n: number to sample, max 20.

    Each result shows the three constituents of the MetaBary triad:
    - child1 and child2: the two BEs/MBs being bridged
    - bridge: the BE/MB that connects them
    Each branch is explained as the set of words reachable through it, so you
    can read a MetaBary as "child1-words ↔ child2-words via bridge-words".
    """
    if not (10 <= level <= 13):
        return "level must be between 10 and 13 (MetaBary range)."
    n = min(max(n, 1), 20)

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
        results.append({
            "id": str(mb_id),
            "level": level,
            "connection_strength": doc.get("connection_strength"),
            "accumulated_weight": doc.get("accumulated_weight"),
            "has_parent": doc.get("parent_edge_id") is not None,
            "triad": _triad_of(mb_id, doc["cm1_id"], doc["cm2_id"]),
        })
    return _fmt(results)


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
                    d["_id"], d["cm1_id"], d["cm2_id"]
                )
            else:
                r["cm_words"] = sorted(cm_leaf_words(_coll, d["_id"]))
        results.append(r)

    return _fmt(results)


@mcp.tool()
def graph_stats() -> str:
    """Return document counts broken down by doc_type, level, node_type, and edge_type.

    Use this to check how much data has been ingested and what stages have run.
    """
    pipeline: list[dict[str, Any]] = [
        {"$group": {
            "_id": {
                "doc_type": "$doc_type",
                "level": "$level",
                "node_type": "$node_type",
                "edge_type": "$edge_type",
            },
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id.doc_type": 1, "_id.level": -1}},
    ]
    rows = list(_coll.aggregate(pipeline))
    return _fmt({
        "total_documents": _coll.estimated_document_count(),
        "breakdown": [
            {k: v for k, v in r["_id"].items() if v is not None} | {"count": r["count"]}
            for r in rows
        ],
    })


@mcp.tool()
def orphan_stats() -> str:
    """Return orphan rates (parent_edge_id=null) per doc_type and level.

    Orphan BEs/MBs have no MetaBary parent yet — indicates how much of the
    graph is still disconnected from the upper hierarchy. Lower orphan rates
    mean better coverage. Run after s08/s09 to gauge pipeline progress.
    """
    orphan_rows = list(_coll.aggregate([
        {"$match": {"parent_edge_id": None}},
        {"$group": {"_id": {"doc_type": "$doc_type", "level": "$level"}, "orphans": {"$sum": 1}}},
        {"$sort": {"_id.doc_type": 1, "_id.level": 1}},
    ]))
    total_rows = list(_coll.aggregate([
        {"$group": {"_id": {"doc_type": "$doc_type", "level": "$level"}, "total": {"$sum": 1}}},
    ]))
    total_map = {(r["_id"]["doc_type"], r["_id"]["level"]): r["total"] for r in total_rows}

    result = []
    for r in orphan_rows:
        dt = r["_id"]["doc_type"]
        lv = r["_id"]["level"]
        total = total_map.get((dt, lv), 0)
        result.append({
            "doc_type": dt,
            "level": lv,
            "orphans": r["orphans"],
            "total": total,
            "orphan_pct": round(100 * r["orphans"] / total, 1) if total else None,
        })
    return _fmt(result)


if __name__ == "__main__":
    mcp.run()
