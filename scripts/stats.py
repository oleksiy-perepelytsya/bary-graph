"""Print graph_stats and orphan_stats directly from MongoDB.

Usage (from repo root):
    python -m scripts.stats
"""

from __future__ import annotations

import json

from lib.config import Settings
from lib.db import get_collection


def graph_stats(coll) -> dict:
    rows = list(coll.aggregate([
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
    ]))
    return {
        "total_documents": coll.estimated_document_count(),
        "breakdown": [
            {k: v for k, v in r["_id"].items() if v is not None} | {"count": r["count"]}
            for r in rows
        ],
    }


def orphan_stats(coll) -> list:
    rows = list(coll.aggregate([
        {"$group": {
            "_id": {"doc_type": "$doc_type", "level": "$level"},
            "total":   {"$sum": 1},
            "orphans": {"$sum": {"$cond": [{"$eq": ["$parent_edge_id", None]}, 1, 0]}},
        }},
        {"$sort": {"_id.doc_type": 1, "_id.level": 1}},
    ]))
    return [
        {
            "doc_type": r["_id"]["doc_type"],
            "level": r["_id"]["level"],
            "orphans": r["orphans"],
            "total": r["total"],
            "orphan_pct": round(100 * r["orphans"] / r["total"], 1) if r["total"] else None,
        }
        for r in rows
    ]


def main() -> None:
    settings = Settings.load()
    coll = get_collection(settings)

    print("=== graph_stats ===")
    print(json.dumps(graph_stats(coll), indent=2))
    print()
    print("=== orphan_stats ===")
    print(json.dumps(orphan_stats(coll), indent=2))


if __name__ == "__main__":
    main()
