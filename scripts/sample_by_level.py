"""Sample N random MetaBary docs (L10-L13) per level and write to JSONL."""
import json
import sys
import pathlib
from datetime import datetime
from bson import ObjectId
from lib.db import get_collection

OUTPUT = pathlib.Path("data/sample_by_level.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
LEVELS = [10, 11, 12, 13]


def _serialize(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")


def _leaf_words(coll, be_id, max_depth=6):
    """Collect representative words reachable from a BE/MB."""
    frontier = {be_id}
    visited: set = set()
    words: list[str] = []
    for _ in range(max_depth):
        to_fetch = frontier - visited
        if not to_fetch:
            break
        visited |= to_fetch
        nxt: set = set()
        for doc in coll.find(
            {"_id": {"$in": list(to_fetch)}},
            {"doc_type": 1, "cm1_id": 1, "cm2_id": 1, "properties": 1},
        ):
            if doc.get("doc_type") == "node":
                w = doc.get("properties", {}).get("word")
                if w:
                    words.append(w)
            else:
                for k in ("cm1_id", "cm2_id"):
                    v = doc.get(k)
                    if v and v not in visited:
                        nxt.add(v)
        frontier = nxt
    return sorted(set(words))[:8]


def main():
    coll = get_collection()
    OUTPUT.parent.mkdir(exist_ok=True)
    total = 0
    with OUTPUT.open("w") as fh:
        for level in LEVELS:
            docs = list(coll.aggregate([
                {"$match": {"doc_type": "baryedge", "level": level}},
                {"$sample": {"size": N}},
                {"$project": {
                    "cm1_id": 1, "cm2_id": 1,
                    "connection_strength": 1,
                    "accumulated_weight": 1,
                    "parent_edge_id": 1,
                }},
            ]))
            for doc in docs:
                mid = doc["_id"]
                entry = {
                    "id": str(mid),
                    "level": level,
                    "connection_strength": doc.get("connection_strength"),
                    "accumulated_weight": doc.get("accumulated_weight"),
                    "triad": {
                        "child1": {
                            "id": str(doc["cm1_id"]),
                            "words": _leaf_words(coll, doc["cm1_id"]),
                        },
                        "child2": {
                            "id": str(doc["cm2_id"]),
                            "words": _leaf_words(coll, doc["cm2_id"]),
                        },
                    },
                    "parent": str(doc["parent_edge_id"]) if doc.get("parent_edge_id") else None,
                }
                fh.write(json.dumps(entry, default=_serialize) + "\n")
                total += 1
            print(f"  level {level}: {len(docs)} docs", file=sys.stderr)
    print(f"Wrote {total} records → {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
