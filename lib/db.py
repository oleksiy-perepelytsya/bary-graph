from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from lib.config import Settings

VECTOR_INDEX_NAME = "barygraph_vector"

# Standard (non-vector) indexes — see CLAUDE.md.
STANDARD_INDEXES: list[list[tuple[str, int]]] = [
    [("doc_type", ASCENDING), ("level", ASCENDING), ("parent_edge_id", ASCENDING)],
    [("cm1_id", ASCENDING)],
    [("cm2_id", ASCENDING)],
    [("node_type", ASCENDING)],
    [("edge_type", ASCENDING), ("level", ASCENDING)],
    [("parent_edge_id", ASCENDING)],
    [("properties.word", ASCENDING), ("properties.pos", ASCENDING)],
    [("properties.sense_id", ASCENDING)],
]


@lru_cache(maxsize=8)
def _cached_client(uri: str) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_client(settings: Settings) -> MongoClient:
    return _cached_client(settings.mongo_uri)


def get_collection(settings: Settings) -> Collection:
    return get_client(settings)[settings.mongo_db][settings.mongo_collection]


def ping(settings: Settings) -> bool:
    try:
        get_client(settings).admin.command("ping")
        return True
    except PyMongoError:
        return False


def ensure_indexes(coll: Collection) -> list[str]:
    names: list[str] = []
    for keys in STANDARD_INDEXES:
        names.append(coll.create_index(keys))
    return names


def bulk_upsert(coll: Collection, docs: Iterable[Mapping[str, Any]], key: str = "_id") -> int:
    ops = [UpdateOne({key: d[key]}, {"$set": dict(d)}, upsert=True) for d in docs]
    if not ops:
        return 0
    res = coll.bulk_write(ops, ordered=False)
    return (res.upserted_count or 0) + (res.modified_count or 0)


def vector_search(
    coll: Collection,
    query_vector: list[float],
    *,
    index: str = VECTOR_INDEX_NAME,
    limit: int = 20,
    num_candidates: int = 200,
    filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """$vectorSearch via the mongot index. Appends `_score` to each result doc.

    `filter` must only reference indexed filter fields: doc_type, level,
    edge_type, node_type. Raises OperationFailure if mongot is not running.
    """
    stage: dict[str, Any] = {
        "index": index,
        "path": "vector",
        "queryVector": query_vector,
        "numCandidates": num_candidates,
        "limit": limit,
    }
    if filter:
        stage["filter"] = filter
    return list(coll.aggregate([
        {"$vectorSearch": stage},
        {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
    ]))


def cm_leaf_words(
    coll: Collection,
    be_id: Any,
    *,
    max_depth: int = 15,
    max_words: int | None = None,
) -> set[str]:
    """Return all words reachable via the CM lineage of a single BE/MB.

    Traverses cm1_id/cm2_id chains (BFS) until reaching node docs.
    Both L15 sense nodes and L14 word nodes carry properties.word.

    ``max_words`` bounds the traversal itself (not just the returned set) —
    a BE/MB near the root of the graph can fan out to a huge fraction of the
    corpus, so callers on a latency budget (e.g. an MCP tool expanding many
    hits) should pass a cap. The returned set may be a few docs over the cap
    since the check runs once per BFS level, not per document.
    """
    frontier: set[Any] = {be_id}
    visited: set[Any] = set()
    words: set[str] = set()

    for _ in range(max_depth):
        to_fetch = frontier - visited
        if not to_fetch:
            break
        visited |= to_fetch
        next_frontier: set[Any] = set()
        for doc in coll.find(
            {"_id": {"$in": list(to_fetch)}},
            {"doc_type": 1, "cm1_id": 1, "cm2_id": 1, "properties.word": 1},
        ):
            if doc.get("doc_type") == "node":
                w = doc.get("properties", {}).get("word")
                if w:
                    words.add(w)
            else:
                next_frontier.add(doc["cm1_id"])
                next_frontier.add(doc["cm2_id"])
        if max_words is not None and len(words) >= max_words:
            # Cap hit: stop descending rather than expanding next_frontier.
            # The returned set has no truncated flag of its own — callers
            # that care must infer it from len(result) >= max_words.
            break
        frontier = next_frontier

    return words


def any_cm_has_word(
    coll: Collection,
    be_ids: list[Any],
    target_word: str,
    *,
    max_depth: int = 15,
) -> bool:
    """BFS through cm1_id/cm2_id chains from be_ids; return True if target_word found.

    Terminates as soon as target_word is found in any node's properties.word.
    Both L15 sense nodes and L14 word nodes carry properties.word. Above L14,
    CMs are BEs/MBs and traversal recurses. max_depth caps infinite loops.
    """
    frontier: set[Any] = set(be_ids)
    visited: set[Any] = set()

    for _ in range(max_depth):
        to_fetch = frontier - visited
        if not to_fetch:
            break
        visited |= to_fetch
        next_frontier: set[Any] = set()
        for doc in coll.find(
            {"_id": {"$in": list(to_fetch)}},
            {"doc_type": 1, "cm1_id": 1, "cm2_id": 1, "properties.word": 1},
        ):
            if doc.get("doc_type") == "node":
                if doc.get("properties", {}).get("word") == target_word:
                    return True
            else:
                next_frontier.add(doc["cm1_id"])
                next_frontier.add(doc["cm2_id"])
        frontier = next_frontier

    return False
