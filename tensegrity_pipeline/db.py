"""Mongo collection access + indexes for the tensegrity collection.

The generic client/collection plumbing lives in :mod:`lib.db`; this module only
supplies the tensegrity-specific index set (the PoC index list references
``properties.word`` etc., which don't apply here).
"""

from __future__ import annotations

from typing import Any

from pymongo import ASCENDING
from pymongo.collection import Collection

from lib.config import Settings
from lib.db import get_client  # re-exported for callers that need the client

VECTOR_INDEX_NAME = "tensegrity_vector"

TG_INDEXES: list[list[tuple[str, int]]] = [
    [("doc_type", ASCENDING), ("node_type", ASCENDING), ("level", ASCENDING)],
    [("cm1_id", ASCENDING)],
    [("cm2_id", ASCENDING)],
    [("parent_edge_id", ASCENDING)],
    [("doc_type", ASCENDING), ("level", ASCENDING), ("edge_type", ASCENDING)],
    [("source", ASCENDING), ("level", ASCENDING), ("doc_type", ASCENDING)],
    [("properties.doc_id", ASCENDING), ("properties.section_idx", ASCENDING)],
    [("properties.sentence_id", ASCENDING)],
    [("properties.title_id", ASCENDING)],
]


def get_collection(settings: Settings) -> Collection:
    return get_client(settings)[settings.mongo_db][settings.mongo_collection]


def ensure_indexes(coll: Collection) -> list[str]:
    names: list[str] = []
    for keys in TG_INDEXES:
        names.append(coll.create_index(keys))
    return names


def collection_names(settings: Settings) -> dict[str, Any]:
    """Small diagnostic: report which DB/collection this profile targets."""
    return {"db": settings.mongo_db, "collection": settings.mongo_collection}
