"""t10_index — create the tensegrity standard + vector search indexes.

Standard B-tree indexes are idempotent. The mongot vector index is created from
``indexes/vector_index_tensegrity.json``; if the server is Community without
mongot, the vector step logs a warning and continues (everything except
``$vectorSearch`` is still queryable).

The ``numDimensions`` in the JSON is a template: the build profile's real
``embed_dim`` (1024 for the qwen0.6 profile) wins, exactly like s10.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import orjson
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from tensegrity_pipeline import doc_bridge
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.db import ensure_indexes, get_collection

STAGE = "10_index"
VECTOR_INDEX_PATH = Path("indexes/vector_index_tensegrity.json")


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start dry_run=%s dim=%d", args.dry_run, settings.embed_dim)

    if args.dry_run:
        log.info("dry-run: would create %d standard indexes + doc_bridges index + vector index", 9)
        return

    names = ensure_indexes(coll)
    log.info("standard indexes: %s", names)

    # Sibling provenance collection (doc_bridge) — idempotent; keeps reverse
    # lookups indexed even if the collection was dropped between runs.
    bridge_names = doc_bridge.ensure_indexes(doc_bridge.get_bridge_collection(settings))
    log.info("%s indexes: %s", settings.doc_bridges_collection, bridge_names)

    n_vec = 0
    if VECTOR_INDEX_PATH.exists():
        defs = orjson.loads(VECTOR_INDEX_PATH.read_bytes())
        for d in defs:
            for field in d.get("definition", {}).get("fields", []):
                if field.get("type") == "vector":
                    field["numDimensions"] = settings.embed_dim
        try:
            existing = {ix["name"] for ix in coll.list_search_indexes()}
            for d in defs:
                if d["name"] in existing:
                    coll.drop_search_index(d["name"])
                    log.info("dropped vector search index for recreation: %s", d["name"])
            models = [
                SearchIndexModel(
                    name=d["name"], type=d.get("type", "vectorSearch"),
                    definition=d["definition"],
                )
                for d in defs
            ]
            if models:
                created = coll.create_search_indexes(models)
                n_vec = len(created)
                log.info("vector search indexes created: %s", created)
        except OperationFailure as e:
            log.warning(
                "vector index unsupported on this server (%s) — "
                "$vectorSearch queries will be unavailable", e,
            )
    else:
        log.warning("no vector index definition at %s", VECTOR_INDEX_PATH)

    cp.processed = len(names) + n_vec
    cp.total = cp.processed
    finish(cp, settings, log)


if __name__ == "__main__":
    run()
