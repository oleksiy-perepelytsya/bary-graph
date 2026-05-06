"""Create standard + vector search indexes.

Standard B-tree indexes are idempotent (``create_index`` is a no-op when
the index already exists). The mongot vector index is created from
``indexes/vector_index.json``; if the server is plain Community without
mongot, the vector step logs a warning and continues — the graph is still
queryable for everything except $vectorSearch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import orjson
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from lib.db import ensure_indexes, get_collection
from scripts._base import bootstrap, finish

STAGE = "10_index"
VECTOR_INDEX_PATH = Path("indexes/vector_index.json")


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start dry_run=%s", args.dry_run)

    if args.dry_run:
        log.info("dry-run: would create %d standard indexes + vector index", 8)
        return

    names = ensure_indexes(coll)
    log.info("standard indexes: %s", names)

    n_vec = 0
    if VECTOR_INDEX_PATH.exists():
        defs = orjson.loads(VECTOR_INDEX_PATH.read_bytes())
        try:
            existing = {ix["name"] for ix in coll.list_search_indexes()}
            # Drop existing indexes whose definition we want to update, then recreate.
            for d in defs:
                if d["name"] in existing:
                    coll.drop_search_index(d["name"])
                    log.info("dropped vector search index for recreation: %s", d["name"])
            models = [
                SearchIndexModel(name=d["name"], type=d.get("type", "vectorSearch"),
                                 definition=d["definition"])
                for d in defs
            ]
            if models:
                created = coll.create_search_indexes(models)
                n_vec = len(created)
                log.info("vector search indexes created: %s", created)
        except OperationFailure as e:
            log.warning(
                "vector index unsupported on this server (%s) — "
                "$vectorSearch queries will be unavailable", e
            )
    else:
        log.warning("no vector index definition at %s", VECTOR_INDEX_PATH)

    cp.processed = len(names) + n_vec
    cp.total = cp.processed
    finish(cp, settings, log)


if __name__ == "__main__":
    run()
