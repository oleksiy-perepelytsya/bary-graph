#!/usr/bin/env python3
"""One-off: re-encode stored vectors as BSON binary-vector subtype 9.

Generic ``binData`` (subtype 0) is silently ignored by mongot — the vector
search index reports ``READY`` but indexes zero documents, so ``$vectorSearch``
returns nothing. mongot *does* ingest subtype 9 (dtype FLOAT32), which is a
2-byte metadata header plus the identical float32 payload. The rewrite is
therefore lossless and size-neutral (4098 B vs 4096 B for 1024-d).

Only the ``vector`` field is converted by default — it is the indexed path.
``type_vector`` is not in the index and stays as-is (``unpack_vec`` reads both);
pass ``--include-type-vector`` to convert it too.

Usage:
    python3 -m scripts.migrate_vec_subtype9                 # poc (.env)
    python3 -m scripts.migrate_vec_subtype9 --env build-all # .env.build-all
    python3 -m scripts.migrate_vec_subtype9 --dry-run --limit 5000
    python3 -m scripts.migrate_vec_subtype9 --after-id <hex> # resume

The rewrite is idempotent: documents already on subtype 9 are skipped.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bson.binary import Binary
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import Settings  # noqa: E402
from lib.db import get_client, get_collection  # noqa: E402
from lib.vector import VECTOR_SUBTYPE, pack_vec, unpack_vec  # noqa: E402

_log = logging.getLogger("migrate_vec_subtype9")


def _is_subtype9(v: object) -> bool:
    return isinstance(v, Binary) and v.subtype == VECTOR_SUBTYPE


def migrate(
    coll,
    *,
    fields: list[str],
    batch_size: int,
    limit: int | None,
    after_id,
    dry_run: bool,
    min_free_gb: float,
) -> tuple[int, int]:
    """Return (scanned, converted)."""
    if after_id is not None:
        try:
            from bson import ObjectId

            after_id = ObjectId(after_id)
        except Exception:  # noqa: BLE001
            _log.warning("--after-id %r is not an ObjectId; using raw value", after_id)

    scanned = converted = 0
    skipped = 0
    t0 = time.time()
    now = datetime.now(timezone.utc)

    while True:
        find = {"_id": {"$gt": after_id}} if after_id is not None else {}
        proj = {f: 1 for f in fields}
        docs = list(
            coll.find(find, proj).sort("_id", 1).limit(batch_size)
        )
        if not docs:
            break

        ops: list[UpdateOne] = []
        for d in docs:
            scanned += 1
            sets: dict[str, object] = {}
            for f in fields:
                old = d.get(f)
                if old is None or _is_subtype9(old):
                    continue
                sets[f] = pack_vec(unpack_vec(old))
            if sets:
                converted += 1
                if not dry_run:
                    sets["updated_at"] = now
                    ops.append(UpdateOne({"_id": d["_id"]}, {"$set": sets}))
            else:
                skipped += 1

        if ops:
            coll.bulk_write(ops, ordered=False)

        after_id = docs[-1]["_id"]
        if scanned % (batch_size * 5) < batch_size or limit is not None:
            free_gb = shutil.disk_usage("/").free / 1e9
            rate = scanned / max(time.time() - t0, 1e-9)
            _log.info(
                "scanned=%d converted=%d skipped=%d (%.0f docs/s) free=%.1fGB last_id=%s",
                scanned, converted, skipped, rate, free_gb, after_id,
            )
            if free_gb < min_free_gb:
                _log.error(
                    "aborting: free disk %.1f GB < --min-free-gb %.1f (resume with --after-id %s)",
                    free_gb, min_free_gb, after_id,
                )
                raise SystemExit(3)

        if limit is not None and scanned >= limit:
            break

    dt = time.time() - t0
    _log.info(
        "done: scanned=%d converted=%d skipped=%d in %.1fs (%.0f docs/s)%s",
        scanned, converted, skipped, dt, scanned / max(dt, 1e-9),
        " [DRY RUN]" if dry_run else "",
    )
    return scanned, converted


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(prog="migrate_vec_subtype9")
    p.add_argument("--env", default=None, help="profile → .env.<NAME> (default: .env)")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None, help="process at most N docs (dev)")
    p.add_argument("--after-id", default=None, help="resume after this _id")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-type-vector", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=8.0)
    a = p.parse_args()

    dotenv = None
    if a.env and a.env not in ("poc", "default"):
        dotenv = f".env.{a.env}"
    settings = Settings.load(dotenv)
    _log.info("target db=%s collection=%s profile=%s", settings.mongo_db,
              settings.mongo_collection, settings.profile)

    coll = get_collection(settings)
    fields = ["vector"] + (["type_vector"] if a.include_type_vector else [])
    _log.info("fields=%s dry_run=%s batch=%d limit=%s", fields, a.dry_run,
              a.batch_size, a.limit)

    migrate(
        coll,
        fields=fields,
        batch_size=a.batch_size,
        limit=a.limit,
        after_id=a.after_id,
        dry_run=a.dry_run,
        min_free_gb=a.min_free_gb,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _log.warning("interrupted — resume with --after-id <last logged id>")
        raise SystemExit(130)
