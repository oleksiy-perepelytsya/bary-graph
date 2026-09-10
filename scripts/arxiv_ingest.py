"""Ingest arXiv papers + OpenAI vectors into cognitive_arxiv collection.

Reads the Kaggle openai-arxiv-embeddings dataset (papers.csv + vectors.dat)
joined with arxiv-metadata-oai-snapshot.json, and bulk-inserts into Mongo
with a text index on title+abstract for BM25 search.

Resume-safe: skips arxiv_ids already present in the collection.

Usage:
    python3.11 -m scripts.arxiv_ingest \
        --csv   /storage/bary/arxiv/papers.csv \
        --vec   /storage/bary/arxiv/vectors.dat \
        --meta  /storage/bary/arxiv/arxiv-metadata-oai-snapshot.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import pymongo
from bson import Binary

log = logging.getLogger("arxiv_ingest")

COLLECTION = "cognitive_arxiv"
VECTOR_DIM = 3072
VECTOR_BYTES = VECTOR_DIM * 4  # float32
BATCH_SIZE = 1024

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017/?directConnection=true")
MONGO_DB = os.environ.get("MONGO_DB", "barygraph_poc")


def _load_csv_index(csv_path: Path) -> dict[str, int]:
    """Return {arxiv_id: row_index} from papers.csv."""
    idx: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx[row["id"]] = int(row["index"])
    log.info("csv index loaded: %d papers", len(idx))
    return idx


def _load_existing_ids(coll) -> set[str]:
    """Resume-safe: fetch arxiv_ids already in the collection."""
    ids = set()
    for doc in coll.find({}, {"arxiv_id": 1, "_id": 0}).batch_size(10_000):
        ids.add(doc["arxiv_id"])
    log.info("existing ids: %d", len(ids))
    return ids


def _open_vectors(vec_path: Path) -> mmap.mmap:
    """Memory-map vectors.dat for random access."""
    f = open(vec_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    log.info("vectors mmap opened: %d bytes", mm.size())
    return mm


def _read_vector(mm: mmap.mmap, index: int) -> bytes:
    """Read 3072 float32 values as raw bytes at the given row index."""
    offset = index * VECTOR_BYTES
    return mm[offset : offset + VECTOR_BYTES]


def _make_doc(arxiv_id: str, meta: dict, vec_bytes: bytes) -> dict:
    """Build a Mongo document from metadata + vector bytes."""
    return {
        "arxiv_id": arxiv_id,
        "title": meta.get("title", ""),
        "abstract": meta.get("abstract", ""),
        "doi": meta.get("doi"),
        "categories": meta.get("categories", ""),
        "authors": meta.get("authors", ""),
        "update_date": meta.get("update_date"),
        "vector": Binary(vec_bytes),
    }


def run(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    vec_path = Path(args.vec)
    meta_path = Path(args.meta)

    for p in (csv_path, vec_path, meta_path):
        if not p.exists():
            log.error("missing: %s", p)
            return 1

    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
    coll = client[MONGO_DB][COLLECTION]
    log.info("target: %s.%s", MONGO_DB, COLLECTION)

    csv_index = _load_csv_index(csv_path)
    existing = _load_existing_ids(coll)
    mm = _open_vectors(vec_path)

    t0 = time.monotonic()
    inserted = 0
    skipped_no_vec = 0
    skipped_existing = 0
    batch: list[dict] = []

    with open(meta_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            rec = json.loads(line)
            aid = rec.get("id", "")
            if not aid:
                continue

            # resume-safe
            if aid in existing:
                skipped_existing += 1
                continue

            # lookup vector position
            row_index = csv_index.get(aid)
            if row_index is None:
                skipped_no_vec += 1
                continue

            vec_bytes = _read_vector(mm, row_index)
            batch.append(_make_doc(aid, rec, vec_bytes))

            if len(batch) >= BATCH_SIZE:
                coll.insert_many(batch, ordered=False)
                inserted += len(batch)
                elapsed = time.monotonic() - t0
                rate = inserted / elapsed if elapsed > 0 else 0
                log.info(
                    "inserted %d (skipped_existing=%d skipped_no_vec=%d) "
                    "rate=%.0f/s elapsed=%.0fs",
                    inserted, skipped_existing, skipped_no_vec, rate, elapsed,
                )
                batch.clear()

    # flush remaining
    if batch:
        coll.insert_many(batch, ordered=False)
        inserted += len(batch)

    mm.close()

    # create text index for BM25 search (idempotent)
    log.info("creating text index on title+abstract ...")
    coll.create_index([("title", "text"), ("abstract", "text")], name="text_search")
    log.info("text index created")

    elapsed = time.monotonic() - t0
    log.info(
        "done | inserted=%d skipped_existing=%d skipped_no_vec=%d elapsed=%.0fs",
        inserted, skipped_existing, skipped_no_vec, elapsed,
    )
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description="Ingest arXiv papers + vectors into Mongo")
    ap.add_argument("--csv", default="/storage/bary/arxiv/papers.csv")
    ap.add_argument("--vec", default="/storage/bary/arxiv/vectors.dat")
    ap.add_argument("--meta", default="/storage/bary/arxiv/arxiv-metadata-oai-snapshot.json")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
