"""Ingest academic-paper term batches into the live BaryGraph.

Reads one or more ``{"doi": ..., "terms": [{"id", "term", "gloss"}, ...]}``
JSONL files (see batch_001.jsonl.txt), matches each term against the
existing graph, and inserts new L15 sense / L14 "term"-pos word nodes as
orphans (``parent_edge_id=None``) — ready to be woven into the existing
BaryEdge/MetaBary structure by the standard incremental stages afterward.

Unlike scripts/s01-s10, this does NOT go through scripts._base.bootstrap():
that machinery assumes one linear, one-shot corpus build, and its stage
names/checkpoints are already taken by the kaikki pipeline. Batches are
small and every write here is an idempotent upsert/$addToSet, so a crash and
a plain re-run is safe without a resumability layer.

Processing is sequential with immediate writes (not deferred to end-of-run):
a later term in the same invocation, or in a later invocation, sees earlier
ones through the same find_existing_word_candidates/near_duplicate_sense DB
lookups used for anything already in the graph. This is why passing many
files to one invocation (recommended for the initial corpus load) gets
full-context dedup "for free" — there's exactly one ingestion mechanism,
used at any batch size. Caveat: under --dry-run nothing is written, so
duplicates that only appear later in the *same* dry-run invocation won't be
cross-referenced against each other — dry-run merge counts are a lower
bound.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lib import doi_bridge
from lib.batch_merge import (
    dedupe_exact,
    find_existing_word_candidates,
    near_duplicate_sense,
    resolve_pos,
)
from lib.config import Settings
from lib.db import get_collection
from lib.docs import sense_node, word_node
from lib.embed import get_embedder
from lib.log import get_logger, setup_logging
from lib.parse_batch import parse_batch_file

WORD_IDS_FILENAME = "batch_word_ids.json"


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest_batch", description="Ingest academic-paper term batches into BaryGraph"
    )
    p.add_argument("files", nargs="+", help="batch JSONL file(s) to ingest")
    p.add_argument("--dry-run", action="store_true", help="do not write to MongoDB")
    return p


def run(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger("ingest_batch")

    coll = get_collection(settings)
    bridge_coll = doi_bridge.get_bridge_collection(settings)
    if not args.dry_run:
        doi_bridge.ensure_indexes(bridge_coll)

    entries = []
    for f in args.files:
        entries.extend(parse_batch_file(f))
    log.info("parsed %d term occurrences from %d file(s)", len(entries), len(args.files))

    entries = dedupe_exact(entries)
    log.info("after exact dedup: %d unique (term, gloss) pairs", len(entries))

    embedder = get_embedder(settings)
    vectors: list[np.ndarray] = []
    for chunk in _batched([ps.embed_text for _, ps in entries], settings.embed_batch_size):
        vectors.extend(embedder.embed(chunk))

    now = datetime.now(timezone.utc)
    n_new_words = n_new_senses_existing_word = n_merged_dup = 0
    touched_word_ids: list = []

    for (pw, ps), vec in zip(entries, vectors, strict=True):
        candidates = find_existing_word_candidates(coll, ps.word)

        if not candidates:
            if args.dry_run:
                n_new_words += 1
                continue
            sense_doc = sense_node(ps, vec)
            sense_res = coll.insert_one(sense_doc)
            word_doc = word_node(pw)
            word_res = coll.insert_one(word_doc)
            doi_bridge.register(bridge_coll, ps.doi, sense_res.inserted_id)
            doi_bridge.register(bridge_coll, ps.doi, word_res.inserted_id)
            touched_word_ids.append(word_res.inserted_id)
            n_new_words += 1
            continue

        target = resolve_pos(coll, vec, candidates) or candidates[0]
        word, pos = target["properties"]["word"], target["properties"]["pos"]

        dup_id = near_duplicate_sense(coll, word, pos, vec, settings.batch_dup_threshold)
        if dup_id is not None:
            if not args.dry_run:
                coll.update_one(
                    {"_id": dup_id},
                    {
                        "$addToSet": {"properties.doi": {"$each": ps.doi}},
                        "$set": {"updated_at": now},
                    },
                )
                doi_bridge.propagate_up_chain(bridge_coll, coll, dup_id, ps.doi)
                doi_bridge.propagate_up_chain(bridge_coll, coll, target["_id"], ps.doi)
            n_merged_dup += 1
            continue

        if args.dry_run:
            n_new_senses_existing_word += 1
            continue
        ps2 = dataclasses.replace(ps, word=word, pos=pos)
        sense_doc = sense_node(ps2, vec)
        sense_res = coll.insert_one(sense_doc)
        coll.update_one(
            {"_id": target["_id"]},
            {
                "$push": {"properties.sense_ids": ps2.sense_id},
                "$inc": {"surface": 1},
                "$set": {"updated_at": now},
            },
        )
        doi_bridge.register(bridge_coll, ps2.doi, sense_res.inserted_id)
        touched_word_ids.append(target["_id"])
        n_new_senses_existing_word += 1

    log.info(
        "done: %d new words, %d new senses under existing words, %d merged into "
        "near-duplicate existing senses",
        n_new_words, n_new_senses_existing_word, n_merged_dup,
    )

    if not args.dry_run and touched_word_ids:
        out_path = Path(settings.pipeline_state_dir) / WORD_IDS_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([str(i) for i in set(touched_word_ids)]))
        log.info("wrote %d touched word ids to %s", len(set(touched_word_ids)), out_path)

    print()
    print("Next, weave the new orphans into the graph (safe to re-run; only")
    print("touches parent_edge_id: None docs). --force is needed on every stage")
    print("below because they've already completed once against the live graph")
    print("(scripts._base.bootstrap refuses to re-run a stage marked done):")
    print("  python -m scripts.s04_l15_edges --force")
    if touched_word_ids and not args.dry_run:
        print(f"  python -m scripts.s05_word_vectors --force --word-ids-file {out_path}")
    else:
        print("  python -m scripts.s05_word_vectors --force")
    print("  python -m scripts.s06_l14_edges --force")
    print("  python -m scripts.s07_orphan_reentry --force")
    print("  python -m scripts.s08_metabary --force")
    print()
    print("Defer s09_extend until the full corpus has been ingested — it sweeps")
    print("the entire unparented-BE pool to convergence, not just new content.")


if __name__ == "__main__":
    run()
