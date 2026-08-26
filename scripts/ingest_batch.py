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
used at any batch size. The write path itself lives in lib.batch_ingest,
shared verbatim with the цех ingest_terms MCP tool. Caveat: under --dry-run
nothing is written, so duplicates that only appear later in the *same*
dry-run invocation won't be cross-referenced against each other — dry-run
merge counts are a lower bound.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lib import doi_bridge
from lib.batch_ingest import ingest_entries
from lib.config import Settings
from lib.db import get_collection
from lib.embed import get_embedder
from lib.log import get_logger, setup_logging
from lib.parse_batch import parse_batch_file

WORD_IDS_FILENAME = "batch_word_ids.json"


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

    embedder = get_embedder(settings)

    if args.dry_run:
        from lib.batch_merge import dedupe_exact

        entries = dedupe_exact(entries)
        n_new_words = n_merged_dup = 0
        log.info("after exact dedup: %d unique (term, gloss) pairs", len(entries))
        log.info("--dry-run: no writes; merge counts unavailable without embedding pass")
        touched_word_ids: list = []
        out_path = Path(settings.pipeline_state_dir) / WORD_IDS_FILENAME
    else:
        stats = ingest_entries(coll, bridge_coll, embedder, settings, entries)
        log.info(
            "after exact dedup: %d unique (term, gloss) pairs",
            stats["n_after_exact_dedup"],
        )
        log.info(
            "done: %d new words, %d new senses under existing words, %d merged into "
            "near-duplicate existing senses",
            stats["n_new_words"], stats["n_new_senses_existing_word"], stats["n_merged_dup"],
        )
        touched_word_ids = stats["touched_word_ids"]
        n_new_words = stats["n_new_words"]
        n_merged_dup = stats["n_merged_dup"]

        if touched_word_ids:
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
