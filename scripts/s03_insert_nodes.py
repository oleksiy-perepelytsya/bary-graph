"""Insert L15 sense + L14 word node docs into MongoDB.

Reads stage-02's ``senses_embedded.jsonl`` and stage-01's ``words.jsonl``
and upserts them as ``doc_type='node'`` documents. Upsert key is the
kaikki-stable ``properties.sense_id`` (senses) and ``(word, pos)`` pair
(words), so re-running this stage is idempotent and never duplicates.

Safeguard: refuses to run against a collection that already contains
``baryedge`` documents unless ``--force`` is given — re-inserting nodes
under existing edges would orphan every ``parent_edge_id`` reference.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import orjson
from pymongo import UpdateOne

from lib import checkpoint as cp_mod
from lib.db import ensure_indexes, get_collection
from lib.docs import sense_node, word_node
from lib.schema import (
    SENSES_EMBEDDED_FILENAME,
    WORDS_FILENAME,
    ParsedSense,
    ParsedSenseRelation,
    ParsedWord,
)
from lib.vector import decode_vec
from scripts._base import bootstrap, finish

STAGE = "03_insert_nodes"


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("rb") as f:
        for line in f:
            yield orjson.loads(line)


def _failed_offsets(src: Path) -> set[int]:
    """Byte offsets of malformed lines, from the sibling failed_lines.txt."""
    failed_path = src.with_name("failed_lines.txt")
    if not failed_path.exists():
        return set()
    offsets: set[int] = set()
    for line in failed_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            offsets.add(int(line.split("\t")[-1]))
        except ValueError:
            continue
    return offsets


def _iter_senses_resumable(
    path: Path,
    resume_offset: int,
    skip_offsets: set[int],
    log,
) -> Iterable[tuple[dict, int, int]]:
    """Yield (record, line_start_offset, next_offset), resuming at
    ``resume_offset`` and skipping+logging any line in ``skip_offsets``."""
    with path.open("rb") as f:
        if resume_offset:
            f.seek(resume_offset)
            log.info("resuming sense phase at byte offset %d", resume_offset)
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if off in skip_offsets:
                log.warning("skipping malformed line at byte offset %d", off)
                continue
            yield orjson.loads(line), off, f.tell()


def _load_sense(rec: dict) -> ParsedSense:
    rec = dict(rec)
    rec["relations"] = [ParsedSenseRelation(**r) for r in rec.get("relations", [])]
    return ParsedSense(**rec)


def _load_word(rec: dict) -> ParsedWord:
    rec = dict(rec)
    rec["relations"] = [ParsedSenseRelation(**r) for r in rec.get("relations", [])]
    return ParsedWord(**rec)


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start processed=%d dry_run=%s", cp.processed, args.dry_run)

    senses_path = Path(settings.parsed_dir) / SENSES_EMBEDDED_FILENAME
    words_path = Path(settings.parsed_dir) / WORDS_FILENAME
    for p in (senses_path, words_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run prior stages first")

    if not args.dry_run:
        n_edges = coll.count_documents({"doc_type": "baryedge"}, limit=1)
        if n_edges and not args.force:
            raise RuntimeError(
                "collection already contains baryedge docs — re-inserting nodes would "
                "corrupt parent_edge_id references. Drop the collection or use --force."
            )
        ensure_indexes(coll)

    batch_n = args.batch_size or settings.batch_size
    n_words = 0

    # --- L15 sense nodes ---
    # Resume from the last committed byte offset (avoids re-reading the whole
    # file on a crash); skip+sense malformed lines recorded by build_line_idx.
    skip_offsets = _failed_offsets(senses_path)
    n_senses = cp.processed
    ops: list[UpdateOne] = []
    batch_next_off = 0
    for rec, _off, next_off in _iter_senses_resumable(
        senses_path, cp.file_offset, skip_offsets, log
    ):
        ps = _load_sense(rec)
        doc = sense_node(ps, decode_vec(rec["vector"]))
        ops.append(
            UpdateOne(
                {"doc_type": "node", "properties.sense_id": ps.sense_id},
                {"$set": doc},
                upsert=True,
            )
        )
        n_senses += 1
        batch_next_off = next_off
        if len(ops) >= batch_n:
            if not args.dry_run:
                coll.bulk_write(ops, ordered=False)
            ops = []
            cp.processed = n_senses
            cp.file_offset = batch_next_off
            cp_mod.save(cp, settings)
    if ops:
        if not args.dry_run:
            coll.bulk_write(ops, ordered=False)
        cp.processed = n_senses
        cp.file_offset = batch_next_off
        cp_mod.save(cp, settings)
    log.info("upserted %d L15 sense nodes", n_senses)

    # --- L14 word nodes (placeholder vectors) ---
    ops = []
    for rec in _iter_jsonl(words_path):
        pw = _load_word(rec)
        doc = word_node(pw)
        ops.append(
            UpdateOne(
                {
                    "doc_type": "node",
                    "node_type": "word",
                    "properties.word": pw.word,
                    "properties.pos": pw.pos,
                    "properties.lang": pw.lang_code,
                },
                {"$set": doc},
                upsert=True,
            )
        )
        n_words += 1
        if len(ops) >= batch_n:
            if not args.dry_run:
                coll.bulk_write(ops, ordered=False)
            ops = []
    if ops and not args.dry_run:
        coll.bulk_write(ops, ordered=False)
    log.info("upserted %d L14 word nodes", n_words)

    cp.processed = n_senses + n_words
    cp.total = n_senses + n_words
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
