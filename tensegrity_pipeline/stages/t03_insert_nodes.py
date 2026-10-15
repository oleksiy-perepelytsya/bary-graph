"""t03_insert_nodes — insert L15 sentence + L14 title node docs.

Reads t02's ``sentences_embedded.jsonl`` and t01's ``titles.jsonl`` and upserts
them as ``doc_type='node'`` documents. Upsert keys are the stable sentence /
title ids so re-running is idempotent. Titles are inserted with a ``vector=None``
placeholder (filled by t05), mirroring the PoC's word nodes.

This is also the *root* of document provenance: every source document is
registered in the sibling ``doc_bridges`` collection together with the nodes it
produced (``doc_bridge.register_nodes_for_docs``), so upper levels can be
propagated later by ``doc_bridge.propagate``.

Safeguard: refuses to run against a collection that already contains
``baryedge`` documents unless ``--force`` is given — re-inserting nodes under
existing edges would orphan every ``parent_edge_id`` reference.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import orjson
from pymongo import UpdateOne

from lib.vector import decode_vec
from tensegrity_pipeline import doc_bridge
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.db import ensure_indexes, get_collection
from tensegrity_pipeline.docs import sentence_node, title_node
from tensegrity_pipeline.schema import (
    SENTENCES_EMBEDDED_FILENAME,
    TITLES_FILENAME,
    ParsedSentence,
    ParsedTitle,
)

STAGE = "03_insert_nodes"


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("rb") as f:
        for line in f:
            yield orjson.loads(line)


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start processed=%d dry_run=%s", cp.processed, args.dry_run)

    sentences_path = Path(settings.parsed_dir) / SENTENCES_EMBEDDED_FILENAME
    titles_path = Path(settings.parsed_dir) / TITLES_FILENAME
    for p in (sentences_path, titles_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run prior stages first")

    if not args.dry_run:
        if coll.count_documents({"doc_type": "baryedge"}, limit=1) and not args.force:
            raise RuntimeError(
                "collection already contains baryedge docs — re-inserting nodes would "
                "corrupt parent_edge_id references. Drop the collection or use --force."
            )
        ensure_indexes(coll)
        doc_bridge.ensure_indexes(doc_bridge.get_bridge_collection(settings))

    batch_n = args.batch_size or settings.batch_size
    bridge_coll = doc_bridge.get_bridge_collection(settings)
    # Source documents of this run — the root registrations below.
    doc_ids: set[str] = set()

    # --- L15 sentence nodes ---
    ops: list[UpdateOne] = []
    n_sentences = 0
    for rec in _iter_jsonl(sentences_path):
        ps = ParsedSentence(**{k: v for k, v in rec.items() if k != "vector"})
        doc = sentence_node(ps, decode_vec(rec["vector"]))
        ops.append(
            UpdateOne(
                {"doc_type": "node", "properties.sentence_id": ps.sentence_id},
                {"$set": doc},
                upsert=True,
            )
        )
        doc_ids.add(ps.doc_id)
        n_sentences += 1
        if len(ops) >= batch_n:
            if not args.dry_run:
                coll.bulk_write(ops, ordered=False)
            ops = []
    if ops and not args.dry_run:
        coll.bulk_write(ops, ordered=False)
    log.info("upserted %d L15 sentence nodes", n_sentences)

    # --- L14 title nodes (placeholder vectors) ---
    ops = []
    n_titles = 0
    for rec in _iter_jsonl(titles_path):
        pt = ParsedTitle(**rec)
        ops.append(
            UpdateOne(
                {"doc_type": "node", "properties.title_id": pt.title_id},
                {"$set": title_node(pt)},
                upsert=True,
            )
        )
        doc_ids.add(pt.doc_id)
        n_titles += 1
        if len(ops) >= batch_n:
            if not args.dry_run:
                coll.bulk_write(ops, ordered=False)
            ops = []
    if ops and not args.dry_run:
        coll.bulk_write(ops, ordered=False)
    log.info("upserted %d L14 title nodes", n_titles)

    # --- doc_bridges: root registration (document -> every node under it) ---
    # The tensegrity equivalent of ingest_batch's doi_bridge.register: each
    # source document gets one entry listing the nodes it produced. Edges and
    # MetaBarys are added to those same entries by doc_bridge.propagate in the
    # edge-building stages (t04/t06/t07/t08/t09).
    if not args.dry_run:
        n_reg = doc_bridge.register_nodes_for_docs(bridge_coll, coll, doc_ids)
        log.info(
            "registered %d nodes under %d documents in %s",
            n_reg, len(doc_ids), settings.doc_bridges_collection,
        )

    cp.processed = n_sentences + n_titles
    cp.total = cp.processed
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
