"""t01_parse — segment an input document into L15 sentences and L14 titles.

Reads a single text/markdown file, runs :func:`tensegrity_pipeline.parse.parse_document`,
and writes ``sentences.jsonl`` / ``titles.jsonl`` under ``Settings.parsed_dir``.
No MongoDB contact. Output is written to ``*.tmp`` and atomically renamed so a
crash never leaves a half-written file for t02 to consume.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import orjson

from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.parse import parse_document
from tensegrity_pipeline.schema import SENTENCES_FILENAME, TITLES_FILENAME

STAGE = "01_parse"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)

    src: Path | None = Path(args.doc) if args.doc else settings.tensegrity_doc
    if src is None:
        raise SystemExit(
            "no input document: pass --doc PATH (or set TENSEGRITY_DOC in the env)"
        )
    if not src.exists():
        raise FileNotFoundError(f"input document not found: {src}")

    text = src.read_text(encoding="utf-8")
    doc_id = src.stem
    log.info("start doc=%s doc_id=%s dry_run=%s", src, doc_id, args.dry_run)

    titles, sentences = parse_document(text, doc_id)
    log.info("parsed %d sentences and %d titles", len(sentences), len(titles))

    out_dir = Path(settings.parsed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentences_tmp = out_dir / (SENTENCES_FILENAME + ".tmp")
    titles_tmp = out_dir / (TITLES_FILENAME + ".tmp")

    if not args.dry_run:
        with sentences_tmp.open("wb") as sf, titles_tmp.open("wb") as tf:
            for s in sentences:
                sf.write(orjson.dumps(s.to_dict()) + b"\n")
            for t in titles:
                tf.write(orjson.dumps(t.to_dict()) + b"\n")
        os.replace(sentences_tmp, out_dir / SENTENCES_FILENAME)
        os.replace(titles_tmp, out_dir / TITLES_FILENAME)

    cp.processed = len(sentences)
    cp.total = len(sentences)
    if not args.dry_run:
        finish(cp, settings, log)
    else:
        log.info("dry-run: outputs not committed")


if __name__ == "__main__":
    run()
