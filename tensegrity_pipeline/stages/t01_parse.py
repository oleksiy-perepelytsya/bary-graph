"""t01_parse — segment input documents into L15 sentences and L14 titles.

Reads one or more text/markdown files (``--doc`` is repeatable), runs
:func:`tensegrity_pipeline.parse.parse_documents`, and writes
``sentences.jsonl`` / ``titles.jsonl`` under ``Settings.parsed_dir``. Every
document keeps a distinct ``doc_id`` (its filename stem) so a multi-document
run preserves source identity; output order follows command-line order.
No MongoDB contact. Output is written to ``*.tmp`` and atomically renamed so a
crash never leaves a half-written file for t02 to consume.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import orjson

from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.parse import parse_documents
from tensegrity_pipeline.schema import SENTENCES_FILENAME, TITLES_FILENAME

STAGE = "01_parse"


def _resolve_inputs(args, settings) -> list[Path]:
    """--doc args win; fall back to the TENSEGRITY_DOC env default."""
    if args.doc:
        return [Path(p) for p in args.doc]
    if settings.tensegrity_doc is not None:
        return [Path(settings.tensegrity_doc)]
    raise SystemExit(
        "no input document: pass --doc PATH (repeatable for multiple docs) "
        "or set TENSEGRITY_DOC in the env"
    )


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)

    paths = _resolve_inputs(args, settings)
    docs: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"input document not found: {path}")
        docs.append((path.stem, path.read_text(encoding="utf-8")))
        log.info("input doc: %s (doc_id=%s)", path, path.stem)

    titles, sentences = parse_documents(docs)
    log.info(
        "parsed %d sentences and %d titles across %d documents",
        len(sentences), len(titles), len(docs),
    )

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
