"""t02_embed — embed L15 sentence texts via ollama (or FakeEmbedder).

Reads ``sentences.jsonl`` from t01, batch-embeds ``embed_text``, and writes
``sentences_embedded.jsonl`` with the ``vector`` field populated (base64 of the
raw float32 blob — see :mod:`lib.vector`). Titles are **not** embedded here:
they are aggregation nodes whose vectors are derived in t05, mirroring the PoC.

Atomic-rename on completion; resumable by line count. Batches that fail after
retries are written with zero vectors and their line offsets logged to
``missed.txt`` for a later patch pass.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import orjson

from lib import checkpoint as cp_mod
from lib.embed import get_embedder
from lib.vector import encode_vec
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.schema import SENTENCES_EMBEDDED_FILENAME, SENTENCES_FILENAME

STAGE = "02_embed"
_log = logging.getLogger(__name__)


def _batched(it, n):
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    src = Path(settings.parsed_dir) / SENTENCES_FILENAME
    log.info("start src=%s processed=%d dry_run=%s", src, cp.processed, args.dry_run)

    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run stage 01_parse first")

    embedder = get_embedder(settings)
    batch_n = args.batch_size or settings.embed_batch_size
    out_tmp = Path(settings.parsed_dir) / (SENTENCES_EMBEDDED_FILENAME + ".tmp")
    missed_path = Path(settings.parsed_dir) / "missed.txt"
    missed_fh = missed_path.open("a")

    if not settings.fake_embed:
        log.info("warming up ollama…")
        try:
            embedder.embed(["warmup"])
            log.info("ollama warm")
        except Exception as exc:  # noqa: BLE001
            log.warning("warmup failed (will retry on first batch): %s", exc)

    skip = cp.processed
    n = 0
    n_missed = 0

    def _lines():
        with src.open("rb") as f:
            for line in f:
                yield orjson.loads(line)

    with out_tmp.open("wb" if skip == 0 else "ab") as out:
        for batch in _batched(_lines(), batch_n):
            if n + len(batch) <= skip:
                n += len(batch)
                continue
            if args.limit and n - skip >= args.limit:
                break
            texts = [rec["embed_text"] or rec["text"] for rec in batch]
            try:
                vecs = embedder.embed(texts)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "batch embed failed at n=%d (%d texts): %s — writing zero vectors",
                    n, len(texts), exc,
                )
                vecs = np.zeros((len(texts), embedder.dim), dtype=np.float32)
                for i, rec in enumerate(batch):
                    missed_fh.write(f"{n + i}\t{rec['doc_id']}\n")
                missed_fh.flush()
                n_missed += len(texts)
            lines: list[bytes] = []
            for rec, v in zip(batch, vecs, strict=True):
                n += 1
                if n <= skip:
                    continue
                rec["vector"] = encode_vec(v)
                lines.append(orjson.dumps(rec) + b"\n")
            if lines and not args.dry_run:
                out.write(b"".join(lines))
                out.flush()
            cp.processed = n
            cp_mod.save(cp, settings)
            if n % (batch_n * 50) == 0:
                log.info("… embedded %d sentences", n)

    missed_fh.close()
    cp.processed = n
    cp.total = n
    _log.info("embedded %d sentences (dim=%d, missed=%d)", n, embedder.dim, n_missed)

    if not args.dry_run:
        os.replace(out_tmp, Path(settings.parsed_dir) / SENTENCES_EMBEDDED_FILENAME)
        finish(cp, settings, log)
    else:
        log.info("dry-run: output not committed")


if __name__ == "__main__":
    run()
