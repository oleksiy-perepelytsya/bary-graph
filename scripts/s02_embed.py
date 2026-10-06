"""Embed sense glosses via ollama (or FakeEmbedder).

Reads ``senses.jsonl`` from stage 01, batch-embeds the ``embed_text`` field,
and writes ``senses_embedded.jsonl`` with the ``vector`` field populated.
Atomic-rename on completion; resumable by line count.

If ollama is unreachable after exhausting retries, the batch is written with
zero vectors and the offset is saved to a ``missed.txt`` file alongside the
output so a follow-up patch pass can fill them in without re-embedding
everything.
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
from lib.schema import SENSES_EMBEDDED_FILENAME, SENSES_FILENAME
from lib.vector import encode_vec
from scripts._base import bootstrap, finish

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
    src = Path(settings.parsed_dir) / SENSES_FILENAME
    log.info("start src=%s processed=%d dry_run=%s", src, cp.processed, args.dry_run)

    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run stage 01_parse first")

    embedder = get_embedder(settings)
    batch_n = args.batch_size or settings.embed_batch_size
    out_tmp = Path(settings.parsed_dir) / (SENSES_EMBEDDED_FILENAME + ".tmp")
    missed_path = Path(settings.parsed_dir) / "missed.txt"
    missed_fh = missed_path.open("a")

    # Warm up ollama: first request after idle takes 60-120s (model load).
    # Subsequent requests are ~8s/batch.  Do a tiny probe so every real
    # batch hits a warm model.
    if not settings.fake_embed:
        log.info("warming up ollama…")
        try:
            embedder.embed(["warmup"])
            log.info("ollama warm")
        except Exception as exc:
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
            texts = [rec["embed_text"] for rec in batch]
            try:
                vecs = embedder.embed(texts)
            except Exception as exc:
                _log.warning("batch embed failed at n=%d (%d texts): %s — "
                             "writing zero vectors", n, len(texts), exc)
                vecs = np.zeros((len(texts), embedder.dim), dtype=np.float32)
                for i, rec in enumerate(batch):
                    missed_fh.write(f"{n + i}\t{rec['sense_id']}\n")
                missed_fh.flush()
                n_missed += len(texts)
            lines: list[bytes] = []
            for rec, v in zip(batch, vecs, strict=True):
                n += 1
                if n <= skip:
                    continue
                rec["vector"] = encode_vec(v)
                lines.append(orjson.dumps(rec) + b"\n")
            # Write whole batch atomically then save checkpoint — a crash
            # between these two leaves at most one batch of duplicates, which
            # stage 03 deduplicates via upsert on sense_id.
            if lines and not args.dry_run:
                out.write(b"".join(lines))
                out.flush()
            cp.processed = n
            cp_mod.save(cp, settings)
            if n % (batch_n * 50) == 0:
                log.info("… embedded %d senses", n)

    missed_fh.close()
    cp.processed = n
    cp.total = n
    _log.info("embedded %d senses (dim=%d, missed=%d)", n, embedder.dim, n_missed)

    if not args.dry_run:
        os.replace(out_tmp, Path(settings.parsed_dir) / SENSES_EMBEDDED_FILENAME)
        finish(cp, settings, log)
    else:
        log.info("dry-run: output not committed")


if __name__ == "__main__":
    run()
