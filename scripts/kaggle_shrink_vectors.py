#!/usr/bin/env python3
"""Shrink a senses_embedded jsonl of fat JSON-float vectors to the compact
base64 float32 form *in place* — no extra disk needed.

Safety: each transformed line is strictly shorter than the source line, so a
write cursor lagging a read cursor (on a separate descriptor of the same
file) can patch the file in a single pass without overwriting unread bytes.
Trailing leftovers are removed with a final truncate. Row content
(sense_id, gloss, …) and row ORDER are preserved, so an s02 checkpoint
whose ``processed == row count`` still lines up with this file (s02 resumes
by row index and appends).

Usage:
    python3 -m scripts.kaggle_shrink_vectors \
        kaggle_work/a/parsed/senses_embedded.jsonl.tmp \
        kaggle_work/b/parsed/senses_embedded.jsonl.tmp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.vector import encode_vec  # noqa: E402

_log = logging.getLogger("kaggle_shrink")


def shrink(path: Path) -> int:
    n_rows = 0
    n_converted = 0
    old_size = path.stat().st_size
    _log.info("%s: %.1f MB", path, old_size / 1e6)
    rd = open(path, "rb")          # independent read cursor
    wr = open(path, "r+b")         # write cursor always behind it
    write_pos = 0
    buf = bytearray()
    try:
        for line in rd:
            try:
                rec = orjson.loads(line)
            except orjson.JSONDecodeError:
                buf.extend(line)
            else:
                if isinstance(rec.get("vector"), list):
                    rec["vector"] = encode_vec(rec["vector"])
                    n_converted += 1
                buf.extend(orjson.dumps(rec) + b"\n")
            if len(buf) >= 1 << 20:
                wr.seek(write_pos)
                wr.write(buf)
                write_pos += len(buf)
                buf.clear()
            n_rows += 1
        if buf:
            wr.seek(write_pos)
            wr.write(buf)
            write_pos += len(buf)
        wr.truncate(write_pos)
    finally:
        rd.close()
        wr.close()
    _log.info(
        "%s: %d rows, %d converted, %.1f MB -> %.1f MB",
        path, n_rows, n_converted, old_size / 1e6, write_pos / 1e6,
    )
    return n_rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(prog="kaggle_shrink_vectors")
    p.add_argument("paths", nargs="+", type=Path)
    a = p.parse_args()
    for path in a.paths:
        if not path.exists():
            _log.error("missing: %s", path)
            return 2
    for path in a.paths:
        rows = shrink(path)
        _log.info("rows after shrink: %d (must equal s02 checkpoint processed)", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
