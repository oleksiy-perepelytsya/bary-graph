"""Pure batch-record -> ParsedWord/ParsedSense extraction (academic corpus).

Batch records look like ``{"doi": "10.xxxx/...", "terms": [{"id": "t01",
"term": "...", "gloss": "..."}, ...]}`` — one JSON object per line, one line
per source paper. Unlike kaikki, batch terms carry no POS and no relations;
every term becomes exactly one (ParsedWord, ParsedSense) pair with
``pos="term"`` (see lib.config for why this sentinel is safe against
kaikki's real POS tag set) and ``sense_idx=0``.

Kept I/O-light so unit tests can drive ``parse_batch_entry`` with fixture
dicts, mirroring lib.parse's role for the kaikki source.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import orjson

from lib.schema import ParsedSense, ParsedWord

BATCH_POS = "term"

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_term(term: str) -> str:
    """Merge key for matching/dedup: collapse whitespace, preserve case.

    Case is preserved because it's still needed for the DB lookup strategy
    (try as-is, then lowercased) — this only strips incidental whitespace
    noise from extraction (e.g. "CO 2 capture efficiency" is left as-is;
    only leading/trailing/doubled whitespace is normalized).
    """
    return _WHITESPACE_RE.sub(" ", term.strip())


def _make_sense_id(doi: str | None, term_id: str) -> str:
    return f"batch:{doi or 'nodoi'}:{term_id}"


def parse_batch_entry(obj: dict[str, Any]) -> list[tuple[ParsedWord, ParsedSense]]:
    """Convert one batch JSONL record to a list of (ParsedWord, ParsedSense).

    Returns one pair per term; unlike lib.parse.parse_entry there is no
    filtering (batch terms have no lang_code/pos to reject on) — an empty
    ``terms`` list just yields an empty result.
    """
    doi = obj.get("doi") or None
    out: list[tuple[ParsedWord, ParsedSense]] = []
    for t in obj.get("terms") or []:
        term = t.get("term")
        gloss = t.get("gloss")
        term_id = t.get("id")
        if not term or not gloss or not term_id:
            continue
        sense_id = _make_sense_id(doi, str(term_id))
        dois = [doi] if doi else []
        ps = ParsedSense(
            word=term,
            pos=BATCH_POS,
            sense_id=sense_id,
            sense_idx=0,
            gloss=gloss,
            doi=dois,
            embed_text=gloss,
        )
        pw = ParsedWord(
            word=term,
            pos=BATCH_POS,
            lang_code="en",
            sense_ids=[sense_id],
        )
        out.append((pw, ps))
    return out


def parse_batch_file(path: str | Path) -> list[tuple[ParsedWord, ParsedSense]]:
    """Read one batch JSONL file and parse every record."""
    out: list[tuple[ParsedWord, ParsedSense]] = []
    with Path(path).open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            out.extend(parse_batch_entry(obj))
    return out
