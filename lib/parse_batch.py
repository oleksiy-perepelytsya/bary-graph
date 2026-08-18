"""Pure batch-record -> ParsedWord/ParsedSense extraction (academic corpus).

Two record shapes are accepted, both one JSON object per line:

- Flat: ``{"doi": "10.xxxx/...", "terms": [{"id": "t01", "term": "...",
  "gloss": "..."}, ...]}``.
- Wrapped (extraction-pipeline output, e.g. data/extracted_terms.jsonl):
  ``{"doi": "10.xxxx/..." | "(no doi)", "cost_usd": ..., "duration_ms": ...,
  "extracted": {"doi": ..., "terms": [...]}}``. The outer ``doi`` is treated
  as authoritative and the inner ``extracted.doi`` is ignored — the two can
  disagree (observed in production: outer full DOI vs. a truncated inner
  copy; outer sentinel string ``"(no doi)"`` vs. inner JSON ``null``), and
  the outer field comes from the paper's own metadata rather than being
  echoed back by the extraction step. ``cost_usd``/``duration_ms`` are
  extraction-run bookkeeping and are ignored.

Unlike kaikki, batch terms carry no POS and no relations; every term becomes
exactly one (ParsedWord, ParsedSense) pair with ``pos="term"`` (see
lib.config for why this sentinel is safe against kaikki's real POS tag set)
and ``sense_idx=0``.

Kept I/O-light so unit tests can drive ``parse_batch_entry`` with fixture
dicts, mirroring lib.parse's role for the kaikki source.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import orjson

from lib.schema import ParsedSense, ParsedWord

BATCH_POS = "term"

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_term(term: str) -> str:
    """Merge key for matching/dedup: collapse whitespace, preserve case.

    Case is preserved because word matching against the DB is exact-case
    (see lib.batch_merge.find_existing_word_candidates) — this only strips
    incidental whitespace noise from extraction (e.g. "CO 2 capture
    efficiency" is left as-is; only leading/trailing/doubled whitespace is
    normalized).
    """
    return _WHITESPACE_RE.sub(" ", term.strip())


_NO_DOI_SENTINELS = {"(no doi)", "no doi", "n/a", "none"}


def _normalize_doi(raw: Any) -> str | None:
    """Collapse missing/placeholder DOI spellings to None.

    The extraction pipeline emits a literal ``"(no doi)"`` string rather than
    JSON null when a paper has no DOI (observed in data/extracted_terms.jsonl).
    """
    if not raw:
        return None
    s = str(raw).strip()
    if not s or s.lower() in _NO_DOI_SENTINELS:
        return None
    return s


def _make_sense_id(doi: str | None, term_id: str, gloss: str) -> str:
    """Stable id for one term occurrence.

    ``term_id`` (e.g. "t01") is only unique within a single source paper —
    with a doi that's enough to key on. Without one (42 of 1057 records in
    data/extracted_terms.jsonl have no doi), "t01" from one paper would
    collide with "t01" from every other doi-less paper, so a short hash of
    the gloss is folded in to keep the id actually unique.
    """
    if doi:
        return f"batch:{doi}:{term_id}"
    h = hashlib.sha1(gloss.encode("utf-8")).hexdigest()[:10]
    return f"batch:nodoi:{term_id}:{h}"


def parse_batch_entry(obj: dict[str, Any]) -> list[tuple[ParsedWord, ParsedSense]]:
    """Convert one batch JSONL record to a list of (ParsedWord, ParsedSense).

    Accepts both the flat shape (``terms`` at top level) and the wrapped
    extraction-pipeline shape (``terms`` under ``obj["extracted"]``) — see
    module docstring. Returns one pair per term; unlike lib.parse.parse_entry
    there is no filtering (batch terms have no lang_code/pos to reject on) —
    an empty ``terms`` list just yields an empty result.
    """
    doi = _normalize_doi(obj.get("doi"))
    terms_source = obj["extracted"] if "extracted" in obj else obj
    out: list[tuple[ParsedWord, ParsedSense]] = []
    for t in (terms_source or {}).get("terms") or []:
        term = t.get("term")
        gloss = t.get("gloss")
        term_id = t.get("id")
        if not term or not gloss or not term_id:
            continue
        sense_id = _make_sense_id(doi, str(term_id), gloss)
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
