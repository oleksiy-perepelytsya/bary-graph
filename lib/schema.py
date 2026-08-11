"""Intermediate record schemas exchanged between pipeline stages.

Stage 01 (parse) emits ``ParsedWord`` / ``ParsedSense`` records to
``data/parsed/*.jsonl``. Stage 02 (embed) reads them and attaches vectors.
Stage 03 (insert_nodes) upserts them into MongoDB as L14/L15 node docs.

Keeping the schema here (rather than in each stage) lets us evolve the
contract in one place and test stage boundaries in isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ParsedSenseRelation:
    """A single kaikki-level relation resolved toward a sense.

    ``sense_idx`` is assigned by ``lib.disambiguate.assign_sense``; ``None``
    means the relation stays at word level (L14) for this PoC run.
    """

    kind: str                          # synonyms | antonyms | hypernyms | ...
    word: str
    sense_idx: int | None = None
    dis1: list[int] = field(default_factory=list)


@dataclass
class ParsedSense:
    """One dictionary sense (becomes an L15 node)."""

    word: str
    pos: str
    sense_id: str                      # kaikki-stable id
    sense_idx: int                     # position within senses[]
    gloss: str
    examples: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    wikidata: list[str] = field(default_factory=list)
    # Sense-level relations (rare but authoritative when present)
    relations: list[ParsedSenseRelation] = field(default_factory=list)
    # Source DOI(s) for senses ingested from the academic batch pipeline.
    # Empty for kaikki-sourced senses.
    doi: list[str] = field(default_factory=list)
    # Text used by stage 02 to produce the embedding
    embed_text: str = ""
    # Filled by stage 02 (embed)
    vector: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedWord:
    """One (word, pos) entry (becomes an L14 node)."""

    word: str
    pos: str
    lang_code: str
    ipa: str | None = None
    forms: list[str] = field(default_factory=list)      # surface form strings only
    etymology: str = ""
    sense_ids: list[str] = field(default_factory=list)  # ids of senses emitted
    # Word-level relations (kaikki default)
    relations: list[ParsedSenseRelation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Stage 01 output files, relative to Settings.parsed_dir.
SENSES_FILENAME = "senses.jsonl"
WORDS_FILENAME = "words.jsonl"
# Stage 02 output: senses.jsonl with `vector` populated.
SENSES_EMBEDDED_FILENAME = "senses_embedded.jsonl"
