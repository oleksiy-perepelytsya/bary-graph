"""Intermediate record schemas for the tensegrity pipeline.

Stage t01 (parse) emits ``ParsedSentence`` / ``ParsedTitle`` records to
``data/parsed_tensegrity/*.jsonl``. t02 (embed) attaches vectors to sentences
only — titles deliberately carry no native embedding (per the user's decision)
and get theirs derived in t05, mirroring the PoC's word-vector stage.

Keeping the schema here (rather than in each stage) lets us evolve the contract
in one place and test stage boundaries in isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ParsedSentence:
    """One document sentence (becomes an L15 node)."""

    doc_id: str
    paragraph_idx: int
    sentence_idx: int
    text: str
    # Index of the enclosing title/section, or -1 when the sentence precedes
    # any title. Membership (sentence → title) is recorded as metadata here,
    # not as pairwise edges — see the architecture doc's "no implicit
    # semantics from co-location".
    section_idx: int = -1
    char_start: int = 0
    char_end: int = 0
    # Text handed to the embedder (== text unless a caller normalizes it).
    embed_text: str = ""
    # Filled by t02 (embed).
    vector: list[float] | None = None

    @property
    def sentence_id(self) -> str:
        return f"{self.doc_id}:p{self.paragraph_idx}:s{self.sentence_idx}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedTitle:
    """One paragraph/section title (becomes an L14 node)."""

    doc_id: str
    section_idx: int
    text: str
    title_level: int = 1
    # Paragraph index where the title appeared (provenance only).
    paragraph_idx: int = 0

    @property
    def title_id(self) -> str:
        return f"{self.doc_id}:t{self.section_idx}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Stage t01 output files, relative to Settings.parsed_dir.
SENTENCES_FILENAME = "sentences.jsonl"
TITLES_FILENAME = "titles.jsonl"
# Stage t02 output: sentences.jsonl with `vector` populated.
SENTENCES_EMBEDDED_FILENAME = "sentences_embedded.jsonl"
