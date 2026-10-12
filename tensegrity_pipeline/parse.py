"""Pure text → sentence/title segmentation.

Kept I/O-free and dependency-free so unit tests can drive it with fixture
strings; stage t01 streams the results to JSONL.

Title detection is **programmatic** (the user's choice): markdown ATX/setext
headings, numbered headings, ALL-CAPS short lines, and Title-Case short lines
without terminal punctuation. No model call is involved.

Sentence splitting is a small deterministic scanner with a common-abbreviation
guard, not a statistical segmenter. The policy is documented and stable, which
is what the architecture spec asks of a segmentation policy.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from tensegrity_pipeline.schema import ParsedSentence, ParsedTitle

# --- title heuristics -------------------------------------------------------

_ATX_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+\S")
_SETEXT_UNDERLINE = re.compile(r"^\s{0,3}(=+|-{2,})\s*$")
_NUMBERED_HEADING = re.compile(
    r"^\s{0,3}(?:(\d+(?:\.\d+)*)[.)]?|([IVXLC]+)[.)]|([A-Z])[.)])\s+\S"
)
_TERMINAL_PUNCT = ".!?,;:"
# Lines longer than this are never treated as titles.
_TITLE_MAX_CHARS = 100
_TITLE_MAX_WORDS = 14

# --- sentence splitting -----------------------------------------------------

_SENT_END_CHARS = ".!?"
_CLOSERS = "\"')]}”’»"
# Lower-cased, period-stripped tokens after which a "." does not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "fig", "no", "vol", "al", "inc", "ltd", "co", "dept",
    "univ", "approx", "est", "min", "max", "cf", "ca", "p", "pp", "ch",
    "sec", "art", "eq", "nr", "ed", "eds", "op", "cit", "ibid",
}
_ACRONYM = re.compile(r"^(?:[A-Za-z]\.){2,}$")  # "U.S.", "e.g."


def _word_before(text: str, end: int) -> str:
    """The whitespace-delimited token ending just before ``end``."""
    i = end
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return text[i:end]


def _is_abbreviation(text: str, start: int, period_idx: int) -> bool:
    # Include the terminating period so acronym tokens like "U.S." are intact.
    tok = _word_before(text, period_idx + 1).strip()
    if not tok:
        return False
    low = tok.lower().rstrip(".")
    if low in _ABBREVIATIONS:
        return True
    # "U.S." / "e.g." style: the token already contains internal periods.
    if _ACRONYM.match(tok):
        return True
    # Single-letter initials (e.g. "J.").
    return len(tok) == 2 and tok[0].isalpha() and tok[1] == "."


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int


def split_sentences_with_spans(text: str) -> list[Span]:
    """Split ``text`` into sentence spans with character offsets.

    Deterministic scanner: a sentence ends at ``.``/``!``/``?`` (plus trailing
    closing quotes/brackets) followed by whitespace, unless the ``.`` belongs
    to a known abbreviation or an acronym/initial.
    """
    spans: list[Span] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        c = text[i]
        if c in _SENT_END_CHARS:
            j = i + 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            at_boundary = j >= n or text[j].isspace()
            if at_boundary and not (c == "." and _is_abbreviation(text, start, i)):
                seg = text[start:j].strip()
                if seg:
                    # Recover trimmed offsets.
                    lead = len(text[start:j]) - len(text[start:j].lstrip())
                    s = start + lead
                    e = s + len(seg)
                    spans.append(Span(seg, s, e))
                k = j
                while k < n and text[k].isspace():
                    k += 1
                start = k
                i = k
                continue
        i += 1
    tail = text[start:].strip()
    if tail:
        lead = len(text[start:]) - len(text[start:].lstrip())
        s = start + lead
        spans.append(Span(tail, s, s + len(tail)))
    return spans


def split_sentences(text: str) -> list[str]:
    """Convenience wrapper returning sentence strings only."""
    return [sp.text for sp in split_sentences_with_spans(text)]


def _clean_title_text(line: str) -> str:
    """Strip markdown ATX markers from a heading for storage/embedding."""
    m = _ATX_HEADING.match(line)
    if m:
        return line.strip()[len(m.group(1)):].strip()
    return line.strip()


def _title_level(line: str, numbered: re.Match[str] | None) -> int:
    m = _ATX_HEADING.match(line)
    if m:
        return len(m.group(1))
    if numbered:
        if numbered.group(1):  # 1.2.3 → depth 3
            return numbered.group(1).count(".") + 1
        return 1
    return 1


def is_title_line(line: str, *, numbered: re.Match[str] | None = None) -> bool:
    """Heuristic title test for a single (already newline-stripped) line."""
    t = line.strip()
    if not t:
        return False
    if _ATX_HEADING.match(line):
        return True
    if numbered is None:
        numbered = _NUMBERED_HEADING.match(line)
    if numbered and len(t) <= _TITLE_MAX_CHARS and not t.endswith(_TERMINAL_PUNCT):
        return True
    if len(t) > _TITLE_MAX_CHARS:
        return False
    if t.endswith(_TERMINAL_PUNCT):
        return False
    words = t.split()
    if not words or len(words) > _TITLE_MAX_WORDS:
        return False
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio >= 0.8:
        return True
    capitalised = sum(1 for w in words if w[:1].isupper())
    return capitalised / len(words) >= 0.6


def parse_document(
    text: str, doc_id: str
) -> tuple[list[ParsedTitle], list[ParsedSentence]]:
    """Segment ``text`` into section titles (L14) and sentences (L15).

    Scans line-by-line, tracking the current title (``section_idx``) so every
    sentence records which section it belongs to. Consecutive non-title,
    non-blank lines form a paragraph, which is then split into sentences.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    titles: list[ParsedTitle] = []
    sentences: list[ParsedSentence] = []
    paragraph_idx = -1          # increments per emitted body paragraph
    section_idx = -1            # index into `titles`; -1 = before any title
    para_lines: list[str] = []
    # Character cursor into `text` used to locate sentence offsets.
    cursor = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_idx, cursor
        if not para_lines:
            return
        paragraph = " ".join(para_lines)
        paragraph_idx += 1
        for s_idx, span in enumerate(split_sentences_with_spans(paragraph)):
            found = text.find(span.text, cursor)
            if found == -1:
                found = cursor
            cursor = found + len(span.text)
            sentences.append(
                ParsedSentence(
                    doc_id=doc_id,
                    paragraph_idx=paragraph_idx,
                    sentence_idx=s_idx,
                    text=span.text,
                    section_idx=section_idx,
                    char_start=found,
                    char_end=found + len(span.text),
                    embed_text=span.text,
                )
            )
        para_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        numbered = _NUMBERED_HEADING.match(line)
        # Setext heading: current line underlined by === / ---.
        setext = False
        if stripped and i + 1 < len(lines):
            nxt = lines[i + 1]
            if _SETEXT_UNDERLINE.match(nxt) and stripped:
                setext = True
        is_title = setext or is_title_line(line, numbered=numbered)
        if is_title:
            flush_paragraph()
            titles.append(
                ParsedTitle(
                    doc_id=doc_id,
                    section_idx=len(titles),
                    text=_clean_title_text(line),
                    title_level=_title_level(line, numbered),
                    paragraph_idx=paragraph_idx + 1,
                )
            )
            section_idx = len(titles) - 1
            if setext:
                i += 2  # consume the underline
                continue
        elif stripped:
            para_lines.append(stripped)
        else:
            flush_paragraph()
        i += 1

    flush_paragraph()
    return titles, sentences


def parse_documents(
    docs: Sequence[tuple[str, str]],
) -> tuple[list[ParsedTitle], list[ParsedSentence]]:
    """Segment multiple ``(doc_id, text)`` pairs, preserving doc order.

    Each document keeps its own ``doc_id`` lineage — L15 sentence and L14 title
    labels embed it — so a multi-document run preserves distinct source
    identity (viz. the GovKM S001 requirement "keep P01/P02/P03 as distinct
    source events"). Raises ``ValueError`` on duplicate ``doc_id`` because node
    labels would collide.
    """
    titles_all: list[ParsedTitle] = []
    sentences_all: list[ParsedSentence] = []
    seen: set[str] = set()
    for doc_id, text in docs:
        if doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc_id!r} in multi-document input")
        seen.add(doc_id)
        titles, sentences = parse_document(text, doc_id)
        titles_all.extend(titles)
        sentences_all.extend(sentences)
    return titles_all, sentences_all
