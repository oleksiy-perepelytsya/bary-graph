from __future__ import annotations

import pytest

from tensegrity_pipeline.parse import (
    is_title_line,
    parse_document,
    parse_documents,
    split_sentences,
    split_sentences_with_spans,
)


def test_split_sentences_basic():
    text = "First sentence. Second one! Third? Fourth without punctuation"
    assert split_sentences(text) == [
        "First sentence.",
        "Second one!",
        "Third?",
        "Fourth without punctuation",
    ]


def test_split_sentences_keeps_abbreviations():
    text = "Dr. Smith went home. He was tired."
    assert split_sentences(text) == ["Dr. Smith went home.", "He was tired."]


def test_split_sentences_handles_acronyms_and_quotes():
    text = 'The U.S. is large. She said "go." Then left.'
    out = split_sentences(text)
    assert out[0] == "The U.S. is large."
    assert out[1] == 'She said "go."'
    assert out[2] == "Then left."


def test_split_sentences_with_spans_offsets_align():
    text = "Alpha beta. Gamma delta."
    spans = split_sentences_with_spans(text)
    assert [s.text for s in spans] == ["Alpha beta.", "Gamma delta."]
    for s in spans:
        assert text[s.start : s.end] == s.text


def test_is_title_line_markdown_and_numbered_and_caps_and_case():
    assert is_title_line("# Introduction")
    assert is_title_line("## Deep Dive")
    assert is_title_line("3.2 Methods")
    assert is_title_line("THE QUICK BROWN FOX")
    assert is_title_line("A Study Of Models")


def test_is_title_line_rejects_sentences():
    assert not is_title_line("This is a normal sentence.")
    assert not is_title_line("")
    assert not is_title_line(
        "A rather long line of prose that keeps going and going until it cannot be a heading."
    )


def test_is_title_line_terminal_punct_guard_actually_fires():
    # Regression: _TERMINAL_PUNCT used to be a plain string handed to
    # str.endswith(), which matches only the literal suffix ".!?,;:" — the
    # guard never fired and body lines streamed through as titles.
    assert not is_title_line("State v. Spaulding,")
    assert not is_title_line("2011 MT 204, 361 Mont. 445, 259 P.3d 793;")
    assert not is_title_line("ANDREW J. GUILFORD, District Judge.")
    # Numbered body sentences ending in punctuation are rejected too…
    assert not is_title_line("1. In this appeal the claim fails.")
    # …while real headings still pass.
    assert is_title_line("STANDARD OF REVIEW")
    assert is_title_line("3.2 Methods")


def test_split_sentences_keeps_legal_citations_intact():
    text = "See id. at 413. The court affirmed. State v. Lovegren (Tex. App. 2019)."
    assert split_sentences(text) == [
        "See id. at 413.",
        "The court affirmed.",
        "State v. Lovegren (Tex. App. 2019).",
    ]
    assert split_sentences("affirmed (4th Cir. 2019). The petition was denied.") == [
        "affirmed (4th Cir. 2019).",
        "The petition was denied.",
    ]
    # Openers ride along with the token, so "(U.S." must still be recognised
    # as an acronym (it previously split at the second period).
    assert split_sentences("relying on (U.S. Const.). Denied.") == [
        "relying on (U.S. Const.).",
        "Denied.",
    ]


def test_split_sentences_compound_reporters_and_section_numerals():
    # Dotted compounds written without spaces: the last component carries the
    # abbreviation, so the final period must not split ("F.Supp." → "supp").
    assert split_sentences("913 F.Supp. 256 (1996). Next case.") == [
        "913 F.Supp. 256 (1996).",
        "Next case.",
    ]
    assert split_sentences("99 S.Ct. 2106 (citations omitted). Next.") == [
        "99 S.Ct. 2106 (citations omitted).",
        "Next.",
    ]
    # Roman section numerals are crumbs, not sentence ends.
    assert split_sentences("II. Ineffective Assistance. In so much as argued.") == [
        "II. Ineffective Assistance.",
        "In so much as argued.",
    ]
    # Pleading/brief furniture.
    assert split_sentences("affirmed. See Pet. for Cert. in No. 93-108.") == [
        "affirmed.",
        "See Pet. for Cert. in No. 93-108.",
    ]


def test_split_sentences_ellipsis_never_ends_a_sentence():
    assert split_sentences("The record is silent . . . and stays so. Next one.") == [
        "The record is silent . . . and stays so.",
        "Next one.",
    ]
    assert split_sentences("held... then denied. Done.") == [
        "held... then denied.",
        "Done.",
    ]


def test_split_sentences_drops_letterless_segments():
    # Star-pagination, footnote numbers and bare punctuation are markers, not
    # sentences: they would become junk L15 nodes (and junk edge candidates).
    assert split_sentences("173. Real content. *259") == ["Real content."]
    assert split_sentences("*173") == []
    assert split_sentences(".....") == []


def test_split_sentences_with_spans_keeps_offsets_after_drops():
    text = "173. Real content stays aligned. *259"
    spans = split_sentences_with_spans(text)
    assert [s.text for s in spans] == ["Real content stays aligned."]
    for s in spans:
        assert text[s.start : s.end] == s.text


FIXTURE_DOC = """\
# Neural Networks

Neural networks are trained with backpropagation.
The ability to reason is not required for learning.

## Compression

Compression arises when representations are shared.
Shared representations reduce redundancy.
"""


def test_parse_document_sections_and_sentence_membership():
    titles, sentences = parse_document(FIXTURE_DOC, "doc1")

    assert [t.text for t in titles] == ["Neural Networks", "Compression"]
    assert titles[0].title_level == 1
    assert titles[1].title_level == 2
    # Sentences carry the enclosing section index, not the other title's.
    assert sentences[0].section_idx == 0
    assert sentences[1].section_idx == 0
    assert sentences[2].section_idx == 1
    assert sentences[3].section_idx == 1
    assert len(sentences) == 4
    # Stable ids.
    assert sentences[0].sentence_id == "doc1:p0:s0"
    assert titles[1].title_id == "doc1:t1"


def test_parse_document_offsets_point_into_source():
    _, sentences = parse_document(FIXTURE_DOC, "doc1")
    for s in sentences:
        assert FIXTURE_DOC[s.char_start : s.char_end] == s.text


def test_parse_document_sentence_before_any_title_has_section_minus_one():
    titles, sentences = parse_document("Loose sentence first. Then more.", "d")
    assert titles == []
    assert sentences[0].section_idx == -1


def test_parse_document_setext_heading():
    titles, sentences = parse_document("Overview\n========\n\nBody text here.", "d")
    assert titles[0].text == "Overview"
    assert sentences[0].text == "Body text here."


def test_parse_document_strips_heading_markers():
    titles, _ = parse_document("# Title\n\nBody.\n", "d")
    assert titles[0].text == "Title"
    assert titles[0].title_level == 1


def test_parse_documents_preserves_doc_identity_and_order():
    titles, sentences = parse_documents(
        [
            ("P01", FIXTURE_DOC),
            ("P02", "Supplier B is recommended.\nThis note is advisory."),
        ]
    )
    # Doc order preserved, per-doc identity intact.
    assert [s.doc_id for s in sentences] == ["P01"] * 4 + ["P02"] * 2
    assert [t.doc_id for t in titles] == ["P01", "P01"]
    # Sentence ids stay unique across documents.
    ids = [s.sentence_id for s in sentences]
    assert len(ids) == len(set(ids))


def test_parse_documents_duplicate_doc_id_raises():
    with pytest.raises(ValueError, match="duplicate doc_id 'P01'"):
        parse_documents([("P01", "text one."), ("P01", "text two.")])


def test_parse_documents_empty_input_is_empty():
    titles, sentences = parse_documents([])
    assert titles == []
    assert sentences == []
