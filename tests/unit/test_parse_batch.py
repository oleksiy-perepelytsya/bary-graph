from __future__ import annotations

from lib.batch_merge import dedupe_exact
from lib.parse_batch import BATCH_POS, normalize_term, parse_batch_entry, parse_batch_file

FIXTURE = "tests/fixtures/batch-sample.jsonl"


def test_parse_batch_entry_basic():
    obj = {
        "doi": "10.1/x",
        "terms": [{"id": "t01", "term": "flue gas", "gloss": "exhaust stream"}],
    }
    out = parse_batch_entry(obj)
    assert len(out) == 1
    pw, ps = out[0]
    assert pw.word == ps.word == "flue gas"
    assert pw.pos == ps.pos == BATCH_POS
    assert ps.sense_idx == 0
    assert ps.embed_text == "exhaust stream"
    assert ps.doi == ["10.1/x"]
    assert ps.sense_id == "batch:10.1/x:t01"
    assert pw.sense_ids == [ps.sense_id]


def test_parse_batch_entry_null_doi():
    obj = {"doi": None, "terms": [{"id": "t01", "term": "viscosity", "gloss": "g"}]}
    _, ps = parse_batch_entry(obj)[0]
    assert ps.doi == []
    assert ps.sense_id == "batch:nodoi:t01"


def test_parse_batch_entry_skips_incomplete_terms():
    obj = {"doi": "d", "terms": [{"id": "t01", "term": "", "gloss": "g"},
                                  {"id": "t02", "term": "x", "gloss": ""},
                                  {"term": "x", "gloss": "g"}]}
    assert parse_batch_entry(obj) == []


def test_normalize_term_collapses_whitespace_preserves_case():
    assert normalize_term("  CO  2   capture  ") == "CO 2 capture"
    assert normalize_term("MDEA") == "MDEA"


def test_parse_batch_file_reads_all_records():
    entries = parse_batch_file(FIXTURE)
    # 2 + 1 + 1 + 1 + 1 = 6 term occurrences across 5 lines
    assert len(entries) == 6


def test_dedupe_exact_merges_identical_term_and_gloss():
    entries = parse_batch_file(FIXTURE)
    deduped = dedupe_exact(entries)
    ccs = [ps for _, ps in deduped if normalize_term(ps.word) == "post-combustion carbon capture"]
    assert len(ccs) == 1
    assert set(ccs[0].doi) == {"10.1016/j.rineng.2024.103574", "10.1063/5.0169382"}


def test_dedupe_exact_does_not_merge_reworded_duplicates():
    """MDEA appears twice with different (but equivalent) gloss text — exact
    dedup must NOT merge these; that's the embedding-similarity step's job."""
    entries = parse_batch_file(FIXTURE)
    deduped = dedupe_exact(entries)
    mdeas = [ps for _, ps in deduped if ps.word == "MDEA"]
    assert len(mdeas) == 2
    assert {ps.gloss for ps in mdeas} == {
        "Initialism of methyl diethanolamine.",
        "Initialism of N-methyldiethanolamine.",
    }


def test_dedupe_exact_preserves_unrelated_terms():
    entries = parse_batch_file(FIXTURE)
    deduped = dedupe_exact(entries)
    words = {ps.word for _, ps in deduped}
    assert "flue gas" in words
    assert "viscosity" in words
