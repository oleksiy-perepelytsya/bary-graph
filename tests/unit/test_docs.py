from __future__ import annotations

import numpy as np

from lib.docs import baryedge, metabary, sense_node, word_node
from lib.schema import ParsedSense, ParsedWord


def test_sense_node_schema():
    s = ParsedSense(word="cat", pos="noun", sense_id="cat-noun-0", sense_idx=0,
                    gloss="a feline")
    doc = sense_node(s, np.zeros(768, dtype=np.float32))
    assert doc["doc_type"] == "node" and doc["node_type"] == "sense"
    assert doc["level"] == 15 and doc["parent_edge_id"] is None
    assert doc["properties"]["sense_id"] == "cat-noun-0"
    assert doc["properties"]["lang"] == "en"
    assert len(doc["vector"]) == 768 * 4  # float32 packed bytes


def test_word_node_placeholder_vector():
    w = ParsedWord(word="cat", pos="noun", lang_code="en", sense_ids=["a", "b"])
    doc = word_node(w)
    assert doc["level"] == 14 and doc["vector"] is None
    assert "char_len" not in doc["properties"]
    assert "syllable_ct" not in doc["properties"]
    assert doc["properties"]["lang"] == "en"
    assert doc["surface"] == 2


def test_baryedge_l14_carries_type_fields():
    v = np.zeros(768, dtype=np.float32)
    doc = baryedge("a", "b", 14, v, 0.85, edge_type="contradicts", type_vector=v,
                   source="ingested")
    assert doc["doc_type"] == "baryedge" and doc["level"] == 14
    assert doc["q"] == doc["connection_strength"] == 0.85
    assert doc["accumulated_weight"] == 0.85
    assert doc["edge_type"] == "contradicts"
    assert "type_vector" in doc


def test_baryedge_explicit_accumulated_weight():
    v = np.zeros(768, dtype=np.float32)
    doc = baryedge("a", "b", 15, v, 0.80, accumulated_weight=0.80)
    assert doc["accumulated_weight"] == 0.80
    assert doc["connection_strength"] == 0.80


def test_metabary_drops_leaf_only_fields():
    v = np.zeros(768, dtype=np.float32)
    doc = metabary("a", "b", 11, v, 0.42, 0.63)
    assert doc["doc_type"] == "baryedge" and doc["level"] == 11
    assert doc["connection_strength"] == 0.42
    assert doc["accumulated_weight"] == 0.63
    for f in ("edge_type", "type_vector", "q", "source", "confidence"):
        assert f not in doc
