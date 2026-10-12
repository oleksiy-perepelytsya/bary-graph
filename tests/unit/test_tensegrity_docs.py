from __future__ import annotations

import numpy as np

from tensegrity_pipeline.docs import (
    metabary,
    sentence_node,
    tensegrity_baryedge,
    title_node,
)
from tensegrity_pipeline.schema import ParsedSentence, ParsedTitle
from tensegrity_pipeline.tg_vec import COMPLEMENT, CONTRADICT


def _vec(dim: int = 4) -> np.ndarray:
    v = np.arange(dim, dtype=np.float32) + 1.0
    return v / np.linalg.norm(v)


def test_sentence_node_shape():
    s = ParsedSentence(
        doc_id="d", paragraph_idx=0, sentence_idx=0, text="hello world",
        section_idx=1, char_start=0, char_end=11, embed_text="hello world",
    )
    doc = sentence_node(s, _vec())
    assert doc["doc_type"] == "node"
    assert doc["node_type"] == "sentence"
    assert doc["level"] == 15
    assert doc["label"] == "d:p0:s0"
    assert doc["properties"]["section_idx"] == 1
    assert doc["vector"] is not None
    assert doc["parent_edge_id"] is None


def test_title_node_has_placeholder_vector():
    t = ParsedTitle(doc_id="d", section_idx=2, text="Methods", title_level=2)
    doc = title_node(t)
    assert doc["node_type"] == "title"
    assert doc["level"] == 14
    assert doc["vector"] is None
    assert doc["properties"]["title_id"] == "d:t2"


def test_tensegrity_baryedge_carries_role_fields():
    doc = tensegrity_baryedge(
        "a", "b", 15, _vec(), 0.83,
        role=CONTRADICT, role_probability=0.91, role_source="tev1",
        type_vector=_vec(),
    )
    assert doc["doc_type"] == "baryedge"
    assert doc["level"] == 15
    assert doc["role"] == CONTRADICT
    assert doc["role_probability"] == 0.91
    assert doc["role_source"] == "tev1"
    assert doc["q"] == 0.83
    assert doc["connection_strength"] == 0.83
    assert doc["vector"] is not None
    assert doc["type_vector"] is not None


def test_unclassified_baryedge_has_none_role_and_zero_probability():
    doc = tensegrity_baryedge("a", "b", 14, _vec(), 0.5, role=None)
    assert doc["role"] is None
    assert doc["role_probability"] == 0.0
    assert doc["role_source"] == "unclassified"


def test_metabary_shape():
    doc = metabary("x", "y", 13, _vec(), 0.6, 0.72, role=COMPLEMENT,
                   role_probability=0.77, role_source="tev1")
    assert doc["level"] == 13
    assert doc["connection_strength"] == 0.6
    assert doc["accumulated_weight"] == 0.72
    assert doc["role"] == COMPLEMENT
    assert doc["role_probability"] == 0.77
