"""Merge/dedup decisions for academic-batch ingestion (DB-touching).

Three questions this module answers, each backed by a design decision made
with the user before implementation:

1. Do two term occurrences (possibly from different papers) describe the
   same sense? -- ``dedupe_exact`` (string-identical glosses, pre-embed) and
   ``near_duplicate_sense`` (cosine-identical glosses, post-embed — catches
   reworded duplicates like "Initialism of N-methyldiethanolamine." vs
   "Initialism of methyl diethanolamine." for the same term "MDEA").
2. Does a term's spelling already exist as a kaikki word? --
   ``find_existing_word_candidates``.
3. If it exists under multiple POS, which one is this term? --
   ``resolve_pos``, cosine-nearest by word vector (falling back to the mean
   of the candidate's current sense vectors if its word vector hasn't been
   computed yet by s05).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymongo.collection import Collection

from lib.bary_vec import normalize
from lib.parse_batch import normalize_term
from lib.schema import ParsedSense, ParsedWord

Entry = tuple[ParsedWord, ParsedSense]


def dedupe_exact(entries: list[Entry]) -> list[Entry]:
    """Collapse term occurrences with identical (normalized term, gloss).

    Unions their doi lists onto the first occurrence kept; does not touch
    genuinely reworded duplicates (different gloss text) — those are left
    for the embedding-based near_duplicate_sense check downstream.
    """
    seen: dict[tuple[str, str], int] = {}
    out: list[Entry] = []
    for pw, ps in entries:
        key = (normalize_term(ps.word), ps.gloss)
        idx = seen.get(key)
        if idx is not None:
            _, existing_ps = out[idx]
            for d in ps.doi:
                if d not in existing_ps.doi:
                    existing_ps.doi.append(d)
            continue
        seen[key] = len(out)
        out.append((pw, ps))
    return out


def find_existing_word_candidates(coll: Collection, term: str) -> list[dict[str, Any]]:
    """Point lookup on properties.word: as-is first, then lowercased.

    Both hit the existing (properties.word, properties.pos) index — no new
    index or backfill needed. Multi-word phrase terms are expected to find
    nothing; that's the common case for this corpus (most terms are noun
    phrases with no kaikki entry).
    """
    docs = list(
        coll.find({"doc_type": "node", "node_type": "word", "properties.word": term})
    )
    if docs:
        return docs
    lower = term.lower()
    if lower != term:
        docs = list(
            coll.find({"doc_type": "node", "node_type": "word", "properties.word": lower})
        )
    return docs


def _word_vector_estimate(coll: Collection, candidate: dict[str, Any]) -> np.ndarray | None:
    vec = candidate.get("vector")
    if vec is not None:
        return np.asarray(vec, dtype=np.float32)
    # Word created by an earlier, not-yet-s05'd batch: fall back to the mean
    # of its current senses' vectors.
    props = candidate["properties"]
    sense_vecs = [
        np.asarray(s["vector"], dtype=np.float32)
        for s in coll.find(
            {
                "doc_type": "node",
                "node_type": "sense",
                "properties.word": props["word"],
                "properties.pos": props["pos"],
            },
            {"vector": 1},
        )
        if s.get("vector") is not None
    ]
    if not sense_vecs:
        return None
    return normalize(np.mean(np.stack(sense_vecs), axis=0))


def resolve_pos(
    coll: Collection, gloss_vec: np.ndarray, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Pick the candidate whose word vector is cosine-nearest to gloss_vec.

    A single candidate is returned directly (no embedding comparison
    needed) — this only does real work on a genuine spelling collision
    across multiple POS entries.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    gloss_vec = np.asarray(gloss_vec, dtype=np.float32)
    best: dict[str, Any] | None = None
    best_cos = -2.0
    for cand in candidates:
        cvec = _word_vector_estimate(coll, cand)
        if cvec is None:
            continue
        cos = float(np.dot(gloss_vec, cvec))
        if cos > best_cos:
            best_cos = cos
            best = cand
    return best


def near_duplicate_sense(
    coll: Collection, word: str, pos: str, gloss_vec: np.ndarray, threshold: float
) -> Any | None:
    """Cosine-nearest existing sense of (word, pos); returns its _id if the
    match is at or above ``threshold``, else None (create a new sense)."""
    gloss_vec = np.asarray(gloss_vec, dtype=np.float32)
    best_id = None
    best_cos = -2.0
    for s in coll.find(
        {
            "doc_type": "node",
            "node_type": "sense",
            "properties.word": word,
            "properties.pos": pos,
        },
        {"vector": 1},
    ):
        v = s.get("vector")
        if v is None:
            continue
        cos = float(np.dot(gloss_vec, np.asarray(v, dtype=np.float32)))
        if cos > best_cos:
            best_cos = cos
            best_id = s["_id"]
    if best_id is not None and best_cos >= threshold:
        return best_id
    return None
