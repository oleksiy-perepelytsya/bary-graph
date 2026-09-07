"""Run the five retrieval arms over barygraph_poc for the 30-query suite.

Arms:
  raw_bm25   - BM25 over paper title+abstract (no graph at all)
  nodes_bm25 - BM25 over ingested word/sense text (flat schema)
  nodes_vec  - vector search, node space
  edges_vec  - vector search, baryedge+MB space
  graph_vec  - nodes+edges merged by score

Writes one JSONL per arm under evaluation/results/ (ranked hits with DOIs).
Read-only; a single batched embed call (~30 queries) against the shared Ollama.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from lib.config import Settings
from lib.db import vector_search
from lib.embed import get_embedder
from lib.log import get_logger, setup_logging
from scripts.eval.academic_common import (
    BARYEDGE_FILTER,
    K_MAX,
    NODE_FILTER,
    NUM_CANDIDATES,
    RESULTS_DIR,
    STOP,
    TOKEN_RE,
    load_rows,
    norm_doi,
    require_poc,
    reverse_doi_index,
    to_str,
)
from scripts.eval.build_queries import build

N_EST = 1_600_000
K1 = 1.2
B = 0.75


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if len(t) > 1 and t not in STOP]


def score_all(
    qterms: list[str],
    idf: dict[str, float],
    postings: dict[str, dict[str, int]],
    dl: dict[str, int],
    avgdl: float,
) -> list[tuple[float, str]]:
    acc: dict[str, float] = {}
    for t in qterms:
        idf_t = idf.get(t)
        if idf_t is None:
            continue
        for oid, tf in postings.get(t, {}).items():
            denom = tf + K1 * (1 - B + B * (dl[oid] / avgdl))
            acc[oid] = acc.get(oid, 0.0) + idf_t * (tf * (K1 + 1)) / denom
    return sorted(((s, oid) for oid, s in acc.items()), reverse=True)[:K_MAX]


def main() -> None:
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger("eval.academic.run")
    require_poc(settings)

    build()
    suite = json.loads(RESULTS_DIR.parent.joinpath("queries.json").read_text())
    query_recs = [(p["problem"], q["qid"], q["text"]) for p in suite for q in p["queries"]]
    log.info("suite: %d problems x 5 queries = %d retrieval queries", len(suite), len(query_recs))

    import pymongo

    client = pymongo.MongoClient(settings.mongo_uri, socketTimeoutMS=600000)
    db = client[settings.mongo_db]
    coll = db[settings.mongo_collection]
    bridge = db[settings.mongo_doi_bridges_collection]
    reverse = reverse_doi_index(bridge)

    def dois_for(oid: str) -> list[str]:
        return sorted(reverse.get(oid, set()))

    seen_terms: set[str] = set()
    for _, _, text in query_recs:
        seen_terms.update(tokens(text))
    union_terms = sorted(seen_terms)
    log.info("indexing node space: %d union terms, single gloss scan", len(union_terms))

    postings: dict[str, dict[str, int]] = {t: {} for t in union_terms}
    df: dict[str, int] = {t: 0 for t in union_terms}
    dl: dict[str, int] = {}
    avgdl = 1.0
    dl_sum = dl_n = 0
    gloss_or = re.compile(rf"\b(?:{'|'.join(re.escape(t) for t in union_terms)})\b", re.I)
    for doc in coll.find(
        {"doc_type": "node", "node_type": "sense", "properties.gloss": gloss_or},
        {"properties.gloss": 1},
        batch_size=5000,
    ):
        oid = to_str(doc["_id"])
        gloss_toks = tokens(doc.get("properties", {}).get("gloss") or "")
        hit = [t for t in union_terms if t in gloss_toks]
        if not hit:
            continue
        dl[oid] = len(gloss_toks) or 1
        dl_sum += dl[oid]
        dl_n += 1
        for t in hit:
            postings[t][oid] = gloss_toks.count(t)
            df[t] += 1
    for t in union_terms:
        if df[t]:
            continue
        for wdoc in coll.find(
            {
                "doc_type": "node",
                "node_type": "word",
                "properties.word": re.compile(rf"^{re.escape(t)}$", re.I),
            },
            {"_id": 1},
        ):
            oid = to_str(wdoc["_id"])
            postings[t][oid] = 1
            df[t] += 1
            dl[oid] = 1
    if dl_n:
        avgdl = dl_sum / dl_n
    idf = {t: math.log(1 + (N_EST - df[t] + 0.5) / (df[t] + 0.5)) for t in union_terms if df[t]}
    log.info(
        "node index: %d candidate docs, avgdl=%.2f, terms_with_df=%d", len(dl), avgdl, len(idf)
    )

    rows = load_rows()
    paper_tokens: dict[str, list[str]] = {}
    for r in rows:
        d = norm_doi(r.get("doi"))
        if not d or d in paper_tokens:
            continue
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        paper_tokens[d] = tokens(title + " " + abstract)
    p_df: Counter = Counter()
    p_dl: dict[str, int] = {}
    for d, toks in paper_tokens.items():
        p_dl[d] = len(toks) or 1
        for t in set(toks):
            p_df[t] += 1
    p_N = len(paper_tokens)
    p_avgdl = sum(p_dl.values()) / p_N
    p_postings: dict[str, dict[str, int]] = {}
    for d, toks in paper_tokens.items():
        cnt = Counter(toks)
        for t in set(toks):
            p_postings.setdefault(t, {})[d] = cnt[t]
    p_idf = {t: math.log(1 + (p_N - v + 0.5) / (v + 0.5)) for t, v in p_df.items()}
    log.info("paper corpus: %d unique DOIs, avgdl=%.1f, vocabulary=%d", p_N, p_avgdl, len(p_df))

    embedder = get_embedder(settings)
    uniq_texts = list(dict.fromkeys(text for _, _, text in query_recs))
    log.info("embedding %d unique query texts (single batched call)", len(uniq_texts))
    vecs = embedder.embed(uniq_texts)
    text_vec = dict(zip(uniq_texts, [v.tolist() for v in vecs], strict=True))

    arms = {a: [] for a in ["raw_bm25", "nodes_bm25", "nodes_vec", "edges_vec", "graph_vec"]}
    for problem, qid, text in query_recs:
        qterms = tokens(text)
        hits_bm25 = score_all(qterms, idf, postings, dl, avgdl)
        hits_raw = score_all(qterms, p_idf, p_postings, p_dl, p_avgdl)
        qv = text_vec[text]
        nv = vector_search(coll, qv, limit=K_MAX, num_candidates=NUM_CANDIDATES, filter=NODE_FILTER)
        ev = vector_search(
            coll, qv, limit=K_MAX, num_candidates=NUM_CANDIDATES, filter=BARYEDGE_FILTER
        )

        def hit(doc: dict, space: str) -> dict:
            return {
                "id": to_str(doc["_id"]),
                "space": space,
                "score": round(float(doc["_score"]), 6),
                "word": (doc.get("properties") or {}).get("word", ""),
                "level": doc.get("level"),
                "dois": dois_for(to_str(doc["_id"])),
            }

        nodes_bm25_hits = [
            {
                "rank": i + 1,
                "id": oid,
                "space": "node",
                "score": round(float(s), 6),
                "word": "",
                "level": 15,
                "dois": dois_for(oid),
            }
            for i, (s, oid) in enumerate(hits_bm25)
        ]
        raw_hits = [
            {
                "rank": i + 1,
                "id": oid,
                "space": "paper",
                "score": round(float(s), 6),
                "word": "",
                "level": None,
                "dois": [oid],
            }
            for i, (s, oid) in enumerate(hits_raw)
        ]
        node_vec = [
            dict(hit(d, "word" if (d.get("node_type") == "word") else "sense"), rank=i + 1)
            for i, d in enumerate(nv)
        ]
        edge_vec = [dict(hit(d, "edge"), rank=i + 1) for i, d in enumerate(ev)]
        merged = sorted(node_vec + edge_vec, key=lambda h: h["score"], reverse=True)[:K_MAX]
        graph_vec = [dict(h, rank=i + 1) for i, h in enumerate(merged)]

        rec = {"problem": problem, "query_id": qid, "text": text}
        rec["hits"] = nodes_bm25_hits
        arms["nodes_bm25"].append(rec)
        arms["raw_bm25"].append({**rec, "hits": raw_hits})
        arms["nodes_vec"].append({**rec, "hits": node_vec})
        arms["edges_vec"].append({**rec, "hits": edge_vec})
        arms["graph_vec"].append({**rec, "hits": graph_vec})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for arm, recs in arms.items():
        with open(RESULTS_DIR / f"{arm}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("wrote %s.jsonl (%d records)", arm, len(recs))
    log.info("done")


if __name__ == "__main__":
    main()
