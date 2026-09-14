"""Internal cognitive-agent MCP server — localhost only (private), never tunneled.

Re-exports a fixed subset of the shared BaryGraph MCP tools by name so the
cognitive worker sees only what it needs:

    read:  context_search  find_word  word_senses  word_edges  semantic_search
           edge_info  traverse_up            (grounding / bridge verification)
    arXiv: arxiv_search  arxiv_get_paper     (BM25 on title+abstract, 3.1M papers)
    write: create_sense  create_word  create_edge  create_structure_meta_bary
           (SMB construction; default Author "cog:worker")

The tool schemas and bodies are the SAME registered functions from
scripts.mcp_server (one source of truth), filtered by name. Because this
server binds loopback only (MCP stdio) and is never tunneled, the write tools
that are stripped from the public server are safe to expose here.

Security boundary is network-level, not in-process: the FastMCP tool list is
reduced to *_requested_* just before startup, and the server refuses to run
over a non-stdio transport.

Usage (always STDIO):
    MCP_MONGO_DB=barygraph_poc \
      python3.11 -m scripts.mcp_int_cog
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

log = logging.getLogger("barygraph.cog")

# ── allowed tool surface ───────────────────────────────────────────────────────
_READ_ONLY = os.environ.get("MCP_READ_ONLY", "0").lower() in ("1", "true", "yes")

_ALLOWED_TOOLS = {
    # read-only grounding
    "context_search",
    "find_word",
    "word_senses",
    "word_edges",
    "semantic_search",
    "edge_info",
    "traverse_up",
    # arXiv corpus search (BM25 on title+abstract)
    "arxiv_search",
    "arxiv_get_paper",
    # extraction pipeline batch storage
    "store_paper_extraction",
}
if not _READ_ONLY:
    # SMB-building write tools (pipeline schema). Omitted when MCP_READ_ONLY=1.
    _ALLOWED_TOOLS |= {
        "create_sense",
        "create_word",
        "create_edge",
        "create_structure_meta_bary",
    }


def _setup_logging() -> None:
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)


# ── import shared engine (registers ALL tools at import-time) ──────────────────
import scripts.mcp_server as srv  # noqa: E402  (reuses engine, schemas, app)
from scripts.mcp_server import _run_thr  # noqa: E402  (shared thread runner)

mcp = srv.mcp  # same FastMCP instance; tools already registered on it


def _restrict_toolset() -> None:
    """Drop every FastMCP tool not in the allow-list so only the ones the cog
    worker should see are listed/executable."""
    fm = getattr(mcp, "_tool_manager", None)
    if fm is None:
        log.warning("no _tool_manager on shared mcp — cannot restrict; exposing full set")
        return
    existing = set(fm._tools.keys()) if hasattr(fm, "_tools") else set()
    to_remove = existing - _ALLOWED_TOOLS
    for name in sorted(to_remove):
        try:
            fm.remove_tool(name)
        except Exception as e:  # pragma: no cover
            log.warning("failed to remove tool %s: %s", name, e)
    kept = existing & _ALLOWED_TOOLS
    log.info(
        "toolset filtered: %d -> %d tools | kept=%s removed=%s",
        len(existing), len(kept), sorted(kept), sorted(to_remove),
    )


def _warmup() -> None:
    try:
        srv._warmup_engine()
    except Exception as e:  # pragma: no cover
        log.warning("cognitive warmup failed: %s", e)


# ── arXiv corpus tools (BM25 on title+abstract) ───────────────────────────────
import pymongo  # noqa: E402

_ARXIV_COLLECTION = os.environ.get("ARXIV_COLLECTION", "cognitive_arxiv")
_arxiv_coll = None


def _get_arxiv_coll():
    global _arxiv_coll
    if _arxiv_coll is None:
        client = pymongo.MongoClient(
            os.environ.get("MONGO_URI", "mongodb://mongodb:27017/?directConnection=true"),
            serverSelectionTimeoutMS=5_000,
        )
        _arxiv_coll = client[os.environ.get("MCP_MONGO_DB", "barygraph_poc")][
            _ARXIV_COLLECTION
        ]
    return _arxiv_coll


def _arxiv_search_body(query: str, top_k: int = 5) -> str:
    """BM25 search on arXiv paper titles and abstracts."""
    import json as _json

    if not query or not query.strip():
        return "query must be non-empty"
    top_k = max(1, min(top_k, 20))

    coll = _get_arxiv_coll()
    try:
        cursor = coll.find(
            {"$text": {"$search": query}},
            {"arxiv_id": 1, "title": 1, "abstract": 1,
             "doi": 1, "categories": 1, "authors": 1},
        ).limit(top_k)
        results = list(cursor)
    except pymongo.errors.OperationFailure as e:
        return f"search failed (is the text index built?): {e}"

    for r in results:
        r["_id"] = str(r["_id"])
        r["abstract"] = r.get("abstract", "")[:500]
    return _json.dumps(results, indent=2, default=str)


def _arxiv_get_body(arxiv_id: str) -> str:
    """Fetch full metadata for an arXiv paper by ID."""
    import json as _json

    if not arxiv_id or not arxiv_id.strip():
        return "arxiv_id must be non-empty"

    coll = _get_arxiv_coll()
    doc = coll.find_one({"arxiv_id": arxiv_id.strip()})
    if doc is None:
        return f"paper not found: {arxiv_id}"
    doc["_id"] = str(doc["_id"])
    for _k in ("vector", "embedding"):
        doc.pop(_k, None)
    return _json.dumps(doc, indent=2, default=str)


@mcp.tool()
async def arxiv_search(query: str, top_k: int = 5) -> str:
    """Search the arXiv corpus (3.1M papers) by title and abstract text.

    Returns top_k results ranked by BM25 relevance score. Each result
    includes arxiv_id, title, abstract (truncated), doi, categories, authors.
    Use this to find papers on a topic before extracting terms for BG grounding.
    """
    return await _run_thr(_arxiv_search_body, query, top_k)


@mcp.tool()
async def arxiv_get_paper(arxiv_id: str) -> str:
    """Fetch full metadata for a single arXiv paper by its ID (e.g. '2301.07041').

    Returns title, abstract, doi, categories, authors, and vector dimensions.
    Use after arxiv_search to read the full abstract before term extraction.
    """
    return await _run_thr(_arxiv_get_body, arxiv_id)


# ── batch storage (extraction pipeline) ────────────────────────────────────────
import json as _json_batch  # noqa: E402
import re as _re  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_BATCH_DIR = _Path(os.environ.get(
    "COGNITIVE_BATCH_DIR",
    _Path(__file__).resolve().parent.parent / "cognitive" / "batches",
))
_TERMS_FILE = _BATCH_DIR / "terms_batch.jsonl"
_PAPERS_FILE = _BATCH_DIR / "papers_batch.jsonl"


def _load_existing_dois(path: _Path) -> set[str]:
    """Read existing DOIs from a JSONL file for dedup."""
    dois: set[str] = set()
    if not path.exists():
        return dois
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = _json_batch.loads(line)
            if rec.get("doi"):
                dois.add(rec["doi"])
        except _json_batch.JSONDecodeError:
            pass
    return dois


def _store_extraction_body(
    doi: str,
    terms_json: str,
    arxiv_id: str = "",
    title: str = "",
    abstract: str = "",
    authors: str = "",
    categories: str = "",
) -> str:
    """Store extracted terms and paper metadata into JSONL batch files.

    Deduplicates by DOI: if a DOI already exists in either file, the store
    is skipped and a message is returned.
    """
    if not doi or not doi.strip():
        return "doi must be non-empty"

    doi = doi.strip()

    # a DOI must look like a DOI — reject arxiv IDs or other bare strings
    if not _re.match(r"^10\.\S+$", doi):
        return f"'{doi}' is not a valid DOI (must start with '10.'). If the paper is arXiv-only, use the 10.48550/arXiv.<id> form or skip it."

    _BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # ensure the file ends with a newline so appends never glue onto a
    # previous record (hand-written files or legacy records may lack one)
    def _ensure_trailing_newline(path: _Path) -> None:
        if path.exists() and path.stat().st_size > 0:
            data = path.read_bytes()
            if not data.endswith(b"\n"):
                with path.open("ab") as fh:
                    fh.write(b"\n")

    _ensure_trailing_newline(_TERMS_FILE)
    _ensure_trailing_newline(_PAPERS_FILE)

    # dedup check
    existing = _load_existing_dois(_TERMS_FILE) | _load_existing_dois(_PAPERS_FILE)
    if doi in existing:
        return f"doi {doi} already in batch files — skipped"

    # parse terms
    try:
        terms = _json_batch.loads(terms_json)
        if not isinstance(terms, list):
            return "terms_json must be a JSON array"
    except _json_batch.JSONDecodeError as e:
        return f"invalid terms_json: {e}"

    # write terms
    terms_rec = {"doi": doi, "terms": terms}
    with _TERMS_FILE.open("a") as fh:
        fh.write(_json_batch.dumps(terms_rec, ensure_ascii=False) + "\n")

    # write paper metadata
    paper_rec = {
        "doi": doi,
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract[:5000],  # cap abstract length
        "authors": authors,
        "categories": categories,
    }
    with _PAPERS_FILE.open("a") as fh:
        fh.write(_json_batch.dumps(paper_rec, ensure_ascii=False) + "\n")

    n_terms = len(terms)
    log.info("stored extraction: doi=%s terms=%d", doi, n_terms)
    return f"stored doi={doi} with {n_terms} terms"


@mcp.tool()
async def store_paper_extraction(
    doi: str,
    terms_json: str,
    arxiv_id: str = "",
    title: str = "",
    abstract: str = "",
    authors: str = "",
    categories: str = "",
) -> str:
    """Store extracted terms and paper metadata into JSONL batch files.

    This is the first step in the cognitive pipeline: after extracting terms
    from a paper's abstract (using the extraction prompt), call this tool to
    store the results. The terms file (terms_batch.jsonl) and paper metadata
    file (papers_batch.jsonl) are used by downstream grounding steps.

    Args:
        doi: Paper DOI (bare, no https://doi.org/ prefix)
        terms_json: JSON array of extracted terms, each with "id", "term", "gloss"
        arxiv_id: Optional arXiv ID (e.g. '2301.07041')
        title: Optional paper title
        abstract: Optional paper abstract
        authors: Optional authors string
        categories: Optional categories string
    """
    return await _run_thr(
        _store_extraction_body,
        doi, terms_json, arxiv_id, title, abstract, authors, categories,
    )


def main() -> int:
    _setup_logging()
    db = os.environ.get("MCP_MONGO_DB", "barygraph_poc")
    log.info(
        "cognitive MCP starting | db=%s read_only=%s tools=%d",
        db, _READ_ONLY, len(_ALLOWED_TOOLS),
    )
    _warmup()
    _restrict_toolset()
    mcp.run(transport="stdio")
    return 0  # not reached


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BaryGraph cognitive-agent MCP (internal)")
    ap.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="STDIO by default — this internal server must NOT be tunneled.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    if args.transport != "stdio":
        sys.stderr.write(
            "cognitive MCP refuses non-stdio transport: internal-only server.\n"
        )
        sys.exit(2)
    main()
