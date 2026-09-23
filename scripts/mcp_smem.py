"""Session-memory MCP server — multilingual BaryGraph (barygraph_all) + working memory pool.

A DEDICATED internal server (never tunneled) adapted from mcp_int_cog.py. It targets
the in-progress all-languages build:

    MONGO_DB=barygraph_all   EMBED_MODEL=qwen3-embedding:8b   EMBED_DIM=4096

so every shared read tool it re-exports grounds against the multilingual graph. The
MetaBary/vector-search tools (sample_metabary, semantic_search, enrich_request,
context_search) only return data once s08/s09/s10 of the all build complete — until
then they fail gracefully with a self-explanatory error; the node/BE tools
(find_word, word_senses, word_edges, edge_info, traverse_up, leaf_nodes) work now.

On top of the shared tools this server adds a small WORKING-MEMORY POOL for the
cognitive agent:

    mnemo_remember  — record/reinforce a coordinate (id-dedup = reinforcement)
    mnemo_recall    — top-k active coordinates from the session, recency-decayed
    mnemo_status    — pool summary (session, size, top entries, store path)
    mnemo_reset     — tombstone a session so earlier remembers no longer resolve

Pool mechanics: an append-only JSONL (durable artifact, one line per record, trailing
newline enforced — same discipline as the batch files). Each session+coordinate key
collapses to one record whose `strength` increments on every remember; recall scores
by strength * recency-decay (half-life configurable) with a text-overlap bonus when
the recall query shares words with the stored coordinate. Remembering the same
coordinate twice = reinforcement, never duplication — the "dedupe by deterministic id"
property of the deferrable-network design, realized locally first.

Usage (STDIO only):
    MONGO_DB=barygraph_all EMBED_MODEL=qwen3-embedding:8b EMBED_DIM=4096 \
    SMEM_POOL_FILE=/workspace/bary-vector/cognitive/batches/smem_pool.jsonl \
      python3.11 -m scripts.mcp_smem
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("barygraph.mnemo")

# ── allowed tool surface (shared read tools + mnemo session-memory tools) ──────
_ALLOWED_TOOLS = {
    # read-only grounding against barygraph_all (node/BE tools work today;
    # MB/vector tools light up as the all build finishes)
    "find_word",
    "word_senses",
    "word_edges",
    "edge_info",
    "traverse_up",
    "leaf_nodes",
    "sample_metabary",
    "semantic_search",
    "enrich_request",
    "context_search",
    # session-memory pool (this server's own tools)
    "mnemo_remember",
    "mnemo_recall",
    "mnemo_status",
    "mnemo_reset",
}


def _setup_logging() -> None:
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)


# ── import shared engine (registers ALL tools at import-time) ──────────────────
# Env is already fork-friendly: MONGO_DB / EMBED_MODEL / EMBED_DIM / OLLAMA_URL set
# by the opencode MCP config determine against which graph this server grounds.
import scripts.mcp_server as srv  # noqa: E402  (reuses engine, schemas, app)
from scripts.mcp_server import _run_thr  # noqa: E402  (shared thread runner)

mcp = srv.mcp  # same FastMCP instance; tools already registered on it


def _restrict_toolset() -> None:
    """Drop every FastMCP tool not in the allow-list so only the grounding +
    session-memory tools the cognitive agent should see are listed/executable."""
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
        log.warning("mnemo warmup failed: %s", e)


# ── working-memory pool (append-only JSONL, id-dedup = reinforcement) ───────────
_POOL_FILE = Path(os.environ.get(
    "SMEM_POOL_FILE",
    Path(__file__).resolve().parent.parent / "cognitive" / "batches" / "smem_pool.jsonl",
))
_pool_lock = threading.Lock()
_pool: dict[tuple[str, str], dict] = {}          # (session_id, coordinate_id) -> record
_reset_marks: dict[str, str] = {}                # session_id -> latest reset ts
_session_last_ts: dict[str, int] = {}            # session_id -> highest ms seen (monotonic)
_rev_level: dict[str, int] = {}                  # level -> reverse sort key helper


def _now_ms() -> int:
    return int(time.time() * 1000)


def _next_ts(session_id: str) -> int:
    """Monotonic timestamp for a session: strictly greater than every prior
    record (remember or reset) in the same session, across process restarts.

    Wall clock has millisecond granularity — rapid sequential calls would
    otherwise share a timestamp and let a post-reset remember look older than
    the tombstone it must survive. This is the fix for that race.
    """
    ts = max(_now_ms(), _session_last_ts.get(session_id, 0) + 1)
    _session_last_ts[session_id] = ts
    return ts


def _load_pool() -> None:
    """Replay the JSONL into _pool, applying latest-reset tombstones per session."""
    if not _POOL_FILE.exists():
        return
    lines = _POOL_FILE.read_text().splitlines()
    resets: dict[str, int] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            log.warning("skipping unparseable line in %s", _POOL_FILE)
            continue
        ms = int(rec.get("_ms", 0))
        if rec.get("kind") == "reset" and rec.get("session_id"):
            resets[rec["session_id"]] = max(resets.get(rec["session_id"], -1), ms)
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ms = int(rec.get("_ms", 0))
        _session_last_ts[rec["session_id"]] = max(
            _session_last_ts.get(rec["session_id"], 0), ms
        )
        if rec.get("kind") == "reset":
            continue
        sid, cid = rec.get("session_id"), rec.get("coordinate_id")
        if not sid or not cid:
            continue
        if resets.get(sid, -1) > ms:
            continue  # remembered before the session's latest reset
        _pool[(sid, cid)] = rec


def _append(rec: dict) -> None:
    _POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _POOL_FILE.exists() and _POOL_FILE.stat().st_size > 0:
        data = _POOL_FILE.read_bytes()
        if not data.endswith(b"\n"):
            with _POOL_FILE.open("ab") as fh:
                fh.write(b"\n")
    with _POOL_FILE.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _norm_words(words) -> list[str]:
    if not words:
        return []
    if isinstance(words, str):
        return [w.strip() for w in words.split(",") if w.strip()]
    return [str(w).strip() for w in words if str(w).strip()]


def _remember_body(
    session_id: str,
    coordinate_id: str,
    level: int,
    words=None,
    query: str = "",
    compactness: str = "",
    origin: str = "origin",
    author: str = "",
) -> str:
    if not session_id or not session_id.strip():
        return "session_id must be non-empty"
    if not coordinate_id or not coordinate_id.strip():
        return "coordinate_id must be non-empty"
    session_id = session_id.strip()
    coordinate_id = coordinate_id.strip()
    ws = sorted(set(_norm_words(words)))
    key = (session_id, coordinate_id)

    with _pool_lock:
        prev = _pool.get(key)
        strength = prev["strength"] + 1 if prev else 1
        if prev:
            ws = sorted(set(prev.get("words", [])) | set(ws))
        rec = {
            "kind": "remember",
            "session_id": session_id,
            "coordinate_id": coordinate_id,
            "level": int(level),
            "words": ws,
            "query": query,
            "compactness": compactness,
            "origin": origin,
            "strength": strength,
            "author": author,
            "_ms": _next_ts(session_id),
        }
        _pool[key] = rec
        _append(rec)
    return json.dumps(
        {"ok": True, "coordinate_id": coordinate_id, "strength": strength,
         "words": ws},
        ensure_ascii=False,
    )


def _recall_body(
    session_id: str,
    query: str = "",
    top_k: int = 10,
    half_life_s: int = 600,
) -> str:
    if not session_id or not session_id.strip():
        return "session_id must be non-empty"
    session_id = session_id.strip()
    top_k = max(1, min(top_k, 50))
    half_life_s = max(1, half_life_s)
    now = _now_ms()
    qwords = set(_norm_words(query))
    half_life_ms = half_life_s * 1000

    with _pool_lock:
        scored = []
        for (sid, cid), rec in _pool.items():
            if sid != session_id:
                continue
            age_ms = max(0, now - int(rec.get("_ms", 0)))
            act = float(rec.get("strength", 1)) * (0.5 ** (age_ms / half_life_ms))
            overlap = len(set(rec.get("words", [])) & qwords)
            if overlap:
                act *= 1.0 + overlap
            scored.append((act, rec))
        scored.sort(key=lambda x: (-x[0], -x[1].get("strength", 0)))

    out = [{
        "coordinate_id": rec["coordinate_id"],
        "level": rec.get("level"),
        "words": rec.get("words", []),
        "strength": rec.get("strength", 1),
        "activation": round(sc, 4),
        "origin": rec.get("origin", "origin"),
        "query": rec.get("query", ""),
        "compactness": rec.get("compactness", ""),
        "author": rec.get("author", ""),
    } for sc, rec in scored[:top_k]]
    return json.dumps({"session_id": session_id, "request": query, "hits": out},
                      ensure_ascii=False)


def _status_body(session_id: str = "") -> str:
    with _pool_lock:
        if session_id:
            recs = [r for (sid, _), r in _pool.items() if sid == session_id]
        else:
            recs = list(_pool.values())
        sessions = sorted({r["session_id"] for r in recs})
        top = sorted(recs, key=lambda r: (-r.get("strength", 0),
                                          -int(r.get("_ms", 0))))[:10]
    return json.dumps({
        "store": str(_POOL_FILE),
        "exists": _POOL_FILE.exists(),
        "sessions": sessions,
        "n_records": len(recs),
        "top": [{
            "session_id": r["session_id"], "coordinate_id": r["coordinate_id"],
            "level": r.get("level"), "words": r.get("words", []),
            "strength": r.get("strength", 1), "origin": r.get("origin"),
        } for r in top],
    }, ensure_ascii=False)


def _reset_body(session_id: str) -> str:
    if not session_id or not session_id.strip():
        return "session_id must be non-empty"
    session_id = session_id.strip()
    with _pool_lock:
        ts = _next_ts(session_id)
        for key in [k for k in _pool if k[0] == session_id]:
            del _pool[key]
        marker = {"kind": "reset", "session_id": session_id, "_ms": ts}
        _append(marker)
    return f"session {session_id} reset (tombstone at ms={ts})"


@mcp.tool()
async def mnemo_remember(
    session_id: str,
    coordinate_id: str,
    level: int,
    words: str = "",
    query: str = "",
    compactness: str = "",
    origin: str = "origin",
    author: str = "",
) -> str:
    """Record or reinforce a coordinate in the session's working-memory pool.

    Remembering the same coordinate_id twice does NOT duplicate it — it
    increments its strength (id-dedup = reinforcement). Words are deduped and
    unioned across remembers.

    Args:
        session_id: cognitive session tag, e.g. "cog-mb-013".
        coordinate_id: the BE/MB/node _id you just retrieved from the graph.
        level: graph level (10-15) of the coordinate.
        words: comma-separated words seen on this coordinate (branch/bridge words).
        query: the probe that surfaced it (optional, aids recall overlap).
        compactness: compactness block string if you have one (mean_cos etc.).
        origin: "origin" (you found it) or "propagation" (borrowed) — for traceability.
        author: model signature, e.g. "big-pickle@opencode-0.5".
    """
    return await _run_thr(
        _remember_body, session_id, coordinate_id, level, words,
        query, compactness, origin, author,
    )


@mcp.tool()
async def mnemo_recall(
    session_id: str,
    query: str = "",
    top_k: int = 10,
    half_life_s: int = 600,
) -> str:
    """Return the top-k active coordinates in the session's working-memory pool.

    Ranking = strength * 0.5^(age/half_life_s) with a multiplicative bonus for
    each stored word that overlaps the recall query's words. This is the
    READ/WORKING side of the memory loop: the cognitive agent calls this before
    its next probe to keep working where it was, instead of re-deriving.

    Args:
        session_id: the pool to recall from.
        query: current probe text (words shared with a stored coordinate raise it).
        top_k: 1-50 (default 10).
        half_life_s: recency decay half-life in seconds (default 600 = 10 min).
    """
    return await _run_thr(_recall_body, session_id, query, top_k, half_life_s)


@mcp.tool()
async def mnemo_status(session_id: str = "") -> str:
    """Summary of the working-memory pool: sessions, record count, top entries.

    Use to check what the cognitive agent has accumulated so far (or for a
    single session's pool before deciding whether to reset it).
    """
    return await _run_thr(_status_body, session_id)


@mcp.tool()
async def mnemo_reset(session_id: str) -> str:
    """Clear a session's working-memory pool (tombstone in the append-only log).

    Earlier remembers for the session stop resolving on replay; the JSONL is
    never rewritten. Use when the current phase is over and the next should not
    inherit stale coordinates.
    """
    return await _run_thr(_reset_body, session_id)


def main() -> int:
    _setup_logging()
    _load_pool()
    db = os.environ.get("MONGO_DB", "barygraph_poc")
    log.info(
        "mnemo MCP starting | db=%s tools=%d store=%s",
        db, len(_ALLOWED_TOOLS), _POOL_FILE,
    )
    _warmup()
    _restrict_toolset()
    mcp.run(transport="stdio")
    return 0  # not reached


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BaryGraph session-memory MCP (internal)")
    ap.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="STDIO by default — this internal server must NOT be tunneled.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    args = ap.parse_args()
    if args.transport != "stdio":
        sys.stderr.write(
            "mnemo MCP refuses non-stdio transport: internal-only server.\n"
        )
        sys.exit(2)
    main()