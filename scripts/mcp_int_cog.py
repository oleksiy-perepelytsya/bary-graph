"""Internal cognitive-agent MCP server — localhost only (private), never tunneled.

Re-exports a fixed subset of the shared BaryGraph MCP tools by name so the
cognitive worker sees only what it needs:

    read:  context_search  find_word  word_senses  word_edges  semantic_search
           edge_info  traverse_up            (grounding / bridge verification)
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
