#!/usr/bin/env python3
"""cog_monitor.py — portable snapshot of live BaryGraph service + cognitive flow.

The older runners (s04_monitor.py, s04_watchdog.py, cog_cycle.py, cog_mb_loop.py)
are VPS-bound: they assume python3.11, /workspace/bary-vector, a local GPU ollama
at 127.0.0.1:11435, and /opt/ollama/run.sh. On a host that has none of those the
autonomous loop cannot run, but the *artifacts* and the live PoC service can still
be watched. This monitor makes no such assumptions — it only reads files and asks
scripts/mcp_server_ctl.sh for service state.

Reports, as one line:
  - BaryGraph MCP server + cloudflared tunnel status (from mcp_server_ctl.sh)
  - cognitive-batch artifact counts: terms, papers, proposals, candidates, builds
  - the most recent SMB build (name / author / level / id)

Usage:
  python3 scripts/cog_monitor.py                # one snapshot, then exit
  python3 scripts/cog_monitor.py --once         # same
  python3 scripts/cog_monitor.py --watch 300    # snapshot every 300s (foreground)
  python3 scripts/cog_monitor.py --json         # machine-readable, no log write

Appends each snapshot to $COG_MONITOR_LOG (default /tmp/opencode/cog_monitor.log).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "cognitive" / "batches"
CTL = ROOT / "scripts" / "mcp_server_ctl.sh"
LOG = Path(os.environ.get("COG_MONITOR_LOG", "/tmp/opencode/cog_monitor.log"))

_ARTIFACTS = {
    "terms": "terms_batch.jsonl",
    "papers": "papers_batch.jsonl",
    "proposals": "smb_proposals.jsonl",
    "candidates": "smb_candidates.jsonl",
    "builds": "smb_builds.jsonl",
}


def _count(path: Path) -> int:
    try:
        return sum(1 for ln in path.read_text().splitlines() if ln.strip())
    except OSError:
        return 0


def _last(path: Path) -> dict | None:
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        return None
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def service_state() -> dict:
    """Ask mcp_server_ctl.sh for MCP + tunnel state (never raises)."""
    out = ""
    try:
        r = subprocess.run(
            ["bash", str(CTL), "status"],
            capture_output=True, text=True, cwd=ROOT, timeout=30,
        )
        out = (r.stdout or "") + (r.stderr or "")
    except (subprocess.SubprocessError, OSError) as e:
        return {"mcp": "?", "tunnel": "?", "tunnel_url": "", "err": repr(e)}
    url = ""
    for ln in out.splitlines():
        if "cloudflared tunnel:" in ln and "https://" in ln:
            url = ln.split("https://", 1)[1].strip().rstrip(")")
        if "tunnel url" in ln.lower() and "https://" in ln:
            url = ln.split("https://", 1)[1].strip()
    return {
        "mcp": "UP" if "MCP server: UP" in out else ("DOWN" if "MCP server: DOWN" in out else "?"),
        "tunnel": "UP" if "cloudflared tunnel: UP" in out else ("DOWN" if "cloudflared tunnel: DOWN" in out else "?"),
        "tunnel_url": url,
    }


def snapshot() -> dict:
    svc = service_state()
    counts = {k: _count(BATCH / fn) for k, fn in _ARTIFACTS.items()}
    last_build = _last(BATCH / _ARTIFACTS["builds"])
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mcp": svc["mcp"],
        "tunnel": svc["tunnel"],
        "tunnel_url": svc.get("tunnel_url", ""),
        "counts": counts,
        "last_build": (
            {
                "name": last_build.get("name"),
                "author": last_build.get("author"),
                "level": last_build.get("level"),
                "smb_id": last_build.get("smb_id"),
                "built_at": last_build.get("built_at"),
            }
            if last_build else None
        ),
    }


def render(s: dict) -> str:
    c = s["counts"]
    lb = s["last_build"]
    last = f"{lb['name']} (L{lb['level']}, {lb['smb_id']})" if lb else "-"
    return (
        f"{s['ts']} mcp={s['mcp']} tunnel={s['tunnel']} "
        f"terms={c['terms']} papers={c['papers']} proposals={c['proposals']} "
        f"candidates={c['candidates']} builds={c['builds']} | last_build={last}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="repeat every SECONDS (0 = one-shot)")
    ap.add_argument("--once", action="store_true", help="one snapshot, then exit")
    ap.add_argument("--json", action="store_true", help="emit JSON, do not append to log")
    args = ap.parse_args()

    while True:
        s = snapshot()
        line = render(s)
        if args.json:
            print(json.dumps(s, ensure_ascii=False))
        else:
            print(line, flush=True)
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a") as fh:
                fh.write(line + "\n")
        if not args.watch or args.once:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
