#!/usr/bin/env bash
# =============================================================================
# BaryGraph MCP server control script (streamable-http over cloudflared).
#
# WHY THIS SCRIPT EXISTS
# ---------------------
# For a long time "restart the MCP" was tribal knowledge living in prior-summary
# files: kill the PID, relaunch with setsid/nohup/disown, don't touch the tunnel,
# tail a log and wait for the warmup line. This script encodes that archaeology
# so any future agent/user session can recover the server in one command.
#
# THE LOAD-BEARING DETAILS (read before "fixing" anything here):
#
# 1. The tunnel is sacred.
#    cloudflared (PID 3242, `cloudflared tunnel --url http://127.0.0.1:8000`)
#    forwards public traffic to this server. This script NEVER touches it.
#    "Restart the MCP" = restart the python mcp_server process ONLY.
#
# 2. setsid + nohup + disown is REQUIRED, not style.
#    A plain `& background & disown` is NOT enough: when the launching shell is
#    killed (opencode/bash tool timeouts kill the whole process group) the
#    naive child dies with it. setsid detaches the process from the controlling
#    terminal/session so it survives the shell's death. This exact failure
#    happened in session history (first launch died with the 30s shell timeout).
#    Redirect stdin from /dev/null too — a stray open FD on the tty keeps the
#    shell "running" and the bash tool hangs until ITS timeout instead of the
#    command returning.
#
# 3. mcp (PyPI) MUST be >= 1.30.0.
#    mcp==1.29.0 crashes long-running calls with
#        AssertionError: Request already responded to   (mcp/shared/session.py)
#    on the streamable-http 202-async path: enrich_request is the FIRST tool
#    slow enough (embed + sequential $vectorSearches) to cross the async
#    threshold, so it kills the session with "MCP error -32001: Request timed
#    out". Upgrade: `pip install "mcp==1.30.0"` then restart. Verify there is
#    no "Session ... crashed" line in the log after restart.
#
# 4. Warmup is real.
#    On boot the server runs a warm $vectorSearch + ensure_indexes and logs
#    "warmup: $vectorSearch got ... hits / ensure_indexes done". A bare
#    "Uvicorn running" line does NOT mean it can serve vector queries yet.
#    `status`/`wait` below check for the warmup-done line, not for the process.
#
# 5. Env & cwd.
#    Run from the repo root so `scripts.mcp_server` and the .env
#    (MONGO_DB=barygraph_poc, EMBED_DIM=768, MONGO_URI=...) resolve. The embed
#    model lives at http://ollama:11434 (docker network sibling).
#
# Usage:
#   scripts/mcp_server_ctl.sh status          # is it up + warm?
#   scripts/mcp_server_ctl.sh start           # start (no-op if already running)
#   scripts/mcp_server_ctl.sh stop            # stop only the python MCP server
#   scripts/mcp_server_ctl.sh restart         # stop + start (tunnel untouched)
#   scripts/mcp_server_ctl.sh logs            # tail the server log
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MCP_LOG:-/tmp/opencode/mcp_server.log}"
PORT="${MCP_PORT:-8000}"
PYTHON="${MCP_PYTHON:-python3.11}"

proc_regex='[m]cp_server --transport'

pid() {
  pgrep -f "$proc_regex" | head -1 || true
}

# Track the byte offset the CURRENT run appends at, so warm detection only
# sees this run's "warmup ... done" line, not leftover ones from prior boots.
_LOG_MARK=0

log_mark() {
  if [[ -f "$LOG" ]]; then _LOG_MARK="$(wc -c <"$LOG")"; else _LOG_MARK=0; fi
}

is_warm() {
  # NOTE: must NOT use grep -q here — with `set -o pipefail`, grep -q exits at
  # the first match, the upstream `tail` gets SIGPIPE, and the pipeline returns
  # 141 even on a successful match. grep -c consumes everything instead.
  local n
  n="$(tail -c +$((_LOG_MARK + 1)) "$LOG" 2>/dev/null | grep -ac "warmup: ensure_indexes done" || true)"
  [[ "${n:-0}" -gt 0 ]]
}

cmd_status() {
  local p
  p="$(pid)"
  if [[ -z "$p" ]]; then
    echo "MCP server: DOWN"
  elif is_warm; then
    echo "MCP server: UP (pid $p, warm)"
  else
    echo "MCP server: UP (pid $p, warming up — wait for 'ensure_indexes done' in the log)"
  fi
  if pgrep -f '[c]loudflared tunnel' >/dev/null; then
    echo "cloudflared tunnel: UP (do not touch)"
  else
    echo "cloudflared tunnel: DOWN (the public URL is NOT working)"
  fi
  if python3.11 -c 'import importlib.metadata,sys; sys.exit(0 if importlib.metadata.version("mcp").split(".") >= ["1","30","0"] else 1)' 2>/dev/null; then
    echo "mcp pkg: $(python3.11 -c 'import importlib.metadata; print(importlib.metadata.version("mcp"))') (>=1.30.0 OK)"
  else
    echo "mcp pkg: $(python3.11 -c 'import importlib.metadata; print(importlib.metadata.version("mcp"))')  <-- BUGGY (<1.30.0, see header comment)"
  fi
}

cmd_start() {
  if [[ -n "$(pid)" ]]; then
    echo "already running (pid $(pid)); use restart to bounce it"
    return 0
  fi
  echo "starting MCP server (log: $LOG) ..."
  # See header: setsid+nohup+</dev/null+disown. Do NOT 'simplify' this.
  # The FD juggling (1>&- during nohup, then setsid re-points via the
  # outer redirection) guarantees the daemon does not keep the caller's
  # stdout pipe open — otherwise this script never returns to an agent shell.
  log_mark
  (
    cd "$REPO_ROOT" || exit 1
    MCP_PUBLIC=1 setsid nohup "$PYTHON" -m scripts.mcp_server \
      --transport streamable-http --host 0.0.0.0 --port "$PORT" \
      </dev/null >>"$LOG" 2>&1 &
    disown
    exit 0
  ) >/dev/null 2>&1
  echo "launched (pid $(pid)); waiting for warmup ..."
  for _ in $(seq 1 60); do
    if is_warm; then echo "warm (pid $(pid))"; return 0; fi
    sleep 2
  done
  echo "warning: warmup not confirmed yet — tail $LOG"
}

cmd_stop() {
  local p
  p="$(pid)"
  if [[ -z "$p" ]]; then
    echo "not running"
    return 0
  fi
  echo "stopping pid $p ..."
  kill "$p"
  for _ in $(seq 1 15); do
    [[ -z "$(pid)" ]] && { echo "stopped"; return 0; }
    sleep 1
  done
  echo "did not exit gracefully; killing ..."
  kill -9 "$p" || true
}

case "${1:-status}" in
  status) cmd_status ;;
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  restart) cmd_stop; sleep 2; cmd_start ;;
  logs)   tail -n 50 "$LOG" ;;
  *) echo "usage: $0 {status|start|stop|restart|logs}"; exit 2 ;;
esac