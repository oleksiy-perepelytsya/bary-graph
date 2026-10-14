#!/usr/bin/env bash
# =============================================================================
# BaryGraph MCP server control script (streamable-http over ngrok).
#
# ARCHITECTURE (rewritten 2026-10-13 — was cloudflared + hand-launched python)
# ---------------------------------------------------------------------------
# The server and the public tunnel are now BOTH systemd **user** units:
#
#   barygraph-mcp.service     python3 -m scripts.mcp_server --transport
#                             streamable-http --host 0.0.0.0 --port 8000
#                             WorkingDirectory = repo root, MCP_PUBLIC=1
#   ngrok-barygraph.service   ngrok start barygraph-mcp   (static free dev
#                             domain, defined in ~/.config/ngrok/ngrok.yml)
#
# This script is now a thin controller/status wrapper over `systemctl --user`.
# linger is enabled (`loginctl enable-linger "$USER"`), so both units come back
# at boot without a login. Unit files live in ~/.config/systemd/user/.
#
# WHY THE OLD SETSID/NOHUP + QUICK-TUNNEL ARCHAEOLOGY IS GONE
# -----------------------------------------------------------
# Until 2026-10-13 the server was launched with `setsid nohup ... & disown` and
# the public endpoint came from a cloudflared *quick* tunnel. Quick tunnels are
# account-less and Cloudflare revokes their hostname server-side without
# warning — observed 3x in 24h on 2026-10-12/13 with shrinking lifetimes
# (~28h, ~6h, ~3h). Each time the process stayed "up" but every request was
# NXDOMAIN, the log filling with
#     ERR Register tunnel error ... error="Unauthorized: Tunnel not found"
# and recovery required killing cloudflared to mint a NEW random hostname,
# which broke every external client. systemd now supervises the process
# (Restart=always) and ngrok provides a stable, non-revocable hostname: the
# free plan includes one immutable dev domain, so the URL never changes.
#
# THE STILL-LOAD-BEARING DETAILS (read before "fixing" anything here)
# ------------------------------------------------------------------
# 1. mcp (PyPI) MUST be >= 1.30.0.
#    mcp==1.29.0 crashes long-running calls with
#        AssertionError: Request already responded to   (mcp/shared/session.py)
#    on the streamable-http 202-async path: enrich_request is the FIRST tool
#    slow enough (embed + sequential $vectorSearches) to cross the async
#    threshold, so it kills the session with "MCP error -32001: Request timed
#    out". Upgrade: `pip install "mcp==1.30.0"` then restart.
#
# 2. Warmup is real.
#    On boot the server runs a warm $vectorSearch + ensure_indexes and logs
#    "warmup: $vectorSearch got ... hits / ensure_indexes done". A bare
#    "Uvicorn running" line does NOT mean it can serve vector queries yet.
#    `status` checks the journal for the warmup-done line, not just the unit
#    state.
#
# 3. cwd & env.
#    The unit runs with WorkingDirectory=<repo root> so `scripts.mcp_server`
#    and the plain .env (MONGO_DB=barygraph_poc, EMBED_DIM=1024, OLLAMA_URL,
#    MONGO_URI) resolve — Settings.load() discovers .env from the cwd. The
#    server loads .env ONCE at startup, so changing any of it needs a restart.
#
# 4. MCP_PUBLIC=1 is set in the unit (public read-only surface).
#    The tunnel must NOT expose the create_* write tools. `tunnel check` does a
#    real MCP handshake through the public URL and fails if any write tool is
#    visible.
#
# Usage:
#   scripts/mcp_server_ctl.sh status                 # unit + warmup + tunnel + public check
#   scripts/mcp_server_ctl.sh start|stop|restart     # the server unit
#   scripts/mcp_server_ctl.sh enable|disable         # boot enablement (server + tunnel)
#   scripts/mcp_server_ctl.sh tunnel status|url|check
#   scripts/mcp_server_ctl.sh tunnel start|stop|restart
#   scripts/mcp_server_ctl.sh logs [N]               # journal of the server (default 50)
#   scripts/mcp_server_ctl.sh env                    # effective env of the running unit
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCP_UNIT="${MCP_UNIT:-barygraph-mcp.service}"
TUNNEL_UNIT="${TUNNEL_UNIT:-ngrok-barygraph.service}"
NGROK_CONFIG="${NGROK_CONFIG:-$HOME/.config/ngrok/ngrok.yml}"
NGROK_API="${NGROK_API:-http://127.0.0.1:4040}"
PORT="${MCP_PORT:-8000}"
TUNNEL_LOG="${TUNNEL_LOG:-/tmp/opencode/ngrok.log}"

# Interpreter used only for the mcp version probe / public check (the unit
# owns the real server process). MCP_PYTHON wins; else python3.11/python3.
if [[ -n "${MCP_PYTHON:-}" ]]; then
  PYTHON="$MCP_PYTHON"
else
  PYTHON=""
  for c in python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import mcp' >/dev/null 2>&1; then
      PYTHON="$c"; break
    fi
  done
  PYTHON="${PYTHON:-python3}"
fi

sc() { systemctl --user "$@"; }

unit_active() { sc is-active -q "$1" 2>/dev/null; }
unit_enabled() { sc is-enabled -q "$1" 2>/dev/null; }

is_warm() {
  # NOTE: grep -q is avoided here on purpose — with `set -o pipefail` the
  # upstream journalctl gets SIGPIPE and the pipeline returns 141 even on a
  # successful match. grep -c consumes everything instead.
  local n
  n="$(journalctl --user -u "$MCP_UNIT" --since "-15min" --no-pager 2>/dev/null \
        | grep -ac "warmup: ensure_indexes done" || true)"
  [[ "${n:-0}" -gt 0 ]]
}

mcp_version() {
  "$PYTHON" -c 'import importlib.metadata as m; print(m.version("mcp"))' 2>/dev/null || echo unknown
}

mcp_version_ok() {
  "$PYTHON" -c 'import importlib.metadata as m, sys; v=[int(x) for x in m.version("mcp").split(".")[:3]]; sys.exit(0 if v >= [1,30,0] else 1)' 2>/dev/null
}

# --- tunnel URL discovery: live ngrok agent API first, config file fallback ---
config_domain() {
  local d
  d="$(sed -nE 's/^[[:space:]]*domain:[[:space:]]*//p' "$NGROK_CONFIG" 2>/dev/null | head -1 | tr -d '"' || true)"
  [[ -n "$d" ]] && printf 'https://%s' "$d" || true
}

ngrok_url() {
  local u
  u="$(curl -s -m 3 "$NGROK_API/api/tunnels" 2>/dev/null \
        | "$PYTHON" -c 'import sys,json
try:
    d=json.load(sys.stdin)
    ts=[t.get("public_url","") for t in d.get("tunnels",[]) if str(t.get("public_url","")).startswith("https")]
    print(ts[0] if ts else "", end="")
except Exception:
    pass' 2>/dev/null || true)"
  [[ -z "$u" ]] && u="$(config_domain)"
  printf '%s' "$u"
}

# Real end-to-end MCP handshake through the public URL. Prints a one-line
# verdict; returns non-zero if unreachable or if write tools are exposed.
public_check() {
  local url="${1:-$(ngrok_url)}"
  [[ -z "$url" ]] && { echo "  public check: SKIPPED (no tunnel URL known)"; return 1; }
  "$PYTHON" - "$url" <<'PY' || return 1
import json, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/mcp"
H = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
def parse(b):
    for ln in b.splitlines():
        if ln.strip().startswith("data:"):
            try: return json.loads(ln.strip()[5:].strip())
            except Exception: pass
    try: return json.loads(b)
    except Exception: return None
def post(payload, extra=None):
    h = dict(H); h.update(extra or {})
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status, r.headers.get("mcp-session-id"), r.read().decode()
try:
    st, sid, body = post({"jsonrpc":"2.0","id":1,"method":"initialize",
        "params":{"protocolVersion":"2025-06-18","capabilities":{},
                  "clientInfo":{"name":"ctl-check","version":"1"}}})
    m = parse(body) or {}
    si = (m.get("result") or {}).get("serverInfo") or {}
    if not si:
        print(f"  public check: FAILED (initialize {st}: {body[:120].replace(chr(10),' ')})"); sys.exit(1)
    hdr = {"Mcp-Session-Id": sid} if sid else {}
    post({"jsonrpc":"2.0","method":"notifications/initialized"}, hdr)
    st2, _, b2 = post({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}, hdr)
    tools = [t["name"] for t in (parse(b2) or {}).get("result",{}).get("tools",[])]
    write = sorted({"create_sense","create_edge","create_word","create_structure_meta_bary"} & set(tools))
    print(f"  public check: OK — {si.get('name')} {si.get('version')}, {len(tools)} tools, url={url}")
    if write:
        print(f"  public check: !! WRITE TOOLS EXPOSED: {', '.join(write)} (MCP_PUBLIC=1 not in effect)")
        sys.exit(1)
except Exception as e:
    print(f"  public check: FAILED ({type(e).__name__}: {e})"); sys.exit(1)
PY
}

cmd_status() {
  if unit_active "$MCP_UNIT"; then
    local p; p="$(sc show -p MainPID --value "$MCP_UNIT" 2>/dev/null || echo '?')"
    if is_warm; then echo "MCP server: UP (pid $p, warm)  [unit $MCP_UNIT]"
    else echo "MCP server: UP (pid $p, warming up — wait for 'ensure_indexes done')"; fi
  else
    echo "MCP server: DOWN  [unit $MCP_UNIT] (run 'start')"
  fi

  if unit_active "$TUNNEL_UNIT"; then
    echo "ngrok tunnel: UP ($(ngrok_url))  [unit $TUNNEL_UNIT]"
  else
    echo "ngrok tunnel: DOWN (public URL will NOT work — run 'tunnel start')"
  fi

  local v; v="$(mcp_version)"
  if mcp_version_ok; then echo "mcp pkg: $v (>=1.30.0 OK)"
  else echo "mcp pkg: $v  <-- BUGGY (<1.30.0, see header comment)"; fi

  unit_enabled "$MCP_UNIT" || echo "note: $MCP_UNIT is NOT enabled at boot"
  unit_enabled "$TUNNEL_UNIT" || echo "note: $TUNNEL_UNIT is NOT enabled at boot"

  public_check || true
}

cmd_start()   { sc start "$MCP_UNIT";  echo "started $MCP_UNIT"; }
cmd_stop()    { sc stop "$MCP_UNIT";   echo "stopped $MCP_UNIT"; }
cmd_restart() { sc restart "$MCP_UNIT"; echo "restarted $MCP_UNIT"; }

cmd_enable()  { sc enable "$MCP_UNIT" "$TUNNEL_UNIT"; loginctl enable-linger "$USER" 2>/dev/null || true; echo "enabled at boot: $MCP_UNIT $TUNNEL_UNIT (linger on)"; }
cmd_disable() { sc disable "$MCP_UNIT" "$TUNNEL_UNIT"; echo "disabled at boot: $MCP_UNIT $TUNNEL_UNIT"; }

cmd_tunnel() {
  case "${1:-status}" in
    status)
      echo "ngrok unit: $TUNNEL_UNIT ($(sc is-active "$TUNNEL_UNIT" 2>/dev/null || echo inactive))"
      echo "tunnel url: $(ngrok_url)"
      echo "ngrok config: $NGROK_CONFIG"
      ;;
    url)     ngrok_url; echo ;;
    check)   public_check ;;
    start)   sc start "$TUNNEL_UNIT";  echo "started $TUNNEL_UNIT" ;;
    stop)    sc stop "$TUNNEL_UNIT";   echo "stopped $TUNNEL_UNIT" ;;
    restart) sc restart "$TUNNEL_UNIT"; echo "restarted $TUNNEL_UNIT ($(ngrok_url))" ;;
    *) echo "usage: $0 tunnel {status|url|check|start|stop|restart}"; exit 2 ;;
  esac
}

cmd_logs() {
  journalctl --user -u "$MCP_UNIT" -n "${1:-50}" --no-pager
}

cmd_env() {
  sc show -p Environment --value "$MCP_UNIT" 2>/dev/null | tr ' ' '\n'
  echo "WorkingDirectory=$(sc show -p WorkingDirectory --value "$MCP_UNIT" 2>/dev/null)"
}

case "${1:-status}" in
  status)  cmd_status ;;
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_restart ;;
  enable)  cmd_enable ;;
  disable) cmd_disable ;;
  tunnel)  shift; cmd_tunnel "${1:-status}" ;;
  logs)    shift; cmd_logs "${1:-50}" ;;
  env)     cmd_env ;;
  *) echo "usage: $0 {status|start|stop|restart|enable|disable|tunnel|logs|env}"; exit 2 ;;
esac
