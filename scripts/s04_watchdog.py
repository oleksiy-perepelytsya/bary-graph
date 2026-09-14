"""s04_watchdog.py — resilience watchdog for the L15 orphan re-entry run.

Purpose
-------
The run's embedding goes through the *local* ollama daemon (127.0.0.1:11435).
If that GPU falls into a bad state, ollama silently falls back to CPU
(/api/ps reports size_vram ~= 0 while still answering embed requests slowly),
or the daemon dies outright. Without intervention the phase crawls or stalls.
This watchdog:
  1. detects CPU fallback / daemon death / driver faults across >=VIOLATIONS
     consecutive polls, then restarts the local daemon via /opt/ollama/run.sh
     and warm-loads qwen3-embedding on the GPU again; and
  2. relaunches s04 (and its monitor) if they die, using the same env as the
     original launch, so a transient embed failure during an ollama bounce is
     self-healing.
Restarts are rate-limited (cooldown + daily cap) and fully logged to
/tmp/opencode/s04_watchdog.log so nothing loops silently.

Run as a detached daemon:
    setsid nohup python3.11 -m scripts.s04_watchdog >> /tmp/opencode/s04_watchdog.log 2>&1 < /dev/null &

Self-test (one health evaluation, no actions):
    python3.11 -m scripts.s04_watchdog --check
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request

OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435").rstrip("/")
MODEL_PREFIX = "qwen3-embedding"
RUN_SH = Path("/opt/ollama/run.sh")

POLL_SECONDS = 30
VIOLATIONS_TO_RESTART = 2
MIN_VRAM_FRAC = 0.25
RESTART_COOLDOWN = 600          # min seconds between ollama restarts
MAX_OLLAMA_RESTARTS = 5         # per watchdog lifetime; then log-and-stop
POST_RESTART_GRACE = 240        # seconds after restart before re-evaluating
SERVER_UP_TIMEOUT = 300
S04_DEAD_POLLS = 2              # s04 dead this many consecutive polls -> relaunch
S04_RELAUNCH_COOLDOWN = 900

S04_LOG = Path("/tmp/opencode/s04_reentry.log")
S04_PIDFILE = Path("/tmp/opencode/s04.pid")
MON_PIDFILE = Path("/tmp/opencode/s04_monitor.pid")
WATCHDOG_PIDFILE = Path("/tmp/opencode/s04_watchdog.pid")
OLLAMA_RUN_LOG = Path("/tmp/opencode/ollama_run.log")

S04_CMD = (
    "cd /workspace/bary-vector; set -a; . ./.env.build-all 2>/dev/null; set +a; "
    "export OLLAMA_URL=http://127.0.0.1:11435; "
    "export EMBED_CACHE_FILE=/storage/bary/s04_reentry_cache.jsonl; "
    "export PYMONGO_SOCKET_TIMEOUT_MS=600000; "
    "setsid nohup python3.11 -m scripts.s04_l15_edges --force "
    "> /tmp/opencode/s04_reentry.log 2>&1 < /dev/null & "
    "echo $! > /tmp/opencode/s04.pid"
)
MON_CMD = (
    "cd /workspace/bary-vector; set -a; . ./.env.build-all 2>/dev/null; set +a; "
    "setsid nohup python3.11 -m scripts.s04_monitor --hours 120 "
    ">> /tmp/opencode/s04_monitor.log 2>&1 < /dev/null & "
    "echo $! > /tmp/opencode/s04_monitor.pid"
)

_log_h = None
_console = False


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [watchdog] {msg}"
    try:
        if _log_h is not None and not _log_h.closed:
            _log_h.write(line + "\n")
            _log_h.flush()
    except OSError:
        pass
    if _console:
        print(line, flush=True)


def _ps() -> dict | None:
    """Hit /api/ps; None on network failure.

    Returns the model record dict or an empty dict if the server is up but
    the model is not loaded.
    """
    try:
        with request.urlopen(f"{OLLAMA_BASE}/api/ps", timeout=10) as r:
            data = json.loads(r.read())
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as e:
        log(f"ps unavailable: {e!r}")
        return None
    for m in data.get("models", []):
        if m.get("name", "").startswith(MODEL_PREFIX):
            return m
    return {}


def vram_frac(m: dict | None) -> float:
    """0.0 for down/absent, else size_vram/size (1.0 = fully on GPU)."""
    if not m:
        return 0.0
    size = m.get("size") or 0
    vram = m.get("size_vram") or 0
    return vram / size if size else 0.0


def nvidia_ok() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_gpu_memory",
             "--format=csv,noheader"],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def daemon_wait_up(timeout: int = SERVER_UP_TIMEOUT) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = _ps()
        if m is not None:
            return True
        time.sleep(5)
    return False


def warm_model() -> float:
    """Force-load the model on the GPU and return the resulting vram frac."""
    body = {
        "model": "qwen3-embedding:8b",
        "input": ["watchdog warm-up probe"],
        "keep_alive": "6h",
        "options": {"num_thread": 8, "num_gpu": 999},
    }
    req = request.Request(
        f"{OLLAMA_BASE}/api/embed",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for _ in range(8):  # up to ~8 min for a cold GPU load
        try:
            with request.urlopen(req, timeout=300) as r:
                r.read()
        except (urlerror.URLError, urlerror.HTTPError, TimeoutError, OSError):
            pass
        frac = vram_frac(_ps())
        if frac >= MIN_VRAM_FRAC:
            return frac
        time.sleep(20)
    return vram_frac(_ps())


def restart_ollama() -> bool:
    log("RESTART ollama: killing serve + llama-server children")
    for pat in ("/opt/ollama/bin/ollama serve", "lib/ollama/llama-server"):
        subprocess.run(["pkill", "-TERM", "-f", pat], capture_output=True)
    time.sleep(4)
    for pat in ("/opt/ollama/bin/ollama serve", "lib/ollama/llama-server"):
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
            subprocess.run(["pkill", "-KILL", "-f", pat], capture_output=True)
            time.sleep(2)
    if not RUN_SH.exists():
        log(f"FATAL: {RUN_SH} missing; cannot restart ollama")
        return False
    with OLLAMA_RUN_LOG.open("ab") as out:
        log(f"starting {RUN_SH} (log {OLLAMA_RUN_LOG})")
        subprocess.Popen(
            [str(RUN_SH)],
            stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if not daemon_wait_up():
        log("ERROR: daemon did not answer /api/ps within timeout")
        return False
    frac = warm_model()
    log(f"after restart: model vram frac={frac:.2f} "
        f"{'HEALTHY' if frac >= MIN_VRAM_FRAC else 'still degraded'}")
    return frac >= MIN_VRAM_FRAC


def _pid_alive(pidfile: Path) -> bool:
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def launch(cmd: str, pidfile: Path, label: str) -> None:
    log(f"relaunching {label}")
    subprocess.Popen(["bash", "-lc", cmd], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(2)
        if _pid_alive(pidfile):
            log(f"{label} relaunched (pid={pidfile.read_text().strip()})")
            return
    log(f"WARNING: {label} relaunch didn't report a pidfile")


def check_once() -> dict:
    m = _ps()
    frac = vram_frac(m)
    nv = nvidia_ok()
    return {"ps": m is not None, "model": bool(m), "vram_frac": frac, "nvidia_ok": nv}


def main() -> int:
    global _log_h, _console
    _log_h = open("/tmp/opencode/s04_watchdog.log", "a")
    _console = "--check" in sys.argv

    if "--check" in sys.argv:
        s = check_once()
        log(f"CHECK {s}")
        return 0

    if WATCHDOG_PIDFILE.exists():
        if _pid_alive(WATCHDOG_PIDFILE):
            log("another watchdog instance is running; exiting")
            return 2
    WATCHDOG_PIDFILE.write_text(str(os.getpid()))

    log(f"watchdog up (poll={POLL_SECONDS}s, min_vram={MIN_VRAM_FRAC}, "
        f"restart_cooldown={RESTART_COOLDOWN}s, cap={MAX_OLLAMA_RESTARTS})")

    ollama_restarts = 0
    last_ollama_restart = 0.0
    last_s04_relaunch = 0.0
    violations = 0
    s04_dead = 0
    grace_until = 0.0

    if not _pid_alive(S04_PIDFILE):
        log("no live s04 pidfile at startup — allowing early relaunch if still dead")
        s04_dead = S04_DEAD_POLLS

    while True:
        time.sleep(POLL_SECONDS)
        now = time.time()

        # ---- ollama health ----
        s = check_once()
        bad = not s["ps"] or not s["model"] or s["vram_frac"] < MIN_VRAM_FRAC or not s["nvidia_ok"]
        if bad:
            violations += 1
            log(f"unhealthy poll {violations}/{VIOLATIONS_TO_RESTART}: {s}")
        else:
            if violations:
                log(f"recovered (vram_frac={s['vram_frac']:.2f}); clearing violations")
            violations = 0

        if (violations >= VIOLATIONS_TO_RESTART and now > grace_until
                and now - last_ollama_restart >= RESTART_COOLDOWN):
            if ollama_restarts >= MAX_OLLAMA_RESTARTS:
                log(f"CAP REACHED ({MAX_OLLAMA_RESTARTS} restarts) — stopping self-heal; "
                    "investigate GPU state manually")
                violations = 0
            else:
                ok = restart_ollama()
                ollama_restarts += 1
                last_ollama_restart = time.time()
                grace_until = now + POST_RESTART_GRACE
                violations = 0 if ok else VIOLATIONS_TO_RESTART - 1
                log(f"ollama restarts so far: {ollama_restarts}; "
                    f"grace until {time.strftime('%H:%M:%S', time.localtime(grace_until))}")

        # ---- s04: relaunch only when ollama is healthy, with a cooldown ----
        if s["ps"]:
            if _pid_alive(S04_PIDFILE):
                s04_dead = 0
            elif now - last_s04_relaunch >= S04_RELAUNCH_COOLDOWN:
                s04_dead += 1
                log(f"s04 unresponsive poll {s04_dead}/{S04_DEAD_POLLS}")
                if s04_dead >= S04_DEAD_POLLS:
                    launch(S04_CMD, S04_PIDFILE, "s04")
                    last_s04_relaunch = time.time()
                    s04_dead = 0
        else:
            s04_dead = 0  # ollama down: hold off, s04 may be mid-retry

        # ---- monitor (best-effort, never critical) ----
        if not _pid_alive(MON_PIDFILE) and now - last_s04_relaunch >= 60:
            launch(MON_CMD, MON_PIDFILE, "s04_monitor")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())