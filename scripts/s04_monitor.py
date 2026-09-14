"""s04 + cognitive cycle hourly monitor.

Logs s04 Phase-A progress (keys done/total, RSS, cache size) and the current
SMB proposal count to /tmp/opencode/s04_monitor.log every --interval seconds.
Detached runner: setsid nohup python3.11 scripts/s04_monitor.py --hours 120
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

LOG = Path("/tmp/opencode/s04_monitor.log")
S04_LOG = Path("/tmp/opencode/s04_reentry.log")
PROPOSALS = Path("/workspace/bary-vector/cognitive/batches/smb_proposals.jsonl")
CACHE = Path("/storage/bary/s04_reentry_cache.jsonl")


def s04_progress() -> dict:
    last = ""
    for line in S04_LOG.read_text().splitlines():
        if "embedded" in line:
            last = line
    if not last:
        return {"done": 0, "total": 0, "err": "no progress line"}
    seg = last.split("embedded")[-1].split()[0]  # "N/M"
    done, total = (int(x) for x in seg.split("/"))
    return {"done": done, "total": total, "pct": 100 * done / total}


def snapshot() -> str:
    out = []
    p = s04_progress()
    pct = f"{p['pct']:.1f}%" if p["total"] else p.get("err", "?")
    out.append(f"s04 keys {p.get('done', 0)}/{p.get('total', 0)} ({pct})")

    try:
        import os
        rss = os.popen("ps -o rss= -p 252088 2>/dev/null").read().strip()
        out.append(f"s04 RSS {int(rss) / 1024 / 1024:.1f} GB" if rss else "s04 RSS ?")
    except Exception:
        pass
    try:
        out.append(f"cache {CACHE.stat().st_size / 1e9:.1f} GB")
    except Exception:
        pass
    try:
        n = sum(1 for l in PROPOSALS.read_text().splitlines() if l.strip())
        out.append(f"proposals {n}")
    except Exception:
        out.append("proposals ?")
    return " | ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=3600)
    ap.add_argument("--hours", type=float, default=120)
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    with LOG.open("a") as f:
        while time.time() < deadline:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {snapshot()}\n")
            f.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())