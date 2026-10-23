#!/usr/bin/env python3
"""Float16 embedding shim for the Kaggle GPU tunnel.

Ollama answers ``/api/embed`` with JSON decimal floats, which costs ~12.7 KB
per 1024-d vector on the wire (~4.9 KB after Cloudflare gzip).  Over a slow
uplink that dominates everything else: the T4 is idle while the link is
saturated.  This shim sits between the tunnel and Ollama, converts the
embeddings to raw float16 (2 KB/vector, no JSON expansion) and returns them as
``application/octet-stream``.  Net effect on a 1.5 Mbps downlink: ~2.4x more
vectors per second.

Client side: ``EMBED_F16=1`` in ``lib/embed.py`` decodes the payload directly.
float16 is more than enough precision for cosine work (error ~3e-4 on a
normalized 1024-d vector; q thresholds are ~0.72).

Every other path is proxied to Ollama verbatim, so this can also serve as the
backend of ``scripts.kaggle_embed_server`` (``OLLAMA_BACKENDS=http://localhost:8767``)
— one tunnel then serves both the JSON API and the f16 embed path.

Usage on Kaggle::

    python3 -m scripts.kaggle_f16_shim --tunnel      # shim + cloudflared, prints URL
    python3 -m scripts.kaggle_f16_shim               # shim only, on :8767

Then locally::

    EMBED_F16=1 OLLAMA_URL=https://<tunnel> KAGGLE_API_TOKEN=<token> \
        python3 -m scripts.s04_l15_edges --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

_log = logging.getLogger("f16_shim")

# Filled in main(); read by the per-request handlers.
TOKEN = ""
BACKEND = "http://localhost:11434"
UPSTREAM_TIMEOUT = 900.0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kaggle-f16-shim/1.0"

    # --- helpers ---------------------------------------------------------
    def _send_json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        if not TOKEN:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {TOKEN}":
            return True
        self._send_json(401, {"error": "unauthorized"})
        return False

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n > 0 else b""

    def _upstream(self, path: str, body: bytes) -> tuple[int, dict, bytes]:
        req = Request(f"{BACKEND}{path}", data=body or None, method=self.command)
        req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
        try:
            with urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:  # noqa: S310
                return resp.getcode(), dict(resp.headers), resp.read()
        except HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def _passthrough(self) -> None:
        body = self._read_body()
        code, headers, payload = self._upstream(self.path, body)
        self.send_response(code)
        ctype = headers.get("Content-Type", "application/octet-stream")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # --- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return
        if self.path.split("?")[0] == "/v1/health":
            self._send_json(200, {"ok": True, "f16": True, "backend": BACKEND})
            return
        self._passthrough()

    def do_POST(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return
        path = self.path.split("?")[0]
        if path != "/api/embed":
            self._passthrough()
            return
        body = self._read_body()
        code, _headers, payload = self._upstream("/api/embed", body)
        if code != 200:
            try:
                err = json.loads(payload)
            except Exception:  # noqa: BLE001 — upstream sent non-JSON
                err = {"error": payload[:500].decode("utf-8", "replace")}
            self._send_json(code, err)
            return
        arr = np.asarray(json.loads(payload)["embeddings"], dtype=np.float16)
        out = arr.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Shape", f"{arr.shape[0]},{arr.shape[1]}")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        _log.debug("http %s", fmt % args)


def _start_tunnel(port: int, work: Path) -> str | None:
    if not _which("cloudflared"):
        _log.error("cloudflared not found — run the kaggle_start.sh dependency step first")
        return None
    log_path = work / "cloudflared_f16.log"
    work.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(  # noqa: S603
        ["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
        stdout=open(log_path, "w"),  # noqa: SIM115 — child owns the fd
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(40):
        time.sleep(1)
        try:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log_path.read_text())
        except OSError:
            m = None
        if m:
            return m.group(0)
    return None


def _which(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


def main() -> int:
    global TOKEN, BACKEND
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--backend", default=os.environ.get("OLLAMA_BACKEND", "http://localhost:11434"))
    ap.add_argument("--token", default=os.environ.get("KAGGLE_API_TOKEN", ""))
    ap.add_argument("--tunnel", action="store_true", help="also start cloudflared and print the URL")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [f16_shim] %(message)s",
    )
    TOKEN = args.token or secrets.token_hex(16)
    BACKEND = args.backend.rstrip("/")

    tunnel = _start_tunnel(args.port, Path(os.environ.get("KAGGLE_WORK_DIR", "kaggle_work"))) if args.tunnel else None

    print("==================================================")
    print(f"  token:  {TOKEN}")
    print(f"  shim:   http://{args.host}:{args.port}  (backend {BACKEND})")
    if args.tunnel:
        print(f"  tunnel: {tunnel or 'FAILED — see kaggle_work/cloudflared_f16.log'}")
        print(f"  health: {tunnel}/v1/health" if tunnel else "  health: n/a")
        print(f"  embed:  {tunnel}/api/embed  (float16 octet-stream)" if tunnel else "")
    print("  client: EMBED_F16=1 OLLAMA_URL=<tunnel or host:port> KAGGLE_API_TOKEN=<token>")
    print("==================================================", flush=True)

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
