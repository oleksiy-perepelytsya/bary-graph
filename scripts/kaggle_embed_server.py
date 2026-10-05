#!/usr/bin/env python3
"""Tiny job API to run BaryGraph embed stages on a remote GPU (e.g. Kaggle).

Endpoints (all behind a bearer token):

    GET  /v1/health                      → {"ok": true, ...}
    POST /v1/upload (multipart: file)    → {"path": ...}  (into WORK_DIR)
    POST /v1/run  {"stage": "s02_embed", "args": [...], "env": {...}}
                                         → {"job_id": ...}
    GET  /v1/jobs                        → [{"job_id", "stage", "status", ...}]
    GET  /v1/jobs/{id}                   → status + last log lines
    GET  /v1/download?path=relpath       → file bytes (under WORK_DIR only)

Runs stages as subprocesses with the repo as CWD, streaming logs to
WORK_DIR/<job_id>.log. The embed-heavy stages on Kaggle should use the
GPU ollama on the same machine (OLLAMA_URL env passed through).

Env:
    KAGGLE_API_TOKEN   bearer token required on every request (default: dev-token)
    KAGGLE_WORK_DIR    where uploads/logs/outputs live (default: ./kaggle_work)
    KAGGLE_REPO_DIR    repo root the stages run from (default: parent of scripts/)
    OLLAMA_URL         forwarded to stage subprocesses (default: http://localhost:11434)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TOKEN = os.environ.get("KAGGLE_API_TOKEN", "dev-token")
WORK = Path(os.environ.get("KAGGLE_WORK_DIR", "kaggle_work")).resolve()
REPO = Path(os.environ.get("KAGGLE_REPO_DIR", Path(__file__).resolve().parent.parent))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# stage shortname → module; extend as the GPU workflow grows
STAGES = {
    "s01_parse": "scripts.s01_parse",
    "s02_embed": "scripts.s02_embed",
    "s04_l15_edges": "scripts.s04_l15_edges",
}

JOBS: dict[str, dict] = {}


def _now() -> float:
    return time.time()


def _tail(path: Path, n: int = 20) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return lines[-n:]
    except FileNotFoundError:
        return []


def _run_job(job_id: str, stage: str, args: list[str], extra_env: dict[str, str]) -> None:
    log_path = WORK / f"{job_id}.log"
    env = dict(os.environ)
    env["OLLAMA_URL"] = OLLAMA_URL
    if (WORK / "kaikki.jsonl").exists():
        env["KAIKKI_PATH"] = str(WORK / "kaikki.jsonl")
    env["PARSED_DIR"] = str(WORK / "parsed")
    env["PIPELINE_STATE_DIR"] = str(WORK / "pipeline_state")
    env.update(extra_env)
    cmd = [os.environ.get("PYTHON", "python3"), "-m", STAGES[stage], *args]
    JOBS[job_id].update(status="running", cmd=" ".join(cmd), started=_now())
    with open(log_path, "w") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        JOBS[job_id]["pid"] = proc.pid
        rc = proc.wait()
    JOBS[job_id].update(status="done" if rc == 0 else "failed", returncode=rc, ended=_now())


class Handler(BaseHTTPRequestHandler):
    server_version = "kaggle-embed-api/0.1"

    # ---- helpers -------------------------------------------------------
    def _auth_ok(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {TOKEN}":
            return True
        self._send(401, {"error": "unauthorized"})
        return False

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send(404, {"error": f"not found: {path.name}"})
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                self.wfile.write(chunk)

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return
        url = urlparse(self.path)
        if url.path == "/v1/download":
            qs = parse_qs(url.query)
            p = (WORK / qs.get("path", [""])[0]).resolve()
            if not str(p).startswith(str(WORK)) or not p.is_file():
                self._send(404, {"error": "not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(p.stat().st_size))
            self.end_headers()
        else:
            self._send(405, {"error": "HEAD not supported here"})

    # ---- routes --------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if url.path == "/v1/health":
            self._send(200, {"ok": True, "ollama_url": OLLAMA_URL, "repo": str(REPO)})
        elif url.path == "/v1/jobs":
            self._send(200, [
                {k: j[k] for k in ("job_id", "stage", "status", "returncode") if k in j}
                | {"job_id": jid}
                for jid, j in JOBS.items()
            ])
        elif len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
            j = JOBS.get(parts[2])
            if j is None:
                self._send(404, {"error": "no such job"})
            else:
                self._send(200, j | {"log_tail": _tail(WORK / f"{parts[2]}.log")})
        elif url.path == "/v1/download":
            qs = parse_qs(url.query)
            rel = qs.get("path", [""])[0]
            p = (WORK / rel).resolve()
            if not str(p).startswith(str(WORK)):
                self._send(403, {"error": "path escapes work dir"})
                return
            self._send_file(p)
        else:
            self._send(404, {"error": url.path})

    def do_POST(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return
        url = urlparse(self.path)
        if url.path == "/v1/upload":
            ctype = self.headers.get("Content-Type", "")
            m = re.search(r"boundary=(.+)", ctype)
            if not m:
                self._send(400, {"error": "expected multipart/form-data"})
                return
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            boundary = m.group(1).encode()
            for segment in body.split(b"--" + boundary):
                if b"filename=" not in segment:
                    continue
                header, _, content = segment.partition(b"\r\n\r\n")
                fname_m = re.search(rb'filename="([^"]+)"', header)
                if not fname_m:
                    continue
                fname = Path(fname_m.group(1).decode()).name
                # optional server-side subpath via ?path=parsed/senses.jsonl
                qs = parse_qs(url.query)
                rel = qs.get("path", [fname])[0]
                dest = (WORK / rel).resolve()
                if not str(dest).startswith(str(WORK)):
                    self._send(403, {"error": "path escapes work dir"})
                    return
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content.rstrip(b"\r\n"))
                self._send(200, {"path": str(dest.relative_to(WORK)), "bytes": dest.stat().st_size})
                return
            self._send(400, {"error": "no file part"})
        elif url.path == "/v1/run":
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except Exception:
                self._send(400, {"error": "invalid json"})
                return
            stage = req.get("stage")
            if stage not in STAGES:
                self._send(400, {"error": f"unknown stage {stage!r}; have {sorted(STAGES)}"})
                return
            job_id = uuid.uuid4().hex[:8]
            JOBS[job_id] = {
                "stage": stage, "args": req.get("args", []),
                "status": "queued", "created": _now(),
            }
            threading.Thread(
                target=_run_job,
                args=(job_id, stage, req.get("args", []), req.get("env", {})),
                daemon=True,
            ).start()
            self._send(202, {"job_id": job_id, "stage": stage})
        else:
            self._send(404, {"error": url.path})

    def log_message(self, fmt, *args):  # quieter logs
        pass


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "parsed").mkdir(exist_ok=True)
    port = int(os.environ.get("KAGGLE_API_PORT", "8765"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    tok = "set" if TOKEN != "dev-token" else "DEV-TOKEN"
    print(f"kaggle embed api on :{port} | work={WORK} repo={REPO} token={tok}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
