#!/usr/bin/env python3
"""Job API + Ollama proxy for running BaryGraph stages on a remote GPU (e.g. Kaggle).

Endpoints (all behind a bearer token):
    GET  /v1/health                      → {"ok": true, ...}
    POST /v1/upload (multipart: file)    → {"path": ...}  (into WORK_DIR)
    POST /v1/run  {"stage": ..., "args": [...], "env": {...}} → {"job_id": ...}
    GET  /v1/jobs                        → [{"job_id", "stage", "status", ...}]
    GET  /v1/jobs/{id}                   → status + last log lines
    GET  /v1/download?path=relpath       → file bytes (under WORK_DIR only)

Ollama proxy (also behind same bearer token):
    GET  /api/*                          → forward to OLLAMA_URL_BASE/api/*
    POST /api/*                          → forward to OLLAMA_URL_BASE/api/*

Runs stages as subprocesses; forwards Ollama API calls to the local Ollama
on Kaggle so remote clients (e.g. s04 on laptop) can use GPU embeddings.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

TOKEN = os.environ.get("KAGGLE_API_TOKEN", "dev-token")
WORK = Path(os.environ.get("KAGGLE_WORK_DIR", "kaggle_work")).resolve()
REPO = Path(os.environ.get("KAGGLE_REPO_DIR", Path(__file__).resolve().parent.parent))
OLLAMA_URL_BASE = os.environ.get("OLLAMA_URL_BASE", "http://localhost:11434")
OLLAMA_FORWARD_URL = os.environ.get("OLLAMA_FORWARD_URL", OLLAMA_URL_BASE)

WORK.mkdir(parents=True, exist_ok=True)
REPO.mkdir(parents=True, exist_ok=True)

# stage shortname → module
STAGES = {
    "s01_parse": "scripts.s01_parse",
    "s02_embed": "scripts.s02_embed",
    "s03_insert_nodes": "scripts.s03_insert_nodes",
    "s04_l15_edges": "scripts.s04_l15_edges",
    "s05_word_vectors": "scripts.s05_word_vectors",
    "s06_l14_edges": "scripts.s06_l14_edges",
    "s07_orphan_reentry": "scripts.s07_orphan_reentry",
    "s08_metabary": "scripts.s08_metabary",
    "s10_index": "scripts.s10_index",
    "ingest_batch": "scripts.ingest_batch",
}

JOBS: dict[str, dict] = {}


def _now() -> float:
    return time.time()


def _run_job(job_id: str, stage: str, args: list[str], extra_env: dict[str, str]) -> None:
    log_path = WORK / f"{job_id}.log"
    env = dict(os.environ)
    env["OLLAMA_URL"] = f"{OLLAMA_FORWARD_URL}/api/embed"
    if (WORK / "kaikki.jsonl").exists():
        env["KAIKKI_PATH"] = str(WORK / "kaikki.jsonl")
    env["PARSED_DIR"] = str(WORK / "parsed")
    env["PIPELINE_STATE_DIR"] = str(WORK / "pipeline_state")
    env["EMBED_MODEL"] = env.get("EMBED_MODEL", "qwen3-embedding:0.6b")
    env["EMBED_DIM"] = env.get("EMBED_DIM", "1024")
    env.update(extra_env)
    cmd = [os.environ.get("PYTHON", "python3"), "-m", STAGES[stage], *args]
    JOBS[job_id].update(status="running", cmd=" ".join(cmd), started=_now())
    try:
        with open(log_path, "w") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            JOBS[job_id]["pid"] = proc.pid
            rc = proc.wait()
        JOBS[job_id].update(status="done" if rc == 0 else "failed", returncode=rc, ended=_now())
    except Exception as e:
        JOBS[job_id].update(status="failed", error=str(e), ended=_now())


class Handler(BaseHTTPRequestHandler):
    server_version = "kaggle-embed-api/0.2"

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

    def _proxy_ollama(self):
        if not self._auth_ok():
            return
        url = urlparse(self.path)
        target = f"{OLLAMA_FORWARD_URL}{url.path}"
        if url.query:
            target += f"?{url.query}"
        content_length = self.headers.get("Content-Length")
        data = None
        if content_length is not None and int(content_length) > 0:
            data = self.rfile.read(int(content_length))
        req = Request(target, data=data, method=self.command)
        for k, v in self.headers.items():
            if k.lower() in ("host", "connection"):
                continue
            req.add_header(k, v)
        try:
            with urlopen(req, timeout=300) as resp:
                self.send_response(resp.getcode())
                for k, v in resp.headers.items():
                    if k.lower() in ("connection", "transfer-encoding"):
                        continue
                    self.send_header(k, v)
                self.end_headers()
                while chunk := resp.read(1 << 20):
                    self.wfile.write(chunk)
        except Exception as e:
            self._send(502, {"error": str(e)})

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            self._proxy_ollama()
            return
        self._send(405, {"error": "method not allowed"})

    def do_GET(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            self._proxy_ollama()
            return
        if url.path == "/v1/health":
            self._send(200, {"ok": True, "ollama_url": f"{OLLAMA_FORWARD_URL}/api/embed", "repo": str(REPO)})
            return
        if url.path == "/v1/jobs":
            jobs = [dict(v) for v in JOBS.values()]
            self._send(200, jobs)
            return
        m = re.match(r"^/v1/jobs/([a-z0-9-]+)$", url.path)
        if m:
            jid = m.group(1)
            if jid not in JOBS:
                self._send(404, {"error": "job not found"})
                return
            j = dict(JOBS[jid])
            j["log_tail"] = _tail(WORK / f"{jid}.log")
            self._send(200, j)
            return
        if url.path == "/v1/download":
            qs = parse_qs(url.query)
            if "path" not in qs or not qs["path"]:
                self._send(400, {"error": "missing path"})
                return
            rel = qs["path"][0]
            p = (WORK / rel).resolve()
            try:
                if not str(p).startswith(str(WORK)):
                    self._send(403, {"error": "forbidden"})
                    return
            except Exception:
                self._send(403, {"error": "forbidden"})
                return
            self._send_file(p)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._auth_ok():
            return
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            self._proxy_ollama()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            self._send(400, {"error": "invalid json"})
            return
        if url.path == "/v1/upload":
            self._send(501, {"error": "upload not implemented"})
            return
        if url.path == "/v1/run":
            stage = payload.get("stage")
            args = payload.get("args") or []
            extra_env = payload.get("env") or {}
            if stage not in STAGES:
                self._send(400, {"error": f"unknown stage: {stage}"})
                return
            jid = str(uuid.uuid4())
            JOBS[jid] = dict(job_id=jid, stage=stage, status="queued", args=list(args), env=dict(extra_env), created=_now())
            t = threading.Thread(target=_run_job, args=(jid, stage, list(args), dict(extra_env)), daemon=True)
            t.start()
            self._send(200, {"job_id": jid})
            return
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # noqa: A003
        return


def main() -> None:
    host = os.environ.get("KAGGLE_API_HOST", "127.0.0.1")
    port = int(os.environ.get("KAGGLE_API_PORT", "8765"))
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"embed api+proxy on http://{host}:{port} (token set) work={WORK}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
