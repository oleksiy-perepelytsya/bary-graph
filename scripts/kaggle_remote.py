#!/usr/bin/env python3
"""Local driver for the Kaggle embed-job API (scripts/kaggle_embed_server.py).

    # upload a stage input
    python3 -m scripts.kaggle_remote --base https://<tunnel>.trycloudflare.com \
        --token $TOK upload data/parsed/senses.jsonl --as senses.jsonl

    # run the embed stage on the Kaggle GPU
    python3 -m scripts.kaggle_remote --base ... --token $TOK run s02_embed \
        --args "--batch-size 1024"

    # watch until done, then pull the artifact
    python3 -m scripts.kaggle_remote --base ... --token $TOK poll <job_id>
    python3 -m scripts.kaggle_remote --base ... --token $TOK download \
        parsed/senses_embedded.jsonl -o data/parsed/senses_embedded.jsonl
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _req(base: str, token: str, method: str, path: str,
         data: bytes | None = None, headers: dict | None = None) -> tuple[int, bytes, dict]:
    h = {"Authorization": f"Bearer {token}"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})


def _multipart(path: Path, field: str = "file") -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
        f'Content-Type: {ctype}\r\n\r\n'
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + path.read_bytes() + tail, f"multipart/form-data; boundary={boundary}"


def main() -> int:
    p = argparse.ArgumentParser(prog="kaggle_remote")
    p.add_argument("--base", required=True)
    p.add_argument("--token", required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("file", type=Path)
    u.add_argument("--as", dest="remote_name", default=None)
    r = sub.add_parser("run")
    r.add_argument("stage")
    r.add_argument("--args", default="")
    r.add_argument("--env", action="append", default=[], help="KEY=VAL (repeatable)")
    po = sub.add_parser("poll")
    po.add_argument("job_id")
    po.add_argument("--interval", type=float, default=15.0)
    d = sub.add_parser("download")
    d.add_argument("path")
    d.add_argument("-o", "--out", type=Path, required=True)
    sub.add_parser("jobs")
    sub.add_parser("health")
    a = p.parse_args()

    if a.cmd == "health":
        s, b, _ = _req(a.base, a.token, "GET", "/v1/health")
        print(b.decode())
        return 0
    if a.cmd == "jobs":
        s, b, _ = _req(a.base, a.token, "GET", "/v1/jobs")
        print(json.dumps(json.loads(b), indent=2))
        return 0
    if a.cmd == "upload":
        body, ctype = _multipart(a.file)
        from urllib.parse import quote
        rel = a.remote_name or a.file.name
        s, b, _ = _req(a.base, a.token, "POST", f"/v1/upload?path={quote(rel)}", data=body,
                       headers={"Content-Type": ctype})
        print(b.decode())
        return 0 if s < 400 else 1
    if a.cmd == "run":
        env = {}
        for kv in a.env:
            k, _, v = kv.partition("=")
            env[k] = v
        req = {"stage": a.stage, "args": a.args.split() if a.args else [], "env": env}
        s, b, _ = _req(a.base, a.token, "POST", "/v1/run",
                       data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
        print(b.decode())
        return 0 if s < 400 else 1
    if a.cmd == "poll":
        while True:
            s, b, _ = _req(a.base, a.token, "GET", f"/v1/jobs/{a.job_id}")
            j = json.loads(b)
            print(j.get("status"), *j.get("log_tail", [])[-1:])
            if j.get("status") in ("done", "failed"):
                return 0 if j["status"] == "done" else 1
            time.sleep(a.interval)
    if a.cmd == "download":
        from urllib.parse import quote
        s, b, _ = _req(a.base, a.token, "GET", f"/v1/download?path={quote(a.path)}")
        if s >= 400:
            print(b.decode())
            return 1
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes(b)
        print(f"wrote {a.out} ({len(b)} bytes)")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
