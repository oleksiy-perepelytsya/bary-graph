"""Routing contract for the Kaggle job API / Ollama proxy.

The proxy must forward Ollama's Jev decision endpoint (``/v1/systemone``)
alongside ``/api/*`` — remote ``TEV1_URL`` clients depend on it for tev1 role
decisions — while every other ``/v1/*`` route stays local to this job API.
"""

from __future__ import annotations

import os
import tempfile

# kaggle_embed_server mkdirs its work dir at import; keep that out of the repo.
os.environ.setdefault("KAGGLE_WORK_DIR", tempfile.mkdtemp(prefix="kaggle_proxy_test_"))

from scripts.kaggle_embed_server import _is_ollama_path  # noqa: E402


def test_api_routes_and_systemone_forward_to_ollama():
    assert _is_ollama_path("/api/embed")
    assert _is_ollama_path("/api/generate")
    assert _is_ollama_path("/v1/systemone")


def test_job_api_routes_stay_local():
    for path in (
        "/v1/health",
        "/v1/jobs",
        "/v1/jobs/abc-123",
        "/v1/run",
        "/v1/download",
        "/v1/systemone/bogus",  # only the exact endpoint proxies
        "/other",
    ):
        assert not _is_ollama_path(path), path
