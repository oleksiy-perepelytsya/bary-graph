from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import orjson

from lib.bary_vec import normalize
from lib.config import Settings

_log = logging.getLogger(__name__)

# Seconds to wait between retries (doubles on each attempt, capped at 60s).
_RETRY_BACKOFF_BASE = 5
_MAX_RETRIES = 3


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalized rows
        ...


class CachedEmbedder:
    """Disk-backed vector cache keyed by sha256(text).

    Wraps any Embedder: a text is embedded at most once across all stages and
    pipeline restarts. Worthwhile because kaikki repeats entries across
    etymology splits (~20% of sense embed texts are duplicates) and later
    stages re-embed type_texts.

    The sidecar file is append-only JSONL of {"h": <hex digest>, "v": [...]},
    flushed every FLUSH_EVERY new entries. A crash may truncate the final
    line — tolerated on load. Losing the sidecar is always safe: it only
    costs re-embedding.
    """

    FLUSH_EVERY = 512

    def __init__(self, inner: Embedder, path: str | Path):
        self.inner = inner
        self.dim = inner.dim
        self._path = Path(path)
        self._cache: dict[bytes, np.ndarray] = {}
        if self._path.exists():
            with self._path.open("rb") as f:
                for line in f:
                    try:
                        rec = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        break  # truncated tail from an interrupted flush
                    self._cache[bytes.fromhex(rec["h"])] = np.asarray(
                        rec["v"], dtype=np.float32
                    )
        self._fh = self._path.open("ab")
        self._pending: list[bytes] = []

    def __len__(self) -> int:
        return len(self._cache)

    def _flush(self) -> None:
        if self._pending:
            self._fh.write(b"".join(self._pending))
            self._fh.flush()
            self._pending.clear()

    def close(self) -> None:
        self._flush()
        self._fh.close()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        hashes = [hashlib.sha256(t.encode("utf-8")).digest() for t in texts]
        miss_idx = [i for i, h in enumerate(hashes) if h not in self._cache]
        if miss_idx:
            fresh = self.inner.embed([texts[i] for i in miss_idx])
            for i, v in zip(miss_idx, fresh, strict=True):
                h = hashes[i]
                self._cache[h] = v
                self._pending.append(orjson.dumps({"h": h.hex(), "v": v.tolist()}) + b"\n")
            if len(self._pending) >= self.FLUSH_EVERY:
                self._flush()
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, h in enumerate(hashes):
            out[i] = self._cache[h]
        return out

    def vector_for(self, text: str) -> np.ndarray:
        """Return the cached vector for one text (embedding it on a cold miss).

        Unlike embed(), this returns a reference into the cache rather than a
        copy, so callers that only need per-key reads (e.g. L15 orphan
        re-entry) don't double the resident vector memory.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        if h not in self._cache:
            self.embed([text])
        return self._cache[h]


class OllamaEmbedder:
    def __init__(self, settings: Settings):
        self.dim = settings.embed_dim
        self._url = settings.ollama_url.rstrip("/") + "/api/embed"
        self._model = settings.embed_model
        # Use a per-request client to avoid stale connection pool state
        # on long-running jobs where ollama may restart.  600s covers
        # ollama cold-start after idle (model load can take 60-120s+ on
        # this CPU, and EMBED_BATCH_SIZE=512 batches add to it).
        self._timeout = min(settings.embed_timeout_seconds, 600)

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self._timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                client = self._client()
                try:
                    resp = client.post(self._url, json={
                        "model": self._model,
                        "input": texts,
                        # Keep the model warm across long phases: without this,
                        # any idle gap (e.g. between the ANN sweep and Phase A,
                        # or between insert batches) longer than ollama's
                        # default keep_alive (~5 min) makes the next request
                        # race a cold reload that can outlast the transport
                        # timeout. 6h >> any inter-phase gap we have.
                        "keep_alive": "6h",
                    })
                    resp.raise_for_status()
                finally:
                    client.close()
                break
            except httpx.HTTPStatusError:
                raise
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = min(_RETRY_BACKOFF_BASE * (2 ** attempt), 60)
                    _log.warning(
                        "embed transport error (attempt %d/%d): %s — "
                        "retrying in %ds", attempt + 1, _MAX_RETRIES + 1,
                        exc, wait,
                    )
                    time.sleep(wait)
                    continue
                _log.error("embed batch failed after %d attempts", _MAX_RETRIES + 1)
                raise last_exc from None
        data = resp.json()
        arr = np.asarray(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms

    def vector_for(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FakeEmbedder:
    """Deterministic, offline embedder for CI and unit tests."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little")
            rng = np.random.default_rng(seed)
            out[i] = normalize(rng.standard_normal(self.dim).astype(np.float32))
        return out

    def vector_for(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def get_embedder(settings: Settings) -> Embedder:
    if settings.fake_embed:
        embedder: Embedder = FakeEmbedder(dim=settings.embed_dim)
    else:
        embedder = OllamaEmbedder(settings)
    if settings.embed_cache_file is not None:
        return CachedEmbedder(embedder, settings.embed_cache_file)
    return embedder
