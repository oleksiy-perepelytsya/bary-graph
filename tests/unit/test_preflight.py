"""Unit-level coverage for individual preflight checks (no services)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.config import Settings
from scripts import preflight


def _settings(monkeypatch, kaikki: Path, parsed: Path, state: Path) -> Settings:
    monkeypatch.setenv("BARY_FAKE_EMBED", "1")
    monkeypatch.setenv("KAIKKI_PATH", str(kaikki))
    monkeypatch.setenv("PARSED_DIR", str(parsed))
    monkeypatch.setenv("PIPELINE_STATE_DIR", str(state))
    return Settings.load()


def test_embed_probe_dim(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path / "x.jsonl", tmp_path / "p", tmp_path / "s")
    c = preflight._check_embed_dim(s)
    assert c.ok, c.detail
    assert "1024" in c.detail


def test_kaikki_missing(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path / "missing.jsonl", tmp_path / "p", tmp_path / "s")
    c = preflight._check_kaikki(s)
    assert not c.ok
    assert "not found" in c.detail


def test_kaikki_too_small(monkeypatch, tmp_path):
    p = tmp_path / "tiny.jsonl"
    p.write_text("{}\n")
    s = _settings(monkeypatch, p, tmp_path / "p", tmp_path / "s")
    c = preflight._check_kaikki(s)
    assert not c.ok
    assert "partial" in c.detail or "bytes" in c.detail


def test_dirs_writable(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path / "x.jsonl", tmp_path / "p", tmp_path / "s")
    c = preflight._check_dirs_writable(s)
    assert c.ok
    assert (tmp_path / "p").is_dir() and (tmp_path / "s").is_dir()


def test_ollama_skipped_in_fake_mode(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path / "x.jsonl", tmp_path / "p", tmp_path / "s")
    c = preflight._check_ollama(s)
    assert c.ok
    assert "skipped" in c.detail


def test_run_fails_without_kaikki(monkeypatch, tmp_path, caplog):
    # mongo.reachable will fail in this env too, but the point is run() returns 1
    # whenever *any* check fails.
    _settings(monkeypatch, tmp_path / "missing.jsonl", tmp_path / "p", tmp_path / "s")
    monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:1/?directConnection=true&serverSelectionTimeoutMS=200")
    rc = preflight.run()
    assert rc == 1


@pytest.mark.parametrize("check", [
    preflight._check_embed_dim,
    preflight._check_dirs_writable,
    preflight._check_ollama,
])
def test_checks_return_check_dataclass(monkeypatch, tmp_path, check):
    s = _settings(monkeypatch, tmp_path / "x.jsonl", tmp_path / "p", tmp_path / "s")
    c = check(s)
    assert hasattr(c, "ok") and hasattr(c, "name")
