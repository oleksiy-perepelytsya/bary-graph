from __future__ import annotations

import os
import uuid

import pytest

from lib.config import Settings
from lib.embed import FakeEmbedder


@pytest.fixture
def tmp_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_STATE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def settings(tmp_state_dir, monkeypatch) -> Settings:
    monkeypatch.setenv("BARY_FAKE_EMBED", "1")
    return Settings.load()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def mongo_test_db(monkeypatch):
    """Integration-only: yields a fresh collection in an isolated project-owned DB.

    Safety invariants:
      * DB name ALWAYS starts with ``settings.mongo_test_db_prefix`` (default
        ``barygraph_test_``). The teardown refuses to drop anything else.
      * Target Mongo URI must be the project port (default 27117) or an
        explicit override; we never silently fall back to the default 27017
        to avoid touching an unrelated Mongo that happens to be listening.
    """
    from lib.db import get_client, get_collection

    prefix = os.environ.get("MONGO_TEST_DB_PREFIX", "barygraph_test_")
    db_name = f"{prefix}{uuid.uuid4().hex[:8]}"
    assert db_name.startswith(prefix), "invariant: test DB name must carry project prefix"

    monkeypatch.setenv("MONGO_DB", db_name)
    uri = os.environ.get("MONGO_URI")
    if not uri:
        pytest.skip(
            "MONGO_URI not set; integration tests require a project-owned Mongo "
            "(run `make up` to start the compose service on port 27117)."
        )

    s = Settings.load()
    coll = get_collection(s)
    try:
        yield coll
    finally:
        # Guard: refuse to drop any DB that doesn't carry the project prefix.
        if not db_name.startswith(s.mongo_test_db_prefix):
            raise RuntimeError(
                f"refusing to drop DB '{db_name}' — missing prefix "
                f"'{s.mongo_test_db_prefix}'"
            )
        get_client(s).drop_database(db_name)
