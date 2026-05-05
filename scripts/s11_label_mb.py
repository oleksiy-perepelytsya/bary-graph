"""Semantic labeling of MetaBary records via Vertex AI Batch API.

Selects up to --batch-size unlabeled MetaBary docs at a given level,
builds a Vertex AI Batch API input JSONL, submits the job, waits for
completion, then writes semantic_label + label_vector back to MongoDB.

GCS is used as a transient hub: input and output files are deleted after
a successful MongoDB write. The output JSONL is always kept locally in
pipeline_state/ so the write step can be replayed without re-calling the API.

Usage:
    python -m scripts.s11_label_mb [options]

Options:
    --level INT           MetaBary level to process (10–13, default: 13)
    --batch-size INT      Max records per job (default: 10000)
    --test                Use 1000 records and skip MongoDB writes
    --model STR           Override VERTEX_MODEL from settings / .env
    --poll-interval INT   Seconds between job status polls (default: 30)
    --resume              Load an in-progress job from pipeline_state/s11_state.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from bson import ObjectId
from pymongo import UpdateOne
from pymongo.collection import Collection

from lib.config import Settings
from lib.db import cm_leaf_words, get_collection
from lib.embed import get_embedder
from lib.log import setup_logging
from lib.vertex_label import VertexBatchClient, is_succeeded, is_terminal

_log = logging.getLogger(__name__)

_STATE_FILE = Path("pipeline_state/s11_state.json")
_TEST_LIMIT = 1000


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

def _triad_of(coll: Collection, mb_id: ObjectId, cm1_id: ObjectId, cm2_id: ObjectId) -> dict:
    """Fetch bridge doc and return triad with leaf words for all three branches."""
    bridge_doc = coll.find_one(
        {"parent_edge_id": mb_id, "_id": {"$nin": [cm1_id, cm2_id]}},
        {"_id": 1},
    )
    bridge_id = bridge_doc["_id"] if bridge_doc else None
    return {
        "child1": {"id": str(cm1_id), "words": sorted(cm_leaf_words(coll, cm1_id))},
        "child2": {"id": str(cm2_id), "words": sorted(cm_leaf_words(coll, cm2_id))},
        "bridge": {
            "id": str(bridge_id) if bridge_id else None,
            "words": sorted(cm_leaf_words(coll, bridge_id)) if bridge_id else [],
        },
    }


def _fetch_records(coll: Collection, level: int, limit: int) -> list[dict]:
    """Return unlabeled MB docs with triad words. Sorted by _id for reproducibility."""
    query = {
        "doc_type": "baryedge",
        "level": level,
        "semantic_label": {"$exists": False},
    }
    total = coll.count_documents(query)
    _log.info("unlabeled L%d MBs in DB: %d  (fetching up to %d)", level, total, limit)

    docs = list(
        coll.find(query, {"cm1_id": 1, "cm2_id": 1, "connection_strength": 1})
        .sort("_id", 1)
        .limit(limit)
    )

    records: list[dict] = []
    for i, doc in enumerate(docs):
        if i % 500 == 0 and i > 0:
            _log.info("  building triads %d/%d ...", i, len(docs))
        mb_id = doc["_id"]
        records.append({
            "id": str(mb_id),
            "level": level,
            "connection_strength": float(doc.get("connection_strength") or 0.0),
            "triad": _triad_of(coll, mb_id, doc["cm1_id"], doc["cm2_id"]),
        })

    _log.info("fetched %d records with triad words", len(records))
    return records


# ---------------------------------------------------------------------------
# JSONL build
# ---------------------------------------------------------------------------

def _build_jsonl(
    records: list[dict],
    vertex: VertexBatchClient,
    settings: Settings,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            line = vertex.format_record(
                mb_id=rec["id"],
                triad=rec["triad"],
                connection_strength=rec["connection_strength"],
                level=rec["level"],
                temperature=settings.vertex_temperature,
                frequency_penalty=settings.vertex_frequency_penalty,
            )
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    _log.info("wrote %d request lines → %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _extract_label(candidates: list[dict]) -> str | None:
    """Pull semantic_label from the first candidate's JSON text."""
    if not candidates:
        return None
    try:
        text = candidates[0]["content"]["parts"][0]["text"]
        data = json.loads(text)
        label = str(data.get("semantic_label", "")).strip()
        if 3 <= len(label) <= 200:
            return label
    except Exception:
        pass
    return None


def _parse_output_jsonl(path: Path) -> dict[str, str]:
    """Parse output JSONL → {mongo_id_str: semantic_label}. Skips failures."""
    results: dict[str, str] = {}
    errors = 0
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                key = obj.get("key")
                if not key:
                    errors += 1
                    continue
                # Check for API-level error in status
                status = obj.get("status") or {}
                if status.get("code") and status["code"] != 0:
                    _log.debug("API error for key %s: %s", key, status.get("message"))
                    errors += 1
                    continue
                label = _extract_label((obj.get("response") or {}).get("candidates") or [])
                if label:
                    results[key] = label
                else:
                    _log.debug("unparseable response for key %s", key)
                    errors += 1
            except Exception as exc:
                _log.debug("line parse error: %s", exc)
                errors += 1

    _log.info("parsed %d valid labels, %d errors/skips", len(results), errors)
    return results


# ---------------------------------------------------------------------------
# Embedding + MongoDB write
# ---------------------------------------------------------------------------

def _embed_in_chunks(texts: list[str], embedder: Any, chunk_size: int) -> np.ndarray:
    """Embed texts in chunks matching embed_batch_size. Returns (n, dim) float32 array."""
    parts: list[np.ndarray] = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        parts.append(embedder.embed(chunk))
        if start % (chunk_size * 20) == 0 and start > 0:
            _log.info("  embedded %d/%d labels ...", start, len(texts))
    return np.vstack(parts) if parts else np.empty((0, embedder.dim), dtype=np.float32)


def _write_to_mongo(
    coll: Collection,
    labels: dict[str, str],
    embedder: Any,
    model: str,
    embed_batch_size: int,
    dry_run: bool,
) -> int:
    ids = list(labels.keys())
    texts = [labels[k] for k in ids]

    _log.info("embedding %d labels via Ollama ...", len(texts))
    vectors = _embed_in_chunks(texts, embedder, embed_batch_size)

    if dry_run:
        _log.info("--test mode: skipping MongoDB write (%d records would be written)", len(ids))
        return 0

    now = datetime.now(timezone.utc)
    ops: list[UpdateOne] = []
    skipped = 0
    for mongo_id, label, vec_row in zip(ids, texts, vectors):
        if vec_row.shape[0] != embedder.dim:
            _log.warning("unexpected vector dim %d for %s — skipping", vec_row.shape[0], mongo_id)
            skipped += 1
            continue
        ops.append(UpdateOne(
            # Double guard: only write if semantic_label still absent
            {"_id": ObjectId(mongo_id), "semantic_label": {"$exists": False}},
            {"$set": {
                "semantic_label": label,
                "label_vector": vec_row.tolist(),
                "label_model": model,
                "label_updated_at": now,
            }},
        ))

    if not ops:
        _log.warning("no valid ops to write (skipped=%d)", skipped)
        return 0

    result = coll.bulk_write(ops, ordered=False)
    written = result.modified_count
    _log.info(
        "MongoDB write complete: %d modified, %d matched, %d skipped (dim/guard)",
        written, result.matched_count, skipped,
    )
    return written


# ---------------------------------------------------------------------------
# State persistence (for --resume)
# ---------------------------------------------------------------------------

def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def _load_state() -> dict | None:
    return json.loads(_STATE_FILE.read_text()) if _STATE_FILE.exists() else None


def _clear_state() -> None:
    _STATE_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m scripts.s11_label_mb",
        description="Label MetaBary records with Vertex AI Batch API",
    )
    p.add_argument("--level", type=int, default=13,
                   help="MetaBary level to process (10–13, default: 13)")
    p.add_argument("--batch-size", type=int, default=10_000,
                   help="Max records per batch job (default: 10000)")
    p.add_argument("--test", action="store_true",
                   help="Use 1000 records and skip MongoDB writes")
    p.add_argument("--model", type=str, default=None,
                   help="Override VERTEX_MODEL (e.g. gemini-2.0-flash-001)")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between job status polls (default: 30)")
    p.add_argument("--resume", action="store_true",
                   help="Resume an in-progress job from pipeline_state/s11_state.json")
    args = p.parse_args(argv)

    settings = Settings.load()
    setup_logging(settings.log_level)

    if not (10 <= args.level <= 13):
        _log.error("--level must be between 10 and 13")
        sys.exit(1)

    if args.model:
        settings = dataclasses.replace(settings, vertex_model=args.model)

    if not settings.vertex_project:
        _log.error("VERTEX_PROJECT is not set — add it to .env or environment")
        sys.exit(1)
    if not settings.vertex_gcs_bucket:
        _log.error("VERTEX_GCS_BUCKET is not set — add it to .env or environment")
        sys.exit(1)

    effective_limit = _TEST_LIMIT if args.test else args.batch_size

    vertex = VertexBatchClient(
        project=settings.vertex_project,
        location=settings.vertex_location,
        model=settings.vertex_model,
        gcs_bucket=settings.vertex_gcs_bucket,
    )
    coll = get_collection(settings)
    embedder = get_embedder(settings)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    gcs_input_key = f"bary-label/{ts}-L{args.level}-input.jsonl"
    gcs_output_prefix = f"gs://{settings.vertex_gcs_bucket}/bary-label/{ts}-L{args.level}-output/"
    local_input = settings.pipeline_state_dir / f"s11_input_{ts}.jsonl"
    local_output = settings.pipeline_state_dir / f"s11_output_{ts}.jsonl"

    # ------------------------------------------------------------------
    # Resume: reload existing job instead of starting a new one
    # ------------------------------------------------------------------
    job_name: str | None = None
    input_uri: str | None = None
    output_prefix = gcs_output_prefix

    if args.resume:
        state = _load_state()
        if state:
            job_name = state.get("job_name")
            input_uri = state.get("input_uri")
            output_prefix = state.get("output_prefix", gcs_output_prefix)
            # Restore timestamps so local file names match the original run
            ts = state.get("ts", ts)
            local_output = settings.pipeline_state_dir / f"s11_output_{ts}.jsonl"
            _log.info("resuming job %s (level=%s)", job_name, state.get("level"))
        else:
            _log.warning("--resume: no state file found, starting fresh")

    # ------------------------------------------------------------------
    # Step 1–4: fetch → build JSONL → upload → submit  (skip if resuming)
    # ------------------------------------------------------------------
    if not job_name:
        records = _fetch_records(coll, args.level, effective_limit)
        if not records:
            _log.info("no unlabeled MBs at L%d — nothing to do", args.level)
            return

        _build_jsonl(records, vertex, settings, local_input)

        _log.info("uploading input JSONL to GCS ...")
        input_uri = vertex.upload_jsonl(local_input, gcs_input_key)

        _log.info("submitting batch job (model=%s) ...", settings.vertex_model)
        job = vertex.submit_batch(input_uri, output_prefix)
        job_name = job.name
        _save_state({
            "job_name": job_name,
            "input_uri": input_uri,
            "output_prefix": output_prefix,
            "level": args.level,
            "ts": ts,
        })
    else:
        _log.info("skipping fetch/upload/submit (resuming existing job)")

    # ------------------------------------------------------------------
    # Step 5: poll until terminal state
    # ------------------------------------------------------------------
    _log.info("polling job %s every %ds ...", job_name, args.poll_interval)
    while True:
        job = vertex.poll_job(job_name)
        from lib.vertex_label import _job_state_name
        state_name = _job_state_name(job)
        _log.info("  job state: %s", state_name)
        if is_terminal(job):
            break
        time.sleep(args.poll_interval)

    if not is_succeeded(job):
        _log.error("batch job did not succeed (state=%s) — state file kept for inspection",
                   _job_state_name(job))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: download output (concatenate shards → single local file)
    # ------------------------------------------------------------------
    output_blobs = vertex.list_output_blobs(output_prefix)
    if not output_blobs:
        _log.error("job succeeded but no output blobs found under %s", output_prefix)
        sys.exit(1)

    _log.info("downloading %d output shard(s) ...", len(output_blobs))
    local_output.parent.mkdir(parents=True, exist_ok=True)
    with local_output.open("w", encoding="utf-8") as out_fh:
        tmp = settings.pipeline_state_dir / f"s11_shard_{ts}.tmp"
        for blob_uri in output_blobs:
            vertex.download_blob(blob_uri, tmp)
            out_fh.write(tmp.read_text(encoding="utf-8"))
        tmp.unlink(missing_ok=True)
    _log.info("output saved locally: %s", local_output)

    # ------------------------------------------------------------------
    # Step 7: parse → embed → write
    # ------------------------------------------------------------------
    labels = _parse_output_jsonl(local_output)
    if not labels:
        _log.error("no valid labels parsed from %s — inspect the file before retrying", local_output)
        sys.exit(1)

    written = _write_to_mongo(
        coll, labels, embedder,
        model=settings.vertex_model,
        embed_batch_size=settings.embed_batch_size,
        dry_run=args.test,
    )

    # ------------------------------------------------------------------
    # Step 8: GCS cleanup (only after confirmed success)
    # ------------------------------------------------------------------
    if written > 0 or args.test:
        _log.info("cleaning up GCS ...")
        if input_uri:
            vertex.delete_blob(input_uri)
        deleted = vertex.delete_prefix(output_prefix)
        _log.info("deleted %d GCS object(s)", deleted + (1 if input_uri else 0))
    else:
        _log.warning("0 records written — GCS files kept for inspection: %s", output_prefix)

    _clear_state()
    _log.info(
        "done: level=L%d  written=%d  test=%s  output_local=%s",
        args.level, written, args.test, local_output,
    )


if __name__ == "__main__":
    run()
