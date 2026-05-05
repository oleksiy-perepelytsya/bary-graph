"""Vertex AI Batch API client for semantic labeling of MetaBary records.

Uses the google-genai SDK with Vertex AI backend (ADC auth — no key file
needed on a GCP VM). The system prompt is defined once as a module constant
and embedded in every request line so the full batch shares the same prompt
without changes mid-run.

Install extra deps:
    pip install -e ".[vertex]"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Defined once; embedded identically in every JSONL request line for the run.
SYSTEM_PROMPT = """\
You are a semantic annotator for a dictionary knowledge graph.

Each query describes a MetaBary — a cluster of three related word groups:
- child1: words forming the first semantic branch
- child2: words forming the second semantic branch
- bridge: words forming the connecting concept

Your task: write one short phrase (5–10 words) that names the shared semantic
theme tying all three groups together. Be specific and concrete. Use plain English.

Respond with valid JSON only, no prose:
{"semantic_label": "your phrase here"}
"""

# States that mean the job will not progress further.
_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_CANCELLING",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
})


def _job_state_name(job: Any) -> str:
    """Return the string name of a BatchJob state, regardless of SDK version."""
    state = job.state
    if hasattr(state, "name"):
        return state.name
    return str(state).split(".")[-1]  # e.g. "JobState.JOB_STATE_SUCCEEDED" → "JOB_STATE_SUCCEEDED"


def is_terminal(job: Any) -> bool:
    return _job_state_name(job) in _TERMINAL_STATES


def is_succeeded(job: Any) -> bool:
    name = _job_state_name(job)
    return name in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


class VertexBatchClient:
    """Thin wrapper around google-genai Batch API + google-cloud-storage.

    Auth: relies on Application Default Credentials (ADC). On a GCP VM with a
    service account, or after `gcloud auth application-default login`, no
    credential files are required.
    """

    def __init__(self, project: str, location: str, model: str, gcs_bucket: str) -> None:
        # Lazy-import so the rest of the codebase doesn't require google-genai.
        from google import genai
        from google.cloud import storage as gcs_mod

        self.project = project
        self.location = location
        self.model = model
        self.bucket_name = gcs_bucket.removeprefix("gs://").split("/")[0]

        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._gcs = gcs_mod.Client(project=project)

    # ------------------------------------------------------------------
    # Request formatting
    # ------------------------------------------------------------------

    def format_record(
        self,
        mb_id: str,
        triad: dict[str, Any],
        connection_strength: float,
        level: int,
        temperature: float,
        frequency_penalty: float,
    ) -> dict[str, Any]:
        """Build one JSONL request line for the Vertex AI Batch API."""
        child1 = ", ".join(triad["child1"]["words"][:10])
        child2 = ", ".join(triad["child2"]["words"][:10])
        bridge = ", ".join(triad["bridge"]["words"][:10])
        user_text = (
            f"Level: {level}\n"
            f"Child1: {child1}\n"
            f"Child2: {child2}\n"
            f"Bridge: {bridge}\n"
            f"Connection strength: {connection_strength:.3f}\n\n"
            "Respond with JSON only."
        )
        return {
            "key": mb_id,
            "request": {
                "system_instruction": {
                    "role": "system",
                    "parts": [{"text": SYSTEM_PROMPT}],
                },
                "contents": [
                    {"role": "user", "parts": [{"text": user_text}]}
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "maxOutputTokens": 80,
                    "temperature": temperature,
                    "frequencyPenalty": frequency_penalty,
                },
            },
        }

    # ------------------------------------------------------------------
    # GCS helpers
    # ------------------------------------------------------------------

    def _blob_key(self, gcs_uri: str) -> str:
        """Strip gs://bucket/ prefix to get the blob key."""
        return gcs_uri.removeprefix(f"gs://{self.bucket_name}/")

    def upload_jsonl(self, local_path: Path, gcs_key: str) -> str:
        """Upload a local JSONL file to GCS. Returns gs:// URI."""
        bucket = self._gcs.bucket(self.bucket_name)
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(local_path), content_type="application/jsonl")
        uri = f"gs://{self.bucket_name}/{gcs_key}"
        _log.info("uploaded %s → %s (%d bytes)", local_path.name, uri, local_path.stat().st_size)
        return uri

    def list_output_blobs(self, gcs_output_prefix: str) -> list[str]:
        """Return gs:// URIs of all .jsonl output blobs under prefix."""
        prefix = self._blob_key(gcs_output_prefix)
        bucket = self._gcs.bucket(self.bucket_name)
        return [
            f"gs://{self.bucket_name}/{b.name}"
            for b in bucket.list_blobs(prefix=prefix)
            if b.name.endswith(".jsonl")
        ]

    def download_blob(self, gcs_uri: str, local_path: Path) -> None:
        """Download a single GCS blob to a local file."""
        key = self._blob_key(gcs_uri)
        self._gcs.bucket(self.bucket_name).blob(key).download_to_filename(str(local_path))

    def delete_blob(self, gcs_uri: str) -> None:
        """Delete a single GCS blob. Silently ignores 404."""
        try:
            key = self._blob_key(gcs_uri)
            self._gcs.bucket(self.bucket_name).blob(key).delete()
        except Exception as exc:
            _log.debug("delete_blob %s: %s", gcs_uri, exc)

    def delete_prefix(self, gcs_prefix: str) -> int:
        """Delete all blobs under a GCS prefix. Returns count deleted."""
        prefix = self._blob_key(gcs_prefix)
        bucket = self._gcs.bucket(self.bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            try:
                blob.delete()
            except Exception as exc:
                _log.debug("delete blob %s: %s", blob.name, exc)
        return len(blobs)

    # ------------------------------------------------------------------
    # Batch job lifecycle
    # ------------------------------------------------------------------

    def submit_batch(self, gcs_input_uri: str, gcs_output_prefix: str) -> Any:
        """Submit a batch job. Returns the BatchJob object (has .name)."""
        from google.genai.types import CreateBatchJobConfig

        job = self._client.batches.create(
            model=self.model,
            src=gcs_input_uri,
            config=CreateBatchJobConfig(dest=gcs_output_prefix),
        )
        _log.info("batch job created: %s  state=%s", job.name, _job_state_name(job))
        return job

    def poll_job(self, job_name: str) -> Any:
        """Fetch the current state of a batch job."""
        return self._client.batches.get(name=job_name)
