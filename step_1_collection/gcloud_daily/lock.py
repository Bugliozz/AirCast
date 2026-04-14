"""GCS-backed distributed lock for the daily refresh job.

Simpler than the backfill checkpoint module — we only need the lock,
not the completed/failed date tracking (the file-existence check in
``main.py`` is the idempotency mechanism).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from google.cloud import storage as gcs

from config import GCS_BUCKET, GCS_LOCK_BLOB, LOCK_TIMEOUT_MINUTES

log = logging.getLogger(__name__)

_client: gcs.Client | None = None


def _bucket() -> gcs.Bucket:
    global _client
    if _client is None:
        _client = gcs.Client()
    return _client.bucket(GCS_BUCKET)


def acquire_lock() -> bool:
    """Return ``True`` on success.  A stale lock (older than
    ``LOCK_TIMEOUT_MINUTES``) is considered abandoned and overwritten."""
    blob = _bucket().blob(GCS_LOCK_BLOB)

    if blob.exists():
        blob.reload()
        created = blob.time_created or datetime.now(timezone.utc)
        age_min = (datetime.now(timezone.utc) - created).total_seconds() / 60
        if age_min < LOCK_TIMEOUT_MINUTES:
            log.warning("Lock held (%.1f min old) — another run in progress.", age_min)
            return False
        log.warning("Stale lock (%.1f min) — overwriting.", age_min)

    blob.upload_from_string(
        datetime.now(timezone.utc).isoformat(),
        content_type="text/plain",
    )
    return True


def release_lock() -> None:
    blob = _bucket().blob(GCS_LOCK_BLOB)
    if blob.exists():
        blob.delete()
