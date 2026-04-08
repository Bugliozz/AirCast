"""GCS-backed checkpoint and distributed lock.

The checkpoint tracks which dates have been fully collected so the job can
resume exactly where it left off after a crash, timeout, or circuit-break.

The lock prevents two Cloud Run executions from overlapping.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from google.cloud import storage as gcs

from config import (
    GCS_BUCKET,
    GCS_CHECKPOINT_BLOB,
    GCS_LOCK_BLOB,
    LOCK_TIMEOUT_MINUTES,
)

log = logging.getLogger(__name__)

# Module-level lazy singleton for the GCS client.
_client: gcs.Client | None = None


def _get_client() -> gcs.Client:
    global _client
    if _client is None:
        _client = gcs.Client()
    return _client


def _bucket() -> gcs.Bucket:
    return _get_client().bucket(GCS_BUCKET)


# -- Distributed lock ---------------------------------------------------------

def acquire_lock() -> bool:
    """Try to acquire a GCS-based lock.  Returns *False* if another
    instance already holds a non-stale lock."""
    bucket = _bucket()
    blob = bucket.blob(GCS_LOCK_BLOB)

    if blob.exists():
        blob.reload()
        created = blob.time_created
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - created).total_seconds()

        if age_s < LOCK_TIMEOUT_MINUTES * 60:
            log.warning(
                "Lock held (age %.0fs < %ds) — another instance may be running.",
                age_s, LOCK_TIMEOUT_MINUTES * 60,
            )
            return False

        log.info("Stale lock detected (age %.0fs) — overwriting.", age_s)
        blob.delete()

    blob.upload_from_string(datetime.now(timezone.utc).isoformat())
    log.info("Lock acquired.")
    return True


def release_lock() -> None:
    """Delete the lock blob (idempotent)."""
    blob = _bucket().blob(GCS_LOCK_BLOB)
    if blob.exists():
        blob.delete()
    log.info("Lock released.")


# -- Checkpoint ---------------------------------------------------------------

class Checkpoint:
    """Tracks completed and failed dates for the backfill."""

    def __init__(self) -> None:
        self.completed: set[str] = set()
        self.failed: dict[str, str] = {}   # date -> last error message
        self.started_at: str = datetime.now(timezone.utc).isoformat()

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_dates": sorted(self.completed),
            "failed_dates": dict(sorted(self.failed.items())),
            "started_at": self.started_at,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "days_completed": len(self.completed),
                "days_failed": len(self.failed),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        cp = cls()
        cp.completed = set(data.get("completed_dates", []))
        cp.failed = dict(data.get("failed_dates", {}))
        cp.started_at = data.get("started_at", cp.started_at)
        return cp

    # -- mutations ------------------------------------------------------------

    def mark_completed(self, date_str: str) -> None:
        self.completed.add(date_str)
        self.failed.pop(date_str, None)

    def mark_failed(self, date_str: str, error: str) -> None:
        self.failed[date_str] = error

    def remaining(self, all_dates: list[str]) -> list[str]:
        """Return dates from *all_dates* that are not yet completed."""
        return [d for d in all_dates if d not in self.completed]


# -- GCS I/O ------------------------------------------------------------------

def load_checkpoint() -> Checkpoint:
    """Load checkpoint from GCS.  Returns a fresh one if none exists."""
    blob = _bucket().blob(GCS_CHECKPOINT_BLOB)
    if blob.exists():
        data = json.loads(blob.download_as_text())
        cp = Checkpoint.from_dict(data)
        log.info(
            "Checkpoint loaded: %d completed, %d failed.",
            len(cp.completed), len(cp.failed),
        )
        return cp
    log.info("No checkpoint found — starting fresh.")
    return Checkpoint()


def save_checkpoint(cp: Checkpoint) -> None:
    """Persist checkpoint to GCS (overwrites)."""
    blob = _bucket().blob(GCS_CHECKPOINT_BLOB)
    blob.upload_from_string(
        json.dumps(cp.to_dict(), indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    log.debug("Checkpoint saved (%d completed).", len(cp.completed))
