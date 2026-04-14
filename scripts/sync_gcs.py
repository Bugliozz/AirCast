"""Download the rolling refresh window from GCS into ``data/recent/``.

Companion to ``step_1_collection/gcloud_daily``: the Cloud Run Job keeps
the bucket fresh, this script pulls the recent JSONs locally so the
regular ingest pipeline can populate MySQL / the parquet feature store.

The download target is intentionally **not** ``data/raw/`` — that
directory is reserved for the immutable historical archive used as the
training source.  Keeping the rolling window in a separate folder
prevents the inference-only data from polluting the training set with
gaps caused by sporadic syncs.

Behaviour:
  - Targets the window ``[today - END_OFFSET - WINDOW + 1 .. today - END_OFFSET]``
    (same defaults as the daily job: 7 days, ending yesterday).
  - Downloads ``{date}_measurements.json`` and ``{date}_weather.json`` for
    every target date — overwriting any local copy so stale files are
    refreshed.
  - Skips dates that are not (yet) present on GCS and logs them.

Env vars (all optional):
  GCS_BUCKET       default: exam-project-backfill
  WINDOW_DAYS      default: 7
  END_OFFSET_DAYS  default: 1

CLI flags:
  --into-raw       download into ``data/raw/`` instead of ``data/recent/``
                   (use this to extend the training archive with a gap-fill
                   after running a historical backfill on GCloud).
  --start YYYY-MM-DD / --end YYYY-MM-DD
                   override the rolling window with an explicit date range.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

from google.cloud import storage as gcs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sync_gcs")

GCS_BUCKET = os.environ.get("GCS_BUCKET", "exam-project-backfill")
GCS_DATA_PREFIX = "data/raw"
WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", "7"))
END_OFFSET_DAYS = int(os.environ.get("END_OFFSET_DAYS", "1"))

ROOT = Path(__file__).resolve().parent.parent
LOCAL_RECENT_DIR = ROOT / "data" / "recent"
LOCAL_RAW_DIR = ROOT / "data" / "raw"


def target_window(start: date | None = None, end: date | None = None) -> list[str]:
    if start is None or end is None:
        end = date.today() - timedelta(days=END_OFFSET_DAYS)
        start = end - timedelta(days=WINDOW_DAYS - 1)
    return [
        (start + timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync GCS JSONs into local data dirs.")
    p.add_argument(
        "--into-raw", action="store_true",
        help="Download into data/raw/ (training archive) instead of data/recent/.",
    )
    p.add_argument("--start", help="Override window start (YYYY-MM-DD).")
    p.add_argument("--end", help="Override window end (YYYY-MM-DD, inclusive).")
    return p.parse_args()


def download_if_present(bucket: gcs.Bucket, blob_name: str, dest: Path) -> bool:
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(dest.as_posix())
    return True


def main() -> int:
    args = _parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    window = target_window(start, end)

    dest_dir = LOCAL_RAW_DIR if args.into_raw else LOCAL_RECENT_DIR
    log.info(
        "Sync window: %s -> %s (%d days) from gs://%s/%s into %s",
        window[0], window[-1], len(window), GCS_BUCKET, GCS_DATA_PREFIX,
        dest_dir.relative_to(ROOT),
    )

    bucket = gcs.Client().bucket(GCS_BUCKET)
    pulled = missing = 0

    for d in window:
        for suffix in ("measurements", "weather"):
            blob_name = f"{GCS_DATA_PREFIX}/{d}_{suffix}.json"
            dest = dest_dir / f"{d}_{suffix}.json"
            if download_if_present(bucket, blob_name, dest):
                pulled += 1
                log.info("  + %s", dest.name)
            else:
                missing += 1
                log.warning("  - %s not on GCS yet", blob_name)

    log.info("Done: %d files pulled, %d still missing.", pulled, missing)
    return 0 if missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
