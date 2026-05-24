"""Runtime configuration for the FastAPI service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(raw_value: str | None, default: Path) -> Path:
    if raw_value is None or raw_value.strip() == "":
        return default
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _first_existing(candidates: Iterable[Path]) -> Path:
    paths = list(candidates)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


GCS_BUCKET = os.environ.get("GCS_BUCKET", "exam-project-backfill")
ARTIFACTS_DIR = _resolve_path(os.environ.get("ARTIFACTS_DIR"), ROOT_DIR / "artifacts")

PARQUET_PATH = _resolve_path(
    os.environ.get("PARQUET_PATH"),
    _first_existing(
        [
            ARTIFACTS_DIR / "daily_dataset_clean.parquet",
            ARTIFACTS_DIR / "step_3_eda" / "daily_dataset_clean.parquet",
            ROOT_DIR / "step_3_eda" / "daily_dataset_clean.parquet",
        ]
    ),
)

REGRESSION_MODEL_PATH = _resolve_path(
    os.environ.get("REGRESSION_MODEL_PATH"),
    _first_existing(
        [
            ARTIFACTS_DIR / "step_4_regression" / "artifacts" / "best_model.joblib",
            ARTIFACTS_DIR / "regression_best_model.joblib",
            ARTIFACTS_DIR / "best_model_regression.joblib",
            ROOT_DIR / "step_4_regression" / "artifacts" / "best_model.joblib",
        ]
    ),
)

CALIBRATED_CLASSIFIER_PATH = _resolve_path(
    os.environ.get("CALIBRATED_CLASSIFIER_PATH"),
    _first_existing(
        [
            ARTIFACTS_DIR
            / "step_5_classification"
            / "artifacts"
            / "xgboost_hybrid_calibrated.joblib",
            ARTIFACTS_DIR / "xgboost_hybrid_calibrated.joblib",
            ARTIFACTS_DIR / "best_model_calibrated.joblib",
            ROOT_DIR
            / "step_5_classification"
            / "artifacts"
            / "xgboost_hybrid_calibrated.joblib",
        ]
    ),
)

HYBRID_PARAMS_PATH = _resolve_path(
    os.environ.get("HYBRID_PARAMS_PATH"),
    _first_existing(
        [
            ARTIFACTS_DIR / "step_5_classification" / "artifacts" / "hybrid_params.json",
            ARTIFACTS_DIR / "hybrid_params.json",
            ROOT_DIR / "step_5_classification" / "artifacts" / "hybrid_params.json",
        ]
    ),
)

SENSORS_REGISTRY_PATH = _resolve_path(
    os.environ.get("SENSORS_REGISTRY_PATH"),
    _first_existing(
        [
            ARTIFACTS_DIR / "sensors_registry.json",
            ROOT_DIR / "data" / "raw" / "sensors_registry.json",
        ]
    ),
)

CLUSTERS_CSV_PATH = _resolve_path(
    os.environ.get("CLUSTERS_CSV_PATH"),
    _first_existing(
        [
            ARTIFACTS_DIR / "station_clusters.csv",
            ROOT_DIR / "step_6_clustering" / "artifacts" / "station_clusters.csv",
        ]
    ),
)

CLUSTER_INTERPRETATION_PATH = _resolve_path(
    os.environ.get("CLUSTER_INTERPRETATION_PATH"),
    _first_existing(
        [
            ARTIFACTS_DIR / "cluster_interpretation.json",
            ROOT_DIR / "step_6_clustering" / "artifacts" / "cluster_interpretation.json",
        ]
    ),
)
