"""Hard anti-leakage checks for Step 5 classification artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain JSON-compatible values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, failing loudly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required guard-rail artifact not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def hash_sequence(values: Iterable[Any]) -> str:
    """Hash an ordered sequence without materializing it as one giant string."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(_jsonable(value)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest for an artifact file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_temporal_holdout(
    train_dates: Sequence[Any],
    test_dates: Sequence[Any],
) -> Tuple[pd.Series, pd.Series]:
    """Assert that every train day is strictly before every test day."""
    train_dt = pd.to_datetime(pd.Series(train_dates), errors="raise")
    test_dt = pd.to_datetime(pd.Series(test_dates), errors="raise")

    if train_dt.empty or test_dt.empty:
        raise AssertionError("Temporal split produced an empty train or test set.")

    max_train_date = train_dt.max()
    min_test_date = test_dt.min()
    assert max_train_date < min_test_date, (
        "Temporal leakage detected: "
        f"max(data_giorno_train)={max_train_date} >= "
        f"min(data_giorno_test)={min_test_date}"
    )

    return train_dt, test_dt


def assert_station_intersection(
    train_stations: Sequence[Any],
    test_stations: Sequence[Any],
) -> Tuple[set[str], set[str], set[str]]:
    """Assert train and test contain the same station set."""
    train_set = set(pd.Series(train_stations).dropna().astype(str).tolist())
    test_set = set(pd.Series(test_stations).dropna().astype(str).tolist())
    intersection = train_set & test_set

    if not train_set or not test_set:
        raise AssertionError("Station guard-rail cannot run on an empty station set.")
    if intersection != train_set or intersection != test_set:
        missing_in_test = sorted(train_set - test_set)
        missing_in_train = sorted(test_set - train_set)
        raise AssertionError(
            "Station intersection incoherent: "
            f"n_train={len(train_set)}, n_test={len(test_set)}, "
            f"n_intersection={len(intersection)}, "
            f"missing_in_test={missing_in_test[:10]}, "
            f"missing_in_train={missing_in_train[:10]}"
        )

    return train_set, test_set, intersection


def build_split_metadata(
    *,
    train_dates: Sequence[Any],
    test_dates: Sequence[Any],
    train_stations: Sequence[Any],
    test_stations: Sequence[Any],
    train_indices: Sequence[Any],
    test_indices: Sequence[Any],
) -> Dict[str, Any]:
    """Build and validate split metadata shared by train/evaluate scripts."""
    train_dt, test_dt = assert_temporal_holdout(train_dates, test_dates)
    train_set, test_set, intersection = assert_station_intersection(
        train_stations, test_stations
    )

    train_indices_list = [int(i) for i in train_indices]
    test_indices_list = [int(i) for i in test_indices]
    split_index_values: List[Any] = (
        ["train"]
        + train_indices_list
        + ["test"]
        + test_indices_list
    )

    return {
        "train_date_range": [str(train_dt.min()), str(train_dt.max())],
        "test_date_range": [str(test_dt.min()), str(test_dt.max())],
        "n_train_samples": int(len(train_dt)),
        "n_test_samples": int(len(test_dt)),
        "n_stations_train": int(len(train_set)),
        "n_stations_test": int(len(test_set)),
        "n_stations_intersection": int(len(intersection)),
        "train_index_hash": hash_sequence(train_indices_list),
        "test_index_hash": hash_sequence(test_indices_list),
        "split_index_hash": hash_sequence(split_index_values),
    }


def hash_cv_splits(cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]]) -> str:
    """Hash train/validation row indices for the complete CV splitter."""
    digest = hashlib.sha256()
    for train_idx, val_idx in cv_splits:
        train_arr = np.asarray(train_idx, dtype=np.int64)
        val_arr = np.asarray(val_idx, dtype=np.int64)
        digest.update(train_arr.tobytes())
        digest.update(b"|")
        digest.update(val_arr.tobytes())
        digest.update(b";")
    return digest.hexdigest()


def assert_cv_guardrails(
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    expected_n_splits: int,
) -> Dict[str, Any]:
    """Assert that CV is the fixed 5-fold temporal split used by all models."""
    assert expected_n_splits == 5, (
        f"N_CV_SPLITS must be 5 for comparable benchmarks, got {expected_n_splits}."
    )
    assert len(cv_splits) == expected_n_splits, (
        f"Expected {expected_n_splits} CV folds, got {len(cv_splits)}."
    )

    fold_sizes: List[Dict[str, int]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        train_arr = np.asarray(train_idx, dtype=np.int64)
        val_arr = np.asarray(val_idx, dtype=np.int64)
        if train_arr.size == 0 or val_arr.size == 0:
            raise AssertionError(f"CV fold {fold_idx} has an empty train or val set.")
        if np.intersect1d(train_arr, val_arr).size:
            raise AssertionError(f"CV fold {fold_idx} has overlapping train/val rows.")
        assert int(train_arr.max()) < int(val_arr.min()), (
            f"CV fold {fold_idx} is not temporal: "
            f"max(train_idx)={int(train_arr.max())} >= "
            f"min(val_idx)={int(val_arr.min())}"
        )
        fold_sizes.append(
            {
                "fold": fold_idx,
                "n_train_rows": int(train_arr.size),
                "n_val_rows": int(val_arr.size),
            }
        )

    return {
        "n_splits": int(len(cv_splits)),
        "cv_signature_hash": hash_cv_splits(cv_splits),
        "fold_sizes": fold_sizes,
    }


def assert_randomized_search_seed(
    search: Any,
    *,
    expected_random_state: int,
    search_name: str,
) -> None:
    """Assert that a RandomizedSearchCV object uses the shared seed."""
    actual = getattr(search, "random_state", None)
    if actual != expected_random_state:
        raise AssertionError(
            f"{search_name} RandomizedSearchCV must set "
            f"random_state={expected_random_state}, got {actual!r}."
        )


def build_model_manifest_entry(
    *,
    artifact_path: Path,
    split_metadata: Mapping[str, Any],
    cv_metadata: Mapping[str, Any],
    random_state: int,
) -> Dict[str, Any]:
    """Build a manifest entry tying one model file to split/CV/seed metadata."""
    return {
        "path": artifact_path.as_posix(),
        "sha256": file_sha256(artifact_path),
        "split_index_hash": split_metadata["split_index_hash"],
        "cv_signature_hash": cv_metadata["cv_signature_hash"],
        "n_cv_splits": int(cv_metadata["n_splits"]),
        "random_state": int(random_state),
    }


def assert_same_split_metadata(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    """Assert two split metadata payloads identify the exact same split."""
    required = ("train_index_hash", "test_index_hash", "split_index_hash")
    for key in required:
        if key not in left or key not in right:
            raise AssertionError(f"Split metadata missing required key: {key}")
        if left[key] != right[key]:
            raise AssertionError(
                f"Split metadata mismatch for {key}: {left[key]} != {right[key]}"
            )


def _resolve_manifest_path(path_value: str, artifacts_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return artifacts_dir / path.name


def assert_manifest_guardrails(
    manifest: Mapping[str, Any],
    *,
    artifacts_dir: Path,
    model_names: Sequence[str],
    expected_n_splits: int,
    expected_random_state: int,
    verify_files: bool,
) -> None:
    """Assert all benchmark models share split, CV signature and seed."""
    if expected_n_splits != 5:
        raise AssertionError(
            f"N_CV_SPLITS must be 5 for comparable benchmarks, got {expected_n_splits}."
        )

    model_artifacts = manifest.get("model_artifacts")
    if not isinstance(model_artifacts, Mapping):
        raise AssertionError("Training manifest missing model_artifacts.")

    split_hashes = set()
    cv_hashes = set()

    for model_name in model_names:
        entry = model_artifacts.get(model_name)
        if not isinstance(entry, Mapping):
            raise AssertionError(f"Training manifest missing entry for {model_name}.")

        split_hash = entry.get("split_index_hash")
        cv_hash = entry.get("cv_signature_hash")
        n_cv_splits = entry.get("n_cv_splits")
        random_state = entry.get("random_state")

        if not split_hash:
            raise AssertionError(f"{model_name} manifest entry has no split hash.")
        if not cv_hash:
            raise AssertionError(f"{model_name} manifest entry has no CV hash.")
        if n_cv_splits != expected_n_splits:
            raise AssertionError(
                f"{model_name} uses {n_cv_splits} CV splits, "
                f"expected {expected_n_splits}."
            )
        if random_state != expected_random_state:
            raise AssertionError(
                f"{model_name} random_state={random_state}, "
                f"expected {expected_random_state}."
            )

        split_hashes.add(split_hash)
        cv_hashes.add(cv_hash)

        if verify_files:
            artifact_path = _resolve_manifest_path(str(entry.get("path")), artifacts_dir)
            if not artifact_path.exists():
                raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
            actual_sha = file_sha256(artifact_path)
            if actual_sha != entry.get("sha256"):
                raise AssertionError(
                    f"{model_name} artifact hash mismatch. "
                    "The model file does not match training_manifest.json."
                )

    if len(split_hashes) != 1:
        raise AssertionError(
            f"Benchmark models do not share the same split: {sorted(split_hashes)}"
        )
    if len(cv_hashes) != 1:
        raise AssertionError(
            f"Benchmark models do not share the same CV splits: {sorted(cv_hashes)}"
        )
