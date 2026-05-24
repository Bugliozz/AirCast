"""Tests for Step 5 anti-leakage guard rails."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from step_5_classification.guardrails import (
    assert_cv_guardrails,
    assert_manifest_guardrails,
    assert_randomized_search_seed,
    assert_station_intersection,
    assert_temporal_holdout,
    build_split_metadata,
)


def test_temporal_holdout_requires_train_before_test() -> None:
    train_dates = pd.Series(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    test_dates = pd.Series(pd.to_datetime(["2025-01-03", "2025-01-04"]))

    with pytest.raises(AssertionError, match="Temporal leakage"):
        assert_temporal_holdout(train_dates, test_dates)


def test_station_intersection_must_match() -> None:
    with pytest.raises(AssertionError, match="Station intersection incoherent"):
        assert_station_intersection(["A", "B"], ["A", "C"])


def test_split_metadata_contains_required_ranges_and_counts() -> None:
    metadata = build_split_metadata(
        train_dates=pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02"])),
        test_dates=pd.Series(pd.to_datetime(["2025-01-03"])),
        train_stations=pd.Series(["A", "B"]),
        test_stations=pd.Series(["A", "B"]),
        train_indices=pd.Series([0, 1]),
        test_indices=pd.Series([2]),
    )

    assert metadata["train_date_range"] == [
        "2025-01-01 00:00:00",
        "2025-01-02 00:00:00",
    ]
    assert metadata["test_date_range"] == [
        "2025-01-03 00:00:00",
        "2025-01-03 00:00:00",
    ]
    assert metadata["n_stations_train"] == 2
    assert metadata["n_stations_test"] == 2
    assert metadata["n_stations_intersection"] == 2
    assert metadata["split_index_hash"]


def test_cv_guardrails_require_five_folds() -> None:
    cv_splits = [
        (np.array([0, 1]), np.array([2])),
        (np.array([0, 1, 2]), np.array([3])),
    ]

    with pytest.raises(AssertionError, match="must be 5"):
        assert_cv_guardrails(cv_splits, expected_n_splits=2)


def test_cv_guardrails_reject_overlap() -> None:
    cv_splits = [
        (np.array([0]), np.array([1])),
        (np.array([0, 1]), np.array([2])),
        (np.array([0, 1, 2]), np.array([3])),
        (np.array([0, 1, 2, 3]), np.array([4])),
        (np.array([0, 1, 2, 3, 4]), np.array([4, 5])),
    ]

    with pytest.raises(AssertionError, match="overlapping"):
        assert_cv_guardrails(cv_splits, expected_n_splits=5)


def test_randomized_search_requires_shared_seed() -> None:
    search = RandomizedSearchCV(
        estimator=LogisticRegression(),
        param_distributions={"C": [0.1, 1.0]},
        n_iter=1,
        random_state=None,
    )

    with pytest.raises(AssertionError, match="random_state=42"):
        assert_randomized_search_seed(
            search,
            expected_random_state=42,
            search_name="example",
        )


def test_manifest_guardrails_reject_mixed_split_hashes(tmp_path) -> None:
    manifest = {
        "model_artifacts": {
            "logistic_regression": {
                "split_index_hash": "split-a",
                "cv_signature_hash": "cv-a",
                "n_cv_splits": 5,
                "random_state": 42,
            },
            "random_forest": {
                "split_index_hash": "split-a",
                "cv_signature_hash": "cv-a",
                "n_cv_splits": 5,
                "random_state": 42,
            },
            "xgboost": {
                "split_index_hash": "split-b",
                "cv_signature_hash": "cv-a",
                "n_cv_splits": 5,
                "random_state": 42,
            },
        }
    }

    with pytest.raises(AssertionError, match="same split"):
        assert_manifest_guardrails(
            manifest,
            artifacts_dir=tmp_path,
            model_names=("logistic_regression", "random_forest", "xgboost"),
            expected_n_splits=5,
            expected_random_state=42,
            verify_files=False,
        )
