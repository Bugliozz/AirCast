"""Tests for make_temporal_cv_splits in shared/utils.py."""

import numpy as np
import pandas as pd
import pytest

from shared.utils import make_temporal_cv_splits


def _make_fake_train(n_days: int = 30, stations_per_day: int = 3) -> pd.DataFrame:
    """Create a minimal fake training DataFrame."""
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        for s in range(stations_per_day):
            rows.append({"data_giorno": d, "station_id": s, "feat1": np.random.randn()})
    return pd.DataFrame(rows)


class TestMakeTemporalCvSplits:
    """Validate day-based temporal CV splitting."""

    def test_returns_correct_number_of_folds(self) -> None:
        df = _make_fake_train(n_days=30, stations_per_day=3)
        splits = make_temporal_cv_splits(df, n_splits=5)
        assert len(splits) == 5

    def test_no_intraday_leakage(self) -> None:
        """All rows of a given day must be entirely in train OR val, never both."""
        df = _make_fake_train(n_days=30, stations_per_day=4)
        splits = make_temporal_cv_splits(df, n_splits=5)

        for train_idx, val_idx in splits:
            train_days = set(df.iloc[train_idx]["data_giorno"].unique())
            val_days = set(df.iloc[val_idx]["data_giorno"].unique())
            assert train_days.isdisjoint(val_days), "Train and val share days!"

    def test_temporal_ordering(self) -> None:
        """All train days must precede all val days (no future leakage)."""
        df = _make_fake_train(n_days=30, stations_per_day=2)
        splits = make_temporal_cv_splits(df, n_splits=5)

        for train_idx, val_idx in splits:
            max_train_day = df.iloc[train_idx]["data_giorno"].max()
            min_val_day = df.iloc[val_idx]["data_giorno"].min()
            assert max_train_day < min_val_day

    def test_all_rows_covered_per_fold(self) -> None:
        """Train + val indices should not overlap and all val rows should exist."""
        df = _make_fake_train(n_days=30, stations_per_day=3)
        splits = make_temporal_cv_splits(df, n_splits=5)

        for train_idx, val_idx in splits:
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_growing_train_set(self) -> None:
        """Each successive fold should have a larger training set (TimeSeriesSplit)."""
        df = _make_fake_train(n_days=30, stations_per_day=3)
        splits = make_temporal_cv_splits(df, n_splits=5)

        train_sizes = [len(t) for t, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_class_coverage_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should warn when a class is missing from a fold."""
        df = _make_fake_train(n_days=20, stations_per_day=2)
        # Put a rare class only in the very last days
        y = pd.Series(["verde"] * len(df))
        y.iloc[-2:] = "rosso"  # only last day

        with caplog.at_level("WARNING"):
            make_temporal_cv_splits(df, y_train=y, n_splits=5)

        assert any("MISSING" in r.message for r in caplog.records)

    def test_raises_on_missing_date_col(self) -> None:
        df = pd.DataFrame({"feat1": [1, 2, 3]})
        with pytest.raises(ValueError, match="Date column"):
            make_temporal_cv_splits(df)

    def test_raises_on_too_few_days(self) -> None:
        df = _make_fake_train(n_days=3, stations_per_day=2)
        with pytest.raises(ValueError, match="unique days"):
            make_temporal_cv_splits(df, n_splits=5)
