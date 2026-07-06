"""Tests for temporal_train_test_split in shared/utils.py."""

import pandas as pd
import pytest

from shared.utils import temporal_train_test_split


def _make_fake_dataset(n_days: int = 20, stations_per_day: int = 3) -> pd.DataFrame:
    """Create a minimal fake station-day DataFrame.

    ``day_index`` is a deterministic per-day proxy (0..n_days-1) so that,
    after the split drops ``data_giorno`` from the feature matrix, tests can
    still verify day-level separation via this surviving column.
    """
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for day_idx, d in enumerate(dates):
        for s in range(stations_per_day):
            rows.append(
                {
                    "data_giorno": d,
                    "idstazione": s,
                    "day_index": day_idx,
                    "pm10": float(day_idx * 10 + s),
                }
            )
    return pd.DataFrame(rows)


class TestTemporalTrainTestSplit:
    """Validate day-based temporal train/test splitting."""

    def test_no_day_in_both_train_and_test(self) -> None:
        """No calendar day may appear in both the train and test partitions."""
        df = _make_fake_dataset(n_days=20, stations_per_day=4)
        X_train, _, X_test, _, _ = temporal_train_test_split(
            df, target_col="pm10", train_ratio=0.7
        )

        train_days = set(X_train["day_index"].unique())
        test_days = set(X_test["day_index"].unique())
        assert train_days.isdisjoint(test_days)
        assert len(train_days) > 0
        assert len(test_days) > 0

    def test_train_precedes_test_temporally(self) -> None:
        """The last train day must be strictly before the first test day."""
        df = _make_fake_dataset(n_days=20, stations_per_day=3)
        X_train, _, X_test, _, train_dates = temporal_train_test_split(
            df, target_col="pm10", train_ratio=0.7
        )

        assert train_dates.max() < df["data_giorno"].max()
        assert X_train["day_index"].max() < X_test["day_index"].min()

    def test_row_counts_add_up(self) -> None:
        df = _make_fake_dataset(n_days=20, stations_per_day=3)
        X_train, y_train, X_test, y_test, _ = temporal_train_test_split(
            df, target_col="pm10", train_ratio=0.7
        )
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(df)

    def test_drops_identifier_and_date_columns(self) -> None:
        df = _make_fake_dataset(n_days=20, stations_per_day=2)
        X_train, _, X_test, _, _ = temporal_train_test_split(
            df, target_col="pm10", train_ratio=0.7, drop_cols=["idstazione"]
        )
        assert "data_giorno" not in X_train.columns
        assert "pm10" not in X_train.columns
        assert "idstazione" not in X_train.columns
        assert "data_giorno" not in X_test.columns

    def test_raises_on_missing_date_col(self) -> None:
        df = pd.DataFrame({"pm10": [1, 2, 3]})
        with pytest.raises(ValueError, match="Date column"):
            temporal_train_test_split(df, target_col="pm10")

    def test_raises_on_missing_target_col(self) -> None:
        df = pd.DataFrame({"data_giorno": pd.date_range("2025-01-01", periods=3)})
        with pytest.raises(ValueError, match="Target column"):
            temporal_train_test_split(df, target_col="pm10")
