"""Tests for impute_missing in shared/utils.py."""

import numpy as np
import pandas as pd

from shared.utils import impute_missing


def _make_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test frames with a weather column (imputable by naming convention)."""
    X_train = pd.DataFrame({"temp_mean": [10.0, 20.0, np.nan, 40.0, 50.0]})
    # Test medians would be wildly different if computed from the test set itself.
    X_test = pd.DataFrame({"temp_mean": [1000.0, np.nan, 3000.0]})
    return X_train, X_test


class TestImputeMissing:
    """Validate leakage-free median imputation."""

    def test_fit_mode_computes_medians_from_train(self) -> None:
        X_train, _ = _make_train_test()
        _, train_medians = impute_missing(X_train)
        assert train_medians["temp_mean"] == X_train["temp_mean"].median()

    def test_test_medians_are_not_recomputed_from_test_data(self) -> None:
        """The medians applied to the test set must come from train, not test."""
        X_train, X_test = _make_train_test()
        _, train_medians = impute_missing(X_train)
        X_test_imputed, medians_used = impute_missing(X_test, medians=train_medians)

        # The value used to fill the NaN in X_test must equal the train median,
        # not the (very different) median of X_test itself.
        filled_value = X_test_imputed.loc[X_test["temp_mean"].isna(), "temp_mean"].iloc[0]
        assert filled_value == train_medians["temp_mean"]
        assert filled_value != X_test["temp_mean"].median()
        assert medians_used == train_medians

    def test_no_nan_remains_in_imputable_columns_after_test_imputation(self) -> None:
        X_train, X_test = _make_train_test()
        _, train_medians = impute_missing(X_train)
        X_test_imputed, _ = impute_missing(X_test, medians=train_medians)
        assert not X_test_imputed["temp_mean"].isna().any()

    def test_does_not_mutate_input_dataframe(self) -> None:
        X_train, _ = _make_train_test()
        original = X_train.copy()
        impute_missing(X_train)
        pd.testing.assert_frame_equal(X_train, original)

    def test_train_dataframe_is_unaffected_by_train_ratio_of_missingness(self) -> None:
        """Columns without any NaN are left untouched and not added to medians."""
        X_train = pd.DataFrame({"temp_mean": [10.0, 20.0, 30.0]})
        _, medians = impute_missing(X_train)
        assert "temp_mean" not in medians
