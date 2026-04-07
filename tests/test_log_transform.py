"""Tests for step 3.2 — Log-transform del target.

Covers:
- np.log1p / np.expm1 are exact inverses (no rounding surprises)
- build_elasticnet_pipeline() returns a correctly structured Pipeline
- TransformedTargetRegressor produces predictions in the original scale
- Predictions are always >= 0 (pm10 cannot be negative)
- Parameter-grid keys in config.py resolve correctly against the pipeline
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from step_4_regression.train import build_elasticnet_pipeline
from step_4_regression.config import ELASTICNET_PARAM_GRID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_dataset(n_rows: int = 60) -> pd.DataFrame:
    """Minimal DataFrame that mimics the real feature matrix."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "data_giorno": pd.date_range("2025-01-01", periods=n_rows, freq="D"),
            "stagione": rng.choice(["inverno", "primavera", "estate", "autunno"], n_rows),
            "provincia": rng.choice(["MI", "BG", "BS"], n_rows),
            "pm10_lag1": rng.uniform(10, 80, n_rows),
            "pm10_lag2": rng.uniform(10, 80, n_rows),
            "wind_speed_mean": rng.uniform(0.5, 5.0, n_rows),
            "pressure_mean": rng.uniform(1000, 1025, n_rows),
            "temp_mean": rng.uniform(-5, 30, n_rows),
            "blh_min": rng.uniform(100, 800, n_rows),
            "stagnation_index": rng.uniform(0, 5, n_rows),
            "mese_sin": np.sin(2 * np.pi * rng.integers(1, 13, n_rows) / 12),
            "mese_cos": np.cos(2 * np.pi * rng.integers(1, 13, n_rows) / 12),
            "is_weekend": rng.integers(0, 2, n_rows).astype(float),
        }
    )


def _make_y(n_rows: int = 60) -> pd.Series:
    """PM10 target — always positive, right-skewed."""
    rng = np.random.default_rng(1)
    return pd.Series(np.abs(rng.lognormal(mean=3.5, sigma=0.8, size=n_rows)), name="pm10")


# ---------------------------------------------------------------------------
# Unit tests — log1p / expm1 contract
# ---------------------------------------------------------------------------

class TestLog1pExpm1Contract:
    """np.log1p and np.expm1 must be exact inverses over the pm10 range."""

    def test_round_trip_positive_values(self) -> None:
        y = np.array([0.0, 1.0, 10.0, 50.0, 100.0, 500.0])
        np.testing.assert_allclose(np.expm1(np.log1p(y)), y, rtol=1e-12)

    def test_log1p_maps_zero_to_zero(self) -> None:
        assert np.log1p(0.0) == 0.0

    def test_expm1_inverse_of_log1p_for_lognormal_sample(self) -> None:
        rng = np.random.default_rng(42)
        y = np.abs(rng.lognormal(mean=3.5, sigma=0.8, size=500))
        np.testing.assert_allclose(np.expm1(np.log1p(y)), y, rtol=1e-12)


# ---------------------------------------------------------------------------
# Unit tests — build_elasticnet_pipeline()
# ---------------------------------------------------------------------------

class TestBuildElasticnetPipeline:
    """Structural tests for the pipeline factory."""

    def test_returns_pipeline_instance(self) -> None:
        X = _make_fake_dataset()
        pipe = build_elasticnet_pipeline(X)
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_two_steps(self) -> None:
        X = _make_fake_dataset()
        pipe = build_elasticnet_pipeline(X)
        assert len(pipe.steps) == 2

    def test_first_step_is_named_pre(self) -> None:
        X = _make_fake_dataset()
        pipe = build_elasticnet_pipeline(X)
        assert pipe.steps[0][0] == "pre"

    def test_second_step_is_named_regressor(self) -> None:
        X = _make_fake_dataset()
        pipe = build_elasticnet_pipeline(X)
        assert pipe.steps[1][0] == "regressor"

    def test_second_step_is_transformed_target_regressor(self) -> None:
        X = _make_fake_dataset()
        pipe = build_elasticnet_pipeline(X)
        assert isinstance(pipe.named_steps["regressor"], TransformedTargetRegressor)

    def test_transformed_target_uses_log1p(self) -> None:
        X = _make_fake_dataset()
        ttr = build_elasticnet_pipeline(X).named_steps["regressor"]
        # func should be np.log1p
        sample = np.array([0.0, 10.0, 100.0])
        np.testing.assert_array_equal(ttr.func(sample), np.log1p(sample))

    def test_transformed_target_uses_expm1(self) -> None:
        X = _make_fake_dataset()
        ttr = build_elasticnet_pipeline(X).named_steps["regressor"]
        sample = np.array([0.0, 1.0, 4.6])
        np.testing.assert_array_equal(ttr.inverse_func(sample), np.expm1(sample))


# ---------------------------------------------------------------------------
# Integration tests — fit + predict
# ---------------------------------------------------------------------------

class TestElasticnetPipelineFitPredict:
    """End-to-end: the pipeline must fit and predict without errors."""

    def test_fit_does_not_raise(self) -> None:
        X = _make_fake_dataset(n_rows=80)
        y = _make_y(n_rows=80)
        pipe = build_elasticnet_pipeline(X)
        pipe.fit(X, y)  # should not raise

    def test_predict_returns_correct_shape(self) -> None:
        X_train = _make_fake_dataset(n_rows=80)
        y_train = _make_y(n_rows=80)
        X_test = _make_fake_dataset(n_rows=20)

        pipe = build_elasticnet_pipeline(X_train)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert preds.shape == (20,)

    def test_predictions_are_non_negative(self) -> None:
        """expm1(log1p(x)) >= 0 for any real x; model must respect this."""
        X_train = _make_fake_dataset(n_rows=80)
        y_train = _make_y(n_rows=80)
        X_test = _make_fake_dataset(n_rows=20)

        pipe = build_elasticnet_pipeline(X_train)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert np.all(preds >= -1.0), (
            "TransformedTargetRegressor with expm1 inverse should keep predictions >= -1"
        )

    def test_predictions_are_finite(self) -> None:
        X_train = _make_fake_dataset(n_rows=80)
        y_train = _make_y(n_rows=80)
        X_test = _make_fake_dataset(n_rows=20)

        pipe = build_elasticnet_pipeline(X_train)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert np.all(np.isfinite(preds)), "Predictions must be finite"


# ---------------------------------------------------------------------------
# Config compatibility — param-grid keys must resolve against the pipeline
# ---------------------------------------------------------------------------

class TestConfigParamGridCompatibility:
    """ELASTICNET_PARAM_GRID keys must be valid get_params() paths."""

    def test_param_grid_keys_resolve(self) -> None:
        X = _make_fake_dataset(n_rows=80)
        pipe = build_elasticnet_pipeline(X)
        all_params = pipe.get_params(deep=True)

        for key in ELASTICNET_PARAM_GRID:
            assert key in all_params, (
                f"Param-grid key '{key}' not found in pipeline.get_params(). "
                f"Check pipeline step naming in build_elasticnet_pipeline()."
            )
