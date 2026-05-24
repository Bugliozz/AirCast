"""Random Matrix Theory diagnostics for Step 3 EDA.

The goal is to complement pairwise Pearson correlations with a spectral view of
the full feature-correlation matrix.  Marchenko-Pastur bounds give a theoretical
noise reference, while the empirical null keeps the analysis more realistic for
station-level time series.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RANDOM_STATE = 42
N_NULL_ITERATIONS = 100
TOP_LOADINGS_PER_COMPONENT = 12
MAX_COMPONENTS_TO_REPORT = 12

DEFAULT_EXCLUDE_COLS = {
    "idstazione",
    "nomestazione",
    "comune",
    "data_giorno",
    "pm10",
    "classe_allerta",
    # Raw calendar ordinals are redundant with cyclic encodings.
    "mese",
    "giorno_settimana",
}

NullMethod = Literal["grouped_circular_shift", "circular_shift", "column_shuffle"]


def marchenko_pastur_bounds(n_samples: int, n_features: int) -> tuple[float, float, float]:
    """Return (q, lambda_minus, lambda_plus) for the correlation matrix."""
    if n_samples <= 1:
        raise ValueError("RMT needs at least two samples.")
    if n_features <= 1:
        raise ValueError("RMT needs at least two features.")

    q = n_features / n_samples
    root_q = np.sqrt(q)
    lambda_minus = (1.0 - root_q) ** 2
    lambda_plus = (1.0 + root_q) ** 2
    return float(q), float(lambda_minus), float(lambda_plus)


def _sort_for_temporal_null(
    df: pd.DataFrame,
    *,
    group_col: str = "idstazione",
    date_col: str = "data_giorno",
) -> pd.DataFrame:
    """Sort by station/date when available so circular shifts respect time order."""
    sort_cols = [c for c in [group_col, date_col] if c in df.columns]
    if not sort_cols:
        return df.copy()
    return df.sort_values(sort_cols).reset_index(drop=True)


def select_rmt_features(
    df: pd.DataFrame,
    *,
    exclude_cols: set[str] | None = None,
    min_unique: int = 2,
) -> pd.DataFrame:
    """Select numeric, non-constant features suitable for RMT diagnostics."""
    excluded = DEFAULT_EXCLUDE_COLS if exclude_cols is None else exclude_cols
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[c for c in excluded if c in numeric.columns], errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    keep_cols: list[str] = []
    dropped: list[str] = []
    for col in numeric.columns:
        non_null = numeric[col].dropna()
        if non_null.nunique() < min_unique:
            dropped.append(col)
            continue
        keep_cols.append(col)

    if dropped:
        log.info("RMT: dropped %d constant/near-constant columns: %s", len(dropped), dropped)

    selected = numeric[keep_cols]
    if selected.shape[1] < 2:
        raise ValueError("RMT feature selection left fewer than two numeric features.")
    return selected


def standardize_features(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Median-impute residual NaNs and z-score features."""
    X = X.copy().replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    valid = stds > 0
    if not valid.all():
        dropped = stds.index[~valid].tolist()
        log.info("RMT: dropped zero-variance columns after imputation: %s", dropped)
        X = X.loc[:, valid]
        means = means.loc[valid]
        stds = stds.loc[valid]

    Z = (X - means) / stds
    if Z.isna().any().any():
        bad_cols = Z.columns[Z.isna().any()].tolist()
        raise ValueError(f"RMT standardization produced NaNs in columns: {bad_cols}")
    return Z, means, stds


def _correlation_eigendecomposition(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues/eigenvectors sorted by descending eigenvalue."""
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def _group_indices(values: np.ndarray) -> list[np.ndarray]:
    """Return row indices for each group value, preserving first-seen order."""
    groups: dict[Any, list[int]] = {}
    for idx, value in enumerate(values):
        groups.setdefault(value, []).append(idx)
    return [np.asarray(rows, dtype=int) for rows in groups.values()]


def _make_null_matrix(
    Z: np.ndarray,
    rng: np.random.Generator,
    *,
    method: NullMethod,
    groups: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Destroy cross-feature alignment while preserving useful marginal structure."""
    n_samples, n_features = Z.shape
    Z_null = np.empty_like(Z)

    if method == "column_shuffle":
        for j in range(n_features):
            Z_null[:, j] = rng.permutation(Z[:, j])
        return Z_null

    if method == "circular_shift":
        for j in range(n_features):
            shift = int(rng.integers(1, n_samples)) if n_samples > 1 else 0
            Z_null[:, j] = np.roll(Z[:, j], shift)
        return Z_null

    if method != "grouped_circular_shift":
        raise ValueError(f"Unknown RMT null method: {method}")

    if not groups:
        return _make_null_matrix(Z, rng, method="circular_shift")

    for rows in groups:
        length = len(rows)
        for j in range(n_features):
            if length <= 1:
                Z_null[rows, j] = Z[rows, j]
            else:
                shift = int(rng.integers(1, length))
                Z_null[rows, j] = np.roll(Z[rows, j], shift)
    return Z_null


def empirical_null_eigenvalues(
    Z: pd.DataFrame,
    *,
    method: NullMethod = "grouped_circular_shift",
    groups: list[np.ndarray] | None = None,
    n_iter: int = N_NULL_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Generate sorted eigenvalue spectra under an empirical null model."""
    rng = np.random.default_rng(random_state)
    values = Z.to_numpy(dtype=float, copy=True)
    spectra: list[np.ndarray] = []

    for _ in range(n_iter):
        null_values = _make_null_matrix(values, rng, method=method, groups=groups)
        eigvals, _ = _correlation_eigendecomposition(null_values)
        spectra.append(eigvals)

    return np.vstack(spectra)


def _build_eigenvalue_table(
    eigvals: np.ndarray,
    *,
    lambda_minus: float,
    lambda_plus: float,
    null_spectra: np.ndarray | None,
) -> pd.DataFrame:
    explained = eigvals / eigvals.sum()
    table = pd.DataFrame({
        "component": np.arange(1, len(eigvals) + 1),
        "eigenvalue": eigvals,
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": np.cumsum(explained),
        "above_mp_bulk": eigvals > lambda_plus,
        "below_mp_bulk": eigvals < lambda_minus,
    })

    if null_spectra is not None and len(null_spectra):
        table["empirical_null_rank_p95"] = np.quantile(null_spectra, 0.95, axis=0)
        table["empirical_null_rank_p99"] = np.quantile(null_spectra, 0.99, axis=0)
        table["above_empirical_p95"] = table["eigenvalue"] > table["empirical_null_rank_p95"]
        table["above_empirical_p99"] = table["eigenvalue"] > table["empirical_null_rank_p99"]

    return table


def _build_loading_table(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    feature_names: list[str],
    eigen_table: pd.DataFrame,
    *,
    top_n: int = TOP_LOADINGS_PER_COMPONENT,
    max_components: int = MAX_COMPONENTS_TO_REPORT,
) -> pd.DataFrame:
    signal_components = eigen_table.loc[eigen_table["above_mp_bulk"], "component"].astype(int).tolist()
    if not signal_components:
        signal_components = eigen_table["component"].head(min(3, len(eigen_table))).astype(int).tolist()

    rows: list[dict[str, Any]] = []
    for component in signal_components[:max_components]:
        idx = component - 1
        loadings = pd.Series(eigvecs[:, idx], index=feature_names)
        top = loadings.abs().sort_values(ascending=False).head(top_n)
        for rank, feature in enumerate(top.index, start=1):
            rows.append({
                "component": component,
                "eigenvalue": float(eigvals[idx]),
                "feature": feature,
                "loading": float(loadings.loc[feature]),
                "abs_loading": float(abs(loadings.loc[feature])),
                "rank": rank,
                "component_above_mp": bool(eigen_table.loc[idx, "above_mp_bulk"]),
            })
    return pd.DataFrame(rows)


def run_rmt_diagnostic(
    df: pd.DataFrame,
    *,
    exclude_cols: set[str] | None = None,
    null_method: NullMethod = "grouped_circular_shift",
    n_null_iter: int = N_NULL_ITERATIONS,
    random_state: int = RANDOM_STATE,
    group_col: str = "idstazione",
    date_col: str = "data_giorno",
) -> dict[str, Any]:
    """Compute RMT diagnostics for a daily feature table."""
    work = _sort_for_temporal_null(df, group_col=group_col, date_col=date_col)
    X = select_rmt_features(work, exclude_cols=exclude_cols)
    Z, means, stds = standardize_features(X)

    n_samples, n_features = Z.shape
    q, lambda_minus, lambda_plus = marchenko_pastur_bounds(n_samples, n_features)
    eigvals, eigvecs = _correlation_eigendecomposition(Z.to_numpy(dtype=float))

    groups = None
    effective_null_method: NullMethod = null_method
    if null_method == "grouped_circular_shift":
        if group_col in work.columns:
            groups = _group_indices(work[group_col].to_numpy())
        else:
            effective_null_method = "circular_shift"

    null_spectra = empirical_null_eigenvalues(
        Z,
        method=effective_null_method,
        groups=groups,
        n_iter=n_null_iter,
        random_state=random_state,
    )
    eigen_table = _build_eigenvalue_table(
        eigvals,
        lambda_minus=lambda_minus,
        lambda_plus=lambda_plus,
        null_spectra=null_spectra,
    )
    loading_table = _build_loading_table(
        eigvals,
        eigvecs,
        Z.columns.tolist(),
        eigen_table,
    )

    above_mp = eigen_table["above_mp_bulk"]
    above_empirical_p95 = eigen_table.get("above_empirical_p95", pd.Series(False, index=eigen_table.index))
    summary = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "q_features_over_samples": float(q),
        "lambda_minus_mp": float(lambda_minus),
        "lambda_plus_mp": float(lambda_plus),
        "n_components_above_mp": int(above_mp.sum()),
        "variance_explained_above_mp": float(eigen_table.loc[above_mp, "explained_variance_ratio"].sum()),
        "n_components_above_empirical_p95": int(above_empirical_p95.sum()),
        "variance_explained_above_empirical_p95": float(
            eigen_table.loc[above_empirical_p95, "explained_variance_ratio"].sum()
        ),
        "empirical_null_method": effective_null_method,
        "n_null_iterations": int(n_null_iter),
        "empirical_lambda_max_p95": float(np.quantile(null_spectra[:, 0], 0.95)),
        "empirical_lambda_max_p99": float(np.quantile(null_spectra[:, 0], 0.99)),
        "random_state": int(random_state),
        "excluded_columns": sorted(DEFAULT_EXCLUDE_COLS if exclude_cols is None else exclude_cols),
        "feature_columns": Z.columns.tolist(),
    }

    log.info(
        "RMT: %d samples, %d features, lambda+ MP=%.4f, components above MP=%d.",
        n_samples,
        n_features,
        lambda_plus,
        summary["n_components_above_mp"],
    )

    return {
        "summary": summary,
        "eigenvalues": eigen_table,
        "loadings": loading_table,
        "feature_means": means,
        "feature_stds": stds,
        "null_spectra": null_spectra,
    }


def save_rmt_artifacts(result: dict[str, Any], out_dir: Path) -> None:
    """Persist RMT summary and tabular artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result["eigenvalues"].to_csv(out_dir / "rmt_eigenvalues.csv", index=False)
    result["loadings"].to_csv(out_dir / "rmt_component_loadings.csv", index=False)
    result["feature_means"].rename("mean").to_csv(out_dir / "rmt_feature_means.csv")
    result["feature_stds"].rename("std").to_csv(out_dir / "rmt_feature_stds.csv")

    with (out_dir / "rmt_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False)

    log.info("RMT artifacts saved to %s", out_dir)


def main() -> None:
    """Run RMT diagnostics directly from an existing parquet file."""
    parser = argparse.ArgumentParser(description="Run RMT spectral diagnostics on the Step 3 dataset.")
    parser.add_argument("--input", default="step_3_eda/daily_dataset_clean.parquet")
    parser.add_argument("--out", default="step_3_eda")
    parser.add_argument("--plots-out", default="step_3_eda/plots")
    parser.add_argument("--n-null", type=int, default=N_NULL_ITERATIONS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    df = pd.read_parquet(args.input)
    result = run_rmt_diagnostic(df, n_null_iter=args.n_null)
    save_rmt_artifacts(result, Path(args.out))

    from step_3_eda.plots import (
        plot_rmt_eigenvalue_spectrum,
        plot_rmt_null_comparison,
        plot_rmt_top_loadings,
    )

    plots_out = Path(args.plots_out)
    plots_out.mkdir(parents=True, exist_ok=True)
    plot_rmt_eigenvalue_spectrum(result["eigenvalues"], result["summary"], plots_out)
    plot_rmt_top_loadings(result["loadings"], plots_out)
    plot_rmt_null_comparison(result["eigenvalues"], result["summary"], plots_out)


if __name__ == "__main__":
    main()
