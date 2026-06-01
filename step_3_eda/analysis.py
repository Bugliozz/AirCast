"""Step 3 — Statistical analysis helpers.

Functions for missing-value reports, class distribution, and descriptive
statistics on the daily analysis DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from step_3_eda.config import MISSING_THRESHOLD

log = logging.getLogger(__name__)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value summary.

    Returns a DataFrame with columns: column, count, missing, missing_pct, dtype.
    """
    total = len(df)
    report = pd.DataFrame({
        "column": df.columns,
        "count": [df[c].notna().sum() for c in df.columns],
        "missing": [df[c].isna().sum() for c in df.columns],
        "missing_pct": [df[c].isna().mean() * 100 for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
    })

    above = report[report["missing_pct"] > MISSING_THRESHOLD * 100]
    if not above.empty:
        log.warning(
            "%d columns exceed %.0f%% missing:\n%s",
            len(above),
            MISSING_THRESHOLD * 100,
            above[["column", "missing_pct"]].to_string(index=False),
        )

    log.info("Missing-value report: %d columns, %d total rows.", len(report), total)
    return report


def station_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station coverage: how many days have PM10 data vs total date range."""
    total_days = df["data_giorno"].nunique()
    coverage = (
        df.groupby(["idstazione", "nomestazione", "provincia"])["data_giorno"]
          .nunique()
          .reset_index()
          .rename(columns={"data_giorno": "days_with_data"})
    )
    coverage["total_days"] = total_days
    coverage["coverage_pct"] = (coverage["days_with_data"] / total_days * 100).round(1)

    sparse = coverage[coverage["coverage_pct"] < (1 - MISSING_THRESHOLD) * 100]
    if not sparse.empty:
        log.warning(
            "%d stations have <%.0f%% coverage:\n%s",
            len(sparse),
            (1 - MISSING_THRESHOLD) * 100,
            sparse[["idstazione", "nomestazione", "coverage_pct"]].to_string(index=False),
        )

    return coverage


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Value counts and percentages for the alert classes."""
    counts = df["classe_allerta"].value_counts().reindex(
        ["verde", "giallo", "arancio", "rosso"], fill_value=0,
    )
    display_labels = {
        "verde": "Green",
        "giallo": "Yellow",
        "arancio": "Orange",
        "rosso": "Red",
    }
    dist = pd.DataFrame({
        "class": [display_labels.get(label, label) for label in counts.index],
        "count": counts.values,
        "pct": (counts.values / counts.sum() * 100).round(1),
    })

    minority = counts.min()
    majority = counts.max()
    ratio = minority / majority if majority > 0 else 0.0
    log.info(
        "Class distribution — balance ratio (min/max): %.3f\n%s",
        ratio,
        dist.to_string(index=False),
    )
    if ratio < 0.10:
        log.warning("Severe class imbalance detected (ratio %.3f). "
                     "SMOTE or class_weight='balanced' strongly recommended.", ratio)

    return dist


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe() with skewness and kurtosis for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    desc = numeric.describe().T
    desc["skew"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()
    return desc


def top_correlations(df: pd.DataFrame, target: str = "pm10", n: int = 10) -> pd.Series:
    """Return the top-n features most correlated with `target` (absolute Pearson)."""
    numeric = df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        log.warning("Target '%s' not in numeric columns.", target)
        return pd.Series(dtype=float)

    corr = numeric.corr()[target].drop(target, errors="ignore").abs().sort_values(ascending=False)
    return corr.head(n)
