"""Step 3 — Plot generation functions.

Each function takes a DataFrame, produces a figure, and saves it to the
output directory.  All plots use a consistent seaborn whitegrid theme.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from step_3_eda.config import (
    ALERT_COLORS,
    FIGSIZE_SQUARE,
    FIGSIZE_WIDE,
    PLOT_DPI,
    PM10_LABELS,
    PM10_THRESHOLDS,
    SEASON_ORDER,
)

log = logging.getLogger(__name__)

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.dpi": PLOT_DPI, "savefig.dpi": PLOT_DPI})

_WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_ALERT_LABELS = {
    "verde": "Green",
    "giallo": "Yellow",
    "arancio": "Orange",
    "rosso": "Red",
}
_SEASON_LABELS = {
    "inverno": "Winter",
    "primavera": "Spring",
    "estate": "Summer",
    "autunno": "Autumn",
}


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot: %s", path.name)


# ──────────────────────────────────────────────────────────────────────────────
# 01 — Missing-values heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame, out: Path) -> None:
    """Heatmap of NaN density per feature (columns) sampled by station."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return

    # Group by station, compute missing % per feature
    missing_by_station = (
        df.groupby("idstazione")[numeric_cols]
          .apply(lambda g: g.isnull().mean())
    )
    if missing_by_station.empty:
        return

    fig, ax = plt.subplots(figsize=(max(14, len(numeric_cols) * 0.5), max(6, len(missing_by_station) * 0.25)))
    sns.heatmap(
        missing_by_station,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        cbar_kws={"label": "% Missing"},
        ax=ax,
    )
    ax.set_title("Missing Values - Station x Feature")
    ax.set_ylabel("Station")
    ax.set_xlabel("")
    _savefig(fig, out / "01_missing_values_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# 02 — PM10 distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_pm10_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram + KDE with coloured alert-zone bands."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Background alert bands
    thresholds = PM10_THRESHOLDS.copy()
    thresholds[-1] = df["pm10"].max() * 1.05 if df["pm10"].max() > 50 else 80
    for i, label in enumerate(PM10_LABELS):
        ax.axvspan(thresholds[i], thresholds[i + 1], alpha=0.12,
                   color=ALERT_COLORS[label], label=_ALERT_LABELS.get(label, label))

    sns.histplot(df["pm10"], kde=True, bins=50, ax=ax, color="steelblue", edgecolor="white")

    # Threshold lines
    for t in PM10_THRESHOLDS[1:-1]:
        ax.axvline(t, ls="--", lw=1.2, color="gray")
        ax.text(t + 0.5, ax.get_ylim()[1] * 0.95, f"{t}", fontsize=9, color="gray")

    ax.set_title("Daily PM10 Distribution")
    ax.set_xlabel("PM10 (µg/m³)")
    ax.set_ylabel("Frequency")
    ax.legend(title="Alert class", loc="upper right")
    _savefig(fig, out / "02_pm10_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# 03 — Class distribution bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of the 4 alert classes with counts and percentages."""
    counts = df["classe_allerta"].value_counts().reindex(PM10_LABELS, fill_value=0)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    display_labels = [_ALERT_LABELS.get(label, label) for label in counts.index]
    bars = ax.bar(display_labels, counts.values,
                  color=[ALERT_COLORS[l] for l in counts.index], edgecolor="white")

    for bar, val in zip(bars, counts.values):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_title("Alert Class Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Alert class")
    _savefig(fig, out / "03_class_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# 04 — Correlation heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Lower-triangle Pearson correlation heatmap for numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(12, len(corr) * 0.55), max(10, len(corr) * 0.45)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 7},
    )
    ax.set_title("Correlation Matrix (Pearson)")
    _savefig(fig, out / "04_correlation_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# 05 — PM10 time series
# ──────────────────────────────────────────────────────────────────────────────

def plot_pm10_timeseries(df: pd.DataFrame, out: Path) -> None:
    """Daily mean PM10 with standard-deviation band."""
    daily = df.groupby("data_giorno")["pm10"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(daily["data_giorno"], daily["mean"], lw=1.2, color="steelblue", label="Daily mean")
    ax.fill_between(
        daily["data_giorno"],
        daily["mean"] - daily["std"],
        daily["mean"] + daily["std"],
        alpha=0.2, color="steelblue", label="± 1 std",
    )

    for t, label in zip(PM10_THRESHOLDS[1:-1], PM10_LABELS[1:]):
        ax.axhline(
            t,
            ls="--",
            lw=0.8,
            color=ALERT_COLORS[label],
            label=f"{_ALERT_LABELS.get(label, label)} threshold ({t})",
        )

    ax.set_title("PM10 - Daily Mean Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM10 (µg/m³)")
    ax.legend(loc="upper right", fontsize=8)
    _savefig(fig, out / "05_pm10_timeseries.png")


# ──────────────────────────────────────────────────────────────────────────────
# 06 — PM10 by season
# ──────────────────────────────────────────────────────────────────────────────

def plot_pm10_by_season(df: pd.DataFrame, out: Path) -> None:
    """Boxplot PM10 by meteorological season."""
    present = [s for s in SEASON_ORDER if s in df["stagione"].values]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="stagione",
        y="pm10",
        order=present,
        hue="stagione",
        hue_order=present,
        ax=ax,
        palette="coolwarm",
        legend=False,
    )
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([_SEASON_LABELS.get(s, s) for s in present])
    ax.set_title("PM10 by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("PM10 (µg/m³)")
    if len(present) < 4:
        ax.annotate("Note: data available only for spring-summer",
                     xy=(0.5, 0.01), xycoords="axes fraction", ha="center",
                     fontsize=8, fontstyle="italic", color="gray")
    _savefig(fig, out / "06_pm10_by_season.png")


# ──────────────────────────────────────────────────────────────────────────────
# 07 — PM10 by weekday
# ──────────────────────────────────────────────────────────────────────────────

def plot_pm10_by_weekday(df: pd.DataFrame, out: Path) -> None:
    """Boxplot PM10 by day of week."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="giorno_settimana",
        y="pm10",
        hue="giorno_settimana",
        ax=ax,
        palette="Blues",
        legend=False,
    )
    ax.set_xticks(range(len(_WEEKDAY_LABELS)))
    ax.set_xticklabels(_WEEKDAY_LABELS)
    ax.set_title("PM10 by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("PM10 (µg/m³)")
    _savefig(fig, out / "07_pm10_by_weekday.png")


# ──────────────────────────────────────────────────────────────────────────────
# 08 — PM10 monthly mean
# ──────────────────────────────────────────────────────────────────────────────

_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_pm10_monthly(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of mean ± std PM10 by calendar month."""
    monthly = (
        df.groupby("mese")["pm10"]
          .agg(["mean", "std"])
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = sns.color_palette("coolwarm", 12)
    ax.bar(
        monthly["mese"], monthly["mean"],
        yerr=monthly["std"], capsize=4,
        color=[colors[int(m) - 1] for m in monthly["mese"]],
        edgecolor="white",
    )
    ax.axhline(50, ls="--", lw=1.2, color="red", label="WHO threshold 50 µg/m³")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_title("Mean PM10 by Month (± std)")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM10 (µg/m³)")
    ax.legend(fontsize=9)
    _savefig(fig, out / "08_pm10_monthly.png")


# ──────────────────────────────────────────────────────────────────────────────
# 09 — PM10 by provincia
# ──────────────────────────────────────────────────────────────────────────────

def plot_pm10_by_provincia(df: pd.DataFrame, out: Path) -> None:
    """Boxplot PM10 by provincia, sorted by median."""
    order = df.groupby("provincia")["pm10"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df,
        x="provincia",
        y="pm10",
        order=order,
        hue="provincia",
        hue_order=order,
        ax=ax,
        palette="OrRd",
        legend=False,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("PM10 by Province (sorted by median)")
    ax.set_xlabel("Province")
    ax.set_ylabel("PM10 (µg/m³)")
    _savefig(fig, out / "09_pm10_by_provincia.png")


# ──────────────────────────────────────────────────────────────────────────────
# 10 — Weather vs PM10 scatter grid
# ──────────────────────────────────────────────────────────────────────────────

_WEATHER_SCATTER_VARS = [
    ("temp_mean", "Mean temperature (°C)"),
    ("humidity_mean", "Relative humidity (%)"),
    ("wind_speed_mean", "Wind speed (m/s)"),
    ("pressure_mean", "Pressure (hPa)"),
    ("blh_mean", "Mean BLH (m)"),
    ("precip_sum", "Precipitation (mm)"),
]


def plot_weather_vs_pm10(df: pd.DataFrame, out: Path) -> None:
    """2x3 scatter grid of key weather variables vs PM10."""
    available = [(col, label) for col, label in _WEATHER_SCATTER_VARS if col in df.columns]
    if not available:
        return

    nrows = (len(available) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, (col, label) in enumerate(available):
        ax = axes[i]
        sns.regplot(
            data=df, x=col, y="pm10", ax=ax,
            scatter_kws={"alpha": 0.15, "s": 8},
            line_kws={"color": "red", "lw": 1},
        )
        r = df[[col, "pm10"]].corr().iloc[0, 1]
        ax.set_title(f"{label}\nr = {r:.3f}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("PM10" if i % 3 == 0 else "")

    # Hide unused axes
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Weather Variables vs PM10", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, out / "10_weather_vs_pm10_scatter.png")


# ──────────────────────────────────────────────────────────────────────────────
# 11 — Stagnation vs PM10
# ──────────────────────────────────────────────────────────────────────────────

def plot_stagnation_vs_pm10(df: pd.DataFrame, out: Path) -> None:
    """Boxplot comparing PM10 in stagnation vs non-stagnation conditions."""
    if "stagnation_flag" not in df.columns or df["stagnation_flag"].isna().all():
        return

    plot_df = df.dropna(subset=["stagnation_flag", "pm10"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=plot_df, x="stagnation_flag", y="pm10", ax=ax,
        hue="stagnation_flag", palette={True: "#e53935", False: "#43a047"},
        legend=False,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No stagnation", "Stagnation"])
    ax.set_title("PM10 Under Atmospheric Stagnation Conditions")
    ax.set_xlabel("")
    ax.set_ylabel("PM10 (µg/m³)")
    _savefig(fig, out / "11_stagnation_vs_pm10.png")


# ──────────────────────────────────────────────────────────────────────────────
# 12 — Industrial proximity vs PM10
# ──────────────────────────────────────────────────────────────────────────────

def plot_industrial_vs_pm10(df: pd.DataFrame, out: Path) -> None:
    """Scatter plot of industrial proximity vs PM10."""
    vars_present = [
        ("dist_industrial_km", "Distance to industrial area (km)"),
    ]
    available = [(col, label) for col, label in vars_present if col in df.columns]
    if not available:
        log.warning("Industrial proximity columns not found — skipping plot 12.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(8 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, available):
        sns.regplot(
            data=df, x=col, y="pm10", ax=ax,
            scatter_kws={"alpha": 0.15, "s": 8},
            line_kws={"color": "red", "lw": 1.2},
        )
        r = df[[col, "pm10"]].corr().iloc[0, 1]
        ax.set_title(f"{label}\nr = {r:.3f}", fontsize=10)
        ax.set_xlabel(label)
        ax.set_ylabel("PM10 (µg/m³)")

    fig.suptitle("Industrial Proximity vs PM10", fontsize=13)
    fig.tight_layout()
    _savefig(fig, out / "12_industrial_vs_pm10.png")


# ──────────────────────────────────────────────────────────────────────────────
# 13 — NO2 hourly profile
# ──────────────────────────────────────────────────────────────────────────────

def plot_no2_hourly_profile(no2_hourly: pd.DataFrame, out: Path) -> None:
    """Mean hourly NO2 profile: weekday vs weekend."""
    if no2_hourly.empty:
        log.warning("NO2 hourly DataFrame is empty — skipping plot 13.")
        return

    if "dt" not in no2_hourly.columns or "no2" not in no2_hourly.columns:
        log.warning("NO2 hourly missing required columns. Got: %s", no2_hourly.columns.tolist())
        return

    df = no2_hourly.assign(
        ora=no2_hourly["dt"].dt.hour,
        tipo=no2_hourly["dt"].dt.dayofweek.isin([5, 6]).map(
            {True: "Weekend", False: "Weekday"}
        ),
    )
    profile = df.groupby(["ora", "tipo"])["no2"].mean().reset_index()

    log.info(f"NO2 hourly profile: {len(profile)} aggregated rows, unique hours: {profile['ora'].nunique()}, types: {profile['tipo'].unique().tolist()}")
    if len(profile) < 10:
        log.warning("Plot 13: Profile has only %d rows — may appear as scatter instead of line", len(profile))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.lineplot(data=profile, x="ora", y="no2", hue="tipo", ax=ax, marker="o", markersize=5)
    ax.set_title("NO2 Hourly Profile - Weekday vs Weekend")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean NO2 (µg/m³)")
    ax.set_xticks(range(24))
    ax.legend(title="")
    _savefig(fig, out / "13_no2_hourly_profile.png")


# -----------------------------------------------------------------------------
# 14-16 - Random Matrix Theory spectral diagnostics
# -----------------------------------------------------------------------------

def plot_rmt_eigenvalue_spectrum(
    eigenvalues: pd.DataFrame,
    summary: dict,
    out: Path,
) -> None:
    """Eigenvalue spectrum of the feature correlation matrix with MP bounds."""
    if eigenvalues.empty:
        log.warning("RMT eigenvalue table is empty - skipping plot 14.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = eigenvalues["component"]
    y = eigenvalues["eigenvalue"]
    above = eigenvalues["above_mp_bulk"]

    ax.plot(x, y, marker="o", ms=4, lw=1.2, color="#355C7D", label="Observed spectrum")
    if above.any():
        ax.scatter(
            x[above],
            y[above],
            s=45,
            color="#D62828",
            zorder=3,
            label="Above MP bulk",
        )

    if "empirical_null_rank_p95" in eigenvalues.columns:
        ax.plot(
            x,
            eigenvalues["empirical_null_rank_p95"],
            ls="--",
            lw=1.1,
            color="#6C757D",
            label="Empirical null p95",
        )

    ax.axhline(
        summary["lambda_plus_mp"],
        ls=":",
        lw=1.4,
        color="#E76F51",
        label=f"MP lambda+ = {summary['lambda_plus_mp']:.3f}",
    )
    ax.axhline(
        summary["lambda_minus_mp"],
        ls=":",
        lw=1.0,
        color="#2A9D8F",
        label=f"MP lambda- = {summary['lambda_minus_mp']:.3f}",
    )

    ax.set_title("RMT eigenvalue spectrum")
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_xlim(0.5, len(eigenvalues) + 0.5)
    ax.legend(fontsize=8)
    _savefig(fig, out / "14_rmt_eigenvalue_spectrum.png")


def plot_rmt_top_loadings(loadings: pd.DataFrame, out: Path) -> None:
    """Top feature loadings for the leading RMT signal components."""
    if loadings.empty:
        log.warning("RMT loading table is empty - skipping plot 15.")
        return

    components = loadings["component"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(13, max(4, 3.2 * len(components))),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, component in zip(axes, components):
        subset = (
            loadings[loadings["component"] == component]
            .sort_values("abs_loading", ascending=True)
        )
        colors = np.where(subset["loading"] >= 0, "#457B9D", "#E76F51")
        ax.barh(subset["feature"], subset["loading"], color=colors)
        eigenvalue = subset["eigenvalue"].iloc[0]
        marker = "MP signal" if subset["component_above_mp"].iloc[0] else "top component"
        ax.set_title(f"Component {component} - {marker} - eigenvalue {eigenvalue:.2f}")
        ax.set_xlabel("Eigenvector loading")
        ax.axvline(0, color="black", lw=0.8)

    fig.tight_layout()
    _savefig(fig, out / "15_rmt_top_eigenvector_loadings.png")


def plot_rmt_null_comparison(
    eigenvalues: pd.DataFrame,
    summary: dict,
    out: Path,
) -> None:
    """Compare leading observed eigenvalues with empirical null quantiles."""
    required = {"empirical_null_rank_p95", "empirical_null_rank_p99"}
    if eigenvalues.empty or not required.issubset(eigenvalues.columns):
        log.warning("RMT empirical null columns missing - skipping plot 16.")
        return

    top = eigenvalues.head(min(15, len(eigenvalues))).copy()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(top))

    ax.bar(
        x,
        top["eigenvalue"],
        color=np.where(top["above_empirical_p95"], "#D62828", "#8ECAE6"),
        edgecolor="white",
        label="Observed eigenvalue",
    )
    ax.plot(x, top["empirical_null_rank_p95"], marker="o", lw=1.2, color="#6C757D", label="Null p95")
    ax.plot(x, top["empirical_null_rank_p99"], marker="s", lw=1.2, color="#343A40", label="Null p99")
    ax.axhline(summary["lambda_plus_mp"], ls=":", color="#E76F51", lw=1.3, label="MP lambda+")

    ax.set_xticks(x)
    ax.set_xticklabels(top["component"].astype(str))
    ax.set_title(
        "RMT observed spectrum vs empirical null "
        f"({summary['empirical_null_method']}, n={summary['n_null_iterations']})"
    )
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Eigenvalue")
    ax.legend(fontsize=8)
    _savefig(fig, out / "16_rmt_empirical_null_comparison.png")
