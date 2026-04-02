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

_WEEKDAY_LABELS = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]


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
    ax.set_title("Missing Values — Stazione x Feature")
    ax.set_ylabel("Stazione")
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
                   color=ALERT_COLORS[label], label=label)

    sns.histplot(df["pm10"], kde=True, bins=50, ax=ax, color="steelblue", edgecolor="white")

    # Threshold lines
    for t in PM10_THRESHOLDS[1:-1]:
        ax.axvline(t, ls="--", lw=1.2, color="gray")
        ax.text(t + 0.5, ax.get_ylim()[1] * 0.95, f"{t}", fontsize=9, color="gray")

    ax.set_title("Distribuzione PM10 giornaliero")
    ax.set_xlabel("PM10 (µg/m³)")
    ax.set_ylabel("Frequenza")
    ax.legend(title="Classe allerta", loc="upper right")
    _savefig(fig, out / "02_pm10_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# 03 — Class distribution bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of the 4 alert classes with counts and percentages."""
    counts = df["classe_allerta"].value_counts().reindex(PM10_LABELS, fill_value=0)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[ALERT_COLORS[l] for l in counts.index], edgecolor="white")

    for bar, val in zip(bars, counts.values):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_title("Distribuzione classi di allerta")
    ax.set_ylabel("Conteggio")
    ax.set_xlabel("Classe allerta")
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
    ax.set_title("Matrice di correlazione (Pearson)")
    _savefig(fig, out / "04_correlation_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# 05 — PM10 time series
# ──────────────────────────────────────────────────────────────────────────────

def plot_pm10_timeseries(df: pd.DataFrame, out: Path) -> None:
    """Daily mean PM10 with standard-deviation band."""
    daily = df.groupby("data_giorno")["pm10"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(daily["data_giorno"], daily["mean"], lw=1.2, color="steelblue", label="Media giornaliera")
    ax.fill_between(
        daily["data_giorno"],
        daily["mean"] - daily["std"],
        daily["mean"] + daily["std"],
        alpha=0.2, color="steelblue", label="± 1 std",
    )

    for t, label in zip(PM10_THRESHOLDS[1:-1], PM10_LABELS[1:]):
        ax.axhline(t, ls="--", lw=0.8, color=ALERT_COLORS[label], label=f"Soglia {label} ({t})")

    ax.set_title("PM10 — Serie temporale media giornaliera")
    ax.set_xlabel("Data")
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
    sns.boxplot(data=df, x="stagione", y="pm10", order=present, ax=ax, palette="coolwarm")
    ax.set_title("PM10 per stagione")
    ax.set_xlabel("Stagione")
    ax.set_ylabel("PM10 (µg/m³)")
    if len(present) < 4:
        ax.annotate("Nota: dati disponibili solo per primavera-estate",
                     xy=(0.5, 0.01), xycoords="axes fraction", ha="center",
                     fontsize=8, fontstyle="italic", color="gray")
    _savefig(fig, out / "06_pm10_by_season.png")


# ──────────────────────────────────────────────────────────────────────────────
# 07 — PM10 by weekday
# ──────────────────────────────────────────────────────────────────────────────

def plot_pm10_by_weekday(df: pd.DataFrame, out: Path) -> None:
    """Boxplot PM10 by day of week."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="giorno_settimana", y="pm10", ax=ax, palette="Blues")
    ax.set_xticklabels(_WEEKDAY_LABELS)
    ax.set_title("PM10 per giorno della settimana")
    ax.set_xlabel("Giorno")
    ax.set_ylabel("PM10 (µg/m³)")
    _savefig(fig, out / "07_pm10_by_weekday.png")


# ──────────────────────────────────────────────────────────────────────────────
# 08 — PM10 monthly mean
# ──────────────────────────────────────────────────────────────────────────────

_MONTH_LABELS = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                 "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]


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
    ax.axhline(50, ls="--", lw=1.2, color="red", label="Soglia OMS 50 µg/m³")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_title("PM10 medio per mese (± std)")
    ax.set_xlabel("Mese")
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
    sns.boxplot(data=df, x="provincia", y="pm10", order=order, ax=ax, palette="OrRd")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("PM10 per provincia (ordinate per mediana)")
    ax.set_xlabel("Provincia")
    ax.set_ylabel("PM10 (µg/m³)")
    _savefig(fig, out / "09_pm10_by_provincia.png")


# ──────────────────────────────────────────────────────────────────────────────
# 10 — Weather vs PM10 scatter grid
# ──────────────────────────────────────────────────────────────────────────────

_WEATHER_SCATTER_VARS = [
    ("temp_mean", "Temperatura media (°C)"),
    ("humidity_mean", "Umidità relativa (%)"),
    ("wind_speed_mean", "Velocità vento (m/s)"),
    ("pressure_mean", "Pressione (hPa)"),
    ("blh_mean", "BLH media (m)"),
    ("precip_sum", "Precipitazione (mm)"),
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

    fig.suptitle("Variabili meteo vs PM10", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, out / "10_weather_vs_pm10_scatter.png")


# ──────────────────────────────────────────────────────────────────────────────
# 11 — Stagnation vs PM10
# ──────────────────────────────────────────────────────────────────────────────

def plot_stagnation_vs_pm10(df: pd.DataFrame, out: Path) -> None:
    """Boxplot comparing PM10 in stagnation vs non-stagnation conditions."""
    if "stagnation_flag" not in df.columns or df["stagnation_flag"].isna().all():
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=df, x="stagnation_flag", y="pm10", ax=ax,
        hue="stagnation_flag", palette={True: "#e53935", False: "#43a047"},
        legend=False,
    )
    ax.set_xticklabels(["No stagnazione", "Stagnazione"])
    ax.set_title("PM10 in condizioni di stagnazione atmosferica")
    ax.set_xlabel("")
    ax.set_ylabel("PM10 (µg/m³)")
    _savefig(fig, out / "11_stagnation_vs_pm10.png")


# ──────────────────────────────────────────────────────────────────────────────
# 12 — Industrial proximity vs PM10
# ──────────────────────────────────────────────────────────────────────────────

def plot_industrial_vs_pm10(df: pd.DataFrame, out: Path) -> None:
    """Scatter plots of industrial proximity features vs PM10."""
    vars_present = [
        ("dist_industrial_km", "Distanza zona industriale (km)"),
        ("n_industrial_zones_15km", "N. zone industriali entro 15 km"),
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

    fig.suptitle("Prossimità industriale vs PM10", fontsize=13)
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
            {True: "Weekend", False: "Feriale"}
        ),
    )
    profile = df.groupby(["ora", "tipo"])["no2"].mean().reset_index()

    log.info(f"NO2 hourly profile: {len(profile)} aggregated rows, unique hours: {profile['ora'].nunique()}, types: {profile['tipo'].unique().tolist()}")
    if len(profile) < 10:
        log.warning("Plot 13: Profile has only %d rows — may appear as scatter instead of line", len(profile))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.lineplot(data=profile, x="ora", y="no2", hue="tipo", ax=ax, marker="o", markersize=5)
    ax.set_title("Profilo orario NO2 — Feriale vs Weekend")
    ax.set_xlabel("Ora del giorno")
    ax.set_ylabel("NO2 medio (µg/m³)")
    ax.set_xticks(range(24))
    ax.legend(title="")
    _savefig(fig, out / "13_no2_hourly_profile.png")
