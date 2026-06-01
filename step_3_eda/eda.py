"""Step 3 — Exploratory Data Analysis.

Usage
-----
# Ensure MySQL is running:
#   cd step_2_ingestion && docker compose up -d
#   python step_2_ingestion/ingest.py   (if not already done)

# Then run from project root:
python -m step_3_eda.eda
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from step_3_eda.analysis import (
    class_distribution,
    descriptive_stats,
    missing_value_report,
    station_coverage_report,
    top_correlations,
)
from step_3_eda.build_dataset import (
    build_daily_dataset,
    clean_dataset,
    impute_missing,
    save_dataset,
)
from step_3_eda.db import connect_mysql, load_no2_hourly
from step_3_eda.plots import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_industrial_vs_pm10,
    plot_missing_values,
    plot_no2_hourly_profile,
    plot_pm10_by_provincia,
    plot_pm10_by_season,
    plot_pm10_by_weekday,
    plot_pm10_distribution,
    plot_pm10_monthly,
    plot_pm10_timeseries,
    plot_rmt_eigenvalue_spectrum,
    plot_rmt_null_comparison,
    plot_rmt_top_loadings,
    plot_stagnation_vs_pm10,
    plot_weather_vs_pm10,
)
from step_3_eda.raw import load_no2_hourly_from_raw
from step_3_eda.rmt import run_rmt_diagnostic, save_rmt_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parent / "plots"


def main() -> None:
    log.info("=== Step 3: EDA start ===")
    PLOTS_DIR.mkdir(exist_ok=True)

    # ── 1. Connect ───────────────────────────────────────────────────────────
    try:
        conn = connect_mysql()
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)

    output_dir = Path(__file__).parent

    try:
        # ── 2. Build raw dataset ─────────────────────────────────────────────
        df_raw = build_daily_dataset(conn)

        # ── 3. Pre-imputation analysis ───────────────────────────────────────
        log.info("─── Missing-value report (pre-imputation) ───")
        mv_report = missing_value_report(df_raw)
        log.info("\n%s", mv_report.to_string(index=False))

        log.info("─── Station coverage ───")
        cov_report = station_coverage_report(df_raw)
        log.info("\n%s", cov_report.to_string(index=False))

        # Plot 01 — missing values (BEFORE imputation)
        plot_missing_values(df_raw, PLOTS_DIR)

        # ── 4. Clean dataset (safe pre-split operations) ─────────────────────
        df_clean = clean_dataset(df_raw)

        # Save the clean (pre-imputation) dataset for ML pipelines.
        # step_4/step_5 MUST call impute_missing() AFTER their train/test split.
        save_dataset(df_clean, output_dir, filename="daily_dataset_clean.parquet")

        # Full-dataset imputation for EDA visualization only (NOT for ML).
        df, _ = impute_missing(df_clean)

        # ── 5. Analysis ─────────────────────────────────────────────────────
        log.info("─── Class distribution ───")
        dist = class_distribution(df)

        log.info("─── Descriptive statistics ───")
        desc = descriptive_stats(df)
        log.info("\n%s", desc.to_string())

        log.info("─── Top correlations with PM10 ───")
        top_corr = top_correlations(df)
        log.info("\n%s", top_corr.to_string())

        log.info("RMT spectral diagnostic")
        rmt_result = None
        try:
            rmt_result = run_rmt_diagnostic(df)
            save_rmt_artifacts(rmt_result, output_dir)
            rmt_summary = rmt_result["summary"]
            log.info(
                "RMT summary: %d/%d components above MP bulk; "
                "%.1f%% variance explained by MP-significant components.",
                rmt_summary["n_components_above_mp"],
                rmt_summary["n_features"],
                rmt_summary["variance_explained_above_mp"] * 100,
            )
        except ValueError as exc:
            log.warning("Skipping RMT diagnostic: %s", exc)

        # ── 6. Plots ────────────────────────────────────────────────────────
        log.info("─── Generating plots ───")
        plot_pm10_distribution(df, PLOTS_DIR)
        plot_class_distribution(df, PLOTS_DIR)
        plot_correlation_heatmap(df, PLOTS_DIR)
        if rmt_result is not None:
            plot_rmt_eigenvalue_spectrum(rmt_result["eigenvalues"], rmt_result["summary"], PLOTS_DIR)
            plot_rmt_top_loadings(rmt_result["loadings"], PLOTS_DIR)
            plot_rmt_null_comparison(rmt_result["eigenvalues"], rmt_result["summary"], PLOTS_DIR)
        plot_pm10_timeseries(df, PLOTS_DIR)
        plot_pm10_by_season(df, PLOTS_DIR)
        plot_pm10_by_weekday(df, PLOTS_DIR)
        plot_pm10_monthly(df, PLOTS_DIR)
        plot_pm10_by_provincia(df, PLOTS_DIR)
        plot_weather_vs_pm10(df, PLOTS_DIR)
        plot_stagnation_vs_pm10(df, PLOTS_DIR)
        plot_industrial_vs_pm10(df, PLOTS_DIR)

        # NO2 hourly profile (hourly granularity; prefer local raw JSON so
        # this artifact can be regenerated without an extra DB round-trip).
        try:
            no2_h = load_no2_hourly_from_raw()
            if no2_h.empty:
                log.warning("Raw NO2 hourly data is empty; falling back to MySQL.")
                no2_h = load_no2_hourly(conn)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Raw NO2 hourly data unavailable (%s); falling back to MySQL.", exc)
            no2_h = load_no2_hourly(conn)
        plot_no2_hourly_profile(no2_h, PLOTS_DIR)

        # ── 7. Save EDA-imputed dataset (for visualization/notebooks only) ──
        save_dataset(df, output_dir)

        # ── 8. Summary ──────────────────────────────────────────────────────
        date_range = f"{df['data_giorno'].min().date()} → {df['data_giorno'].max().date()}"
        log.info(
            "\n"
            "═══════════════════════════════════════\n"
            "  EDA Summary\n"
            "═══════════════════════════════════════\n"
            "  Rows:           %d\n"
            "  Stations:       %d\n"
            "  Date range:     %s\n"
            "  Columns:        %d\n"
            "  Class counts:\n%s\n"
            "  Top-5 corr with PM10:\n%s\n"
            "═══════════════════════════════════════",
            len(df),
            df["idstazione"].nunique(),
            date_range,
            df.shape[1],
            dist.to_string(index=False),
            top_corr.head(5).to_string(),
        )

    finally:
        conn.close()

    log.info("=== Step 3: EDA complete ===")


if __name__ == "__main__":
    main()
