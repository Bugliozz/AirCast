"""Build one-row-per-station aggregated profiles for clustering."""

from __future__ import annotations

import pandas as pd

from step_6_clustering.config import META_COLS, PROFILE_FEATURES, PROFILES_FILE


def build_station_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows to one profile per station.

    Parameters
    ----------
    df:
        daily_dataset_clean with one row per (idstazione, data_giorno).

    Returns
    -------
    pd.DataFrame with 67 rows, columns = META_COLS + PROFILE_FEATURES.
    """
    critical = df["classe_allerta"].isin({"arancio", "rosso"})

    # Per-station winter and summer PM10 means for seasonality_ratio
    winter_pm10 = (
        df[df["stagione"] == "inverno"]
        .groupby("idstazione")["pm10"]
        .mean()
        .rename("_pm10_winter")
    )
    summer_pm10 = (
        df[df["stagione"] == "estate"]
        .groupby("idstazione")["pm10"]
        .mean()
        .rename("_pm10_summer")
    )

    pct_critical = (
        df.assign(_critical=critical)
        .groupby("idstazione")["_critical"]
        .mean()
        .rename("pct_critical_days")
    )

    profiles = (
        df.groupby("idstazione")
        .agg(
            nomestazione=("nomestazione", "first"),
            provincia=("provincia", "first"),
            comune=("comune", "first"),
            lat=("lat", "first"),
            lng=("lng", "first"),
            pm10_mean=("pm10", "mean"),
            stagnation_index_mean=("stagnation_index", "mean"),
            dist_industrial_km=("dist_industrial_km", "first"),
            no2_mean=("no2_mean", "mean"),
            quota=("quota", "first"),
            wind_speed_mean=("wind_speed_mean", "mean"),
        )
        .reset_index()
        .join(pct_critical, on="idstazione")
    )

    profiles = profiles.join(winter_pm10, on="idstazione").join(
        summer_pm10, on="idstazione"
    )
    profiles["seasonality_ratio"] = (
        profiles["_pm10_winter"] / profiles["_pm10_summer"]
    )
    profiles = profiles.drop(columns=["_pm10_winter", "_pm10_summer"])

    # Impute missing numeric features with province median, fallback global median
    for col in ("quota", "no2_mean"):
        null_mask = profiles[col].isna()
        if not null_mask.any():
            continue
        prov_median = (
            profiles.dropna(subset=[col])
            .groupby("provincia")[col]
            .median()
        )
        global_median = profiles[col].median()
        for idx in profiles.index[null_mask]:
            prov = profiles.at[idx, "provincia"]
            profiles.at[idx, col] = prov_median.get(prov, global_median)

    col_order = ["idstazione"] + META_COLS[1:] + PROFILE_FEATURES
    profiles = profiles[col_order]

    assert not profiles[PROFILE_FEATURES].isna().any().any(), (
        "NaN found in PROFILE_FEATURES after build — check source data."
    )

    return profiles


def main() -> None:
    df = pd.read_parquet("step_3_eda/daily_dataset_clean.parquet")
    profiles = build_station_profiles(df)
    profiles.to_parquet(PROFILES_FILE, index=False)
    print(f"Saved {len(profiles)} station profiles -> {PROFILES_FILE}")
    print(profiles[PROFILE_FEATURES].describe().round(3).to_string())


if __name__ == "__main__":
    main()
