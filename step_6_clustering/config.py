"""Configuration for Step 6 — Station clustering (N=67 stations).

Feature set is deliberately lean: 8 domain-meaningful variables that capture
air-quality severity, seasonality, geography, and meteorology without
collinearity from raw daily counts.
"""

from typing import List

# ---------------------------------------------------------------------------
# Clustering features (8 station-level aggregates, no meta/identifier cols)
# ---------------------------------------------------------------------------
PROFILE_FEATURES: List[str] = [
    "pm10_mean",               # overall pollution level
    "pct_critical_days",       # fraction of days above alert threshold
    "seasonality_ratio",       # winter/summer PM10 ratio (seasonal pattern)
    "stagnation_index_mean",   # atmospheric stagnation proxy
    "dist_industrial_km",      # proximity to industrial sources
    "no2_mean",                # traffic/combustion tracer
    "quota",                   # altitude (topographic dilution effect)
    "wind_speed_mean",         # dispersion capacity
]

# ---------------------------------------------------------------------------
# Station metadata columns — kept for map/validation but NOT used as features
# ---------------------------------------------------------------------------
META_COLS: List[str] = [
    "idstazione",
    "nomestazione",
    "provincia",
    "comune",
    "lat",
    "lng",
]

# ---------------------------------------------------------------------------
# External label for cluster validation (independent of feature set)
# ---------------------------------------------------------------------------
EXTERNAL_LABEL_COL: str = "provincia"

# ---------------------------------------------------------------------------
# Scaler — RobustScaler is preferred for small N and outlier-prone features
# (option "standard" available for ablation comparison)
# ---------------------------------------------------------------------------
SCALER: str = "robust"  # "standard" for comparison

# ---------------------------------------------------------------------------
# KMeans / Agglomerative — candidate k values
# Range kept narrow given N=67: too large a k yields unstable micro-clusters
# ---------------------------------------------------------------------------
K_RANGE = range(2, 9)

KMEANS_PARAMS = {
    "n_init": 10,
    "random_state": 42,
}

AGGLOMERATIVE_PARAMS = {
    "linkage": "ward",  # minimises within-cluster variance; works with Euclidean
}

# ---------------------------------------------------------------------------
# DBSCAN — grid over eps on scaled space; min_samples kept low given N=67
# ---------------------------------------------------------------------------
DBSCAN_EPS_GRID: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]
DBSCAN_MIN_SAMPLES: int = 3

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR: str = "step_6_clustering/artifacts"
PLOTS_DIR: str = "step_6_clustering/artifacts/plots"
PROFILES_FILE: str = "step_6_clustering/artifacts/station_profiles.parquet"
CLUSTERS_FILE: str = "step_6_clustering/artifacts/station_clusters.csv"
METRICS_FILE: str = "step_6_clustering/artifacts/clustering_metrics.json"
MAP_FILE: str = "step_6_clustering/artifacts/cluster_map.html"
