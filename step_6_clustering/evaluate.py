"""Evaluate clustering results — validation, severity analysis, and plots.

Pipeline
--------
1.  Load station_profiles.parquet + station_clusters.csv → merged DataFrame.
2.  External validation vs ``provincia`` (independent): ARI, homogeneity,
    completeness, v_measure → appended to clustering_metrics.json.
3.  Severity separation: Kruskal-Wallis on pm10_mean and pct_critical_days
    per cluster + boxplot.
4.  Caveat circularity: homogeneity vs dominant classe_allerta (NOT
    independent — derived from same PM10 values).  Saved with
    ``"non_independent": true`` flag.
5.  Cluster interpretation table: mean raw features per cluster → CSV + JSON.
6.  Plots:
      elbow.png                     inertia vs k (KMeans)
      silhouette_vs_k.png           silhouette for KMeans + Agglomerative
      dendrogram.png                Ward linkage dendrogram
      dbscan_kdistance.png          sorted k-distance for eps selection
      pca_2d_clusters.png           PCA 2D coloured by cluster
      cluster_profile_heatmap.png   mean standardised features per cluster
      pm10_by_cluster_boxplot.png   pm10_mean distribution per cluster

Usage::

    python -m step_6_clustering.evaluate
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder

from step_6_clustering.config import (
    ARTIFACTS_DIR,
    CLUSTERS_FILE,
    DBSCAN_MIN_SAMPLES,
    METRICS_FILE,
    PLOTS_DIR,
    PROFILE_FEATURES,
    PROFILES_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_RAW_DATA_PATH = "step_3_eda/daily_dataset_clean.parquet"
_INTERPRETATION_CSV = f"{ARTIFACTS_DIR}/cluster_interpretation.csv"
_INTERPRETATION_JSON = f"{ARTIFACTS_DIR}/cluster_interpretation.json"

# Matplotlib style: clean, no interactive backend
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 10,
    }
)

_CLUSTER_PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


def _cluster_color(label: int) -> str:
    return _CLUSTER_PALETTE[label % len(_CLUSTER_PALETTE)]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_merged() -> pd.DataFrame:
    profiles = pd.read_parquet(PROFILES_FILE)
    clusters = pd.read_csv(CLUSTERS_FILE)
    clusters["idstazione"] = clusters["idstazione"].astype(str)
    df = profiles.merge(clusters, on="idstazione")
    logger.info("Loaded merged data: %d stations, %d columns.", *df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. External validation vs provincia
# ---------------------------------------------------------------------------


def external_validation(df: pd.DataFrame) -> dict[str, float]:
    le = LabelEncoder()
    labels_true = le.fit_transform(df["provincia"])
    labels_pred = df["cluster"].values

    metrics = {
        "adjusted_rand_score": float(adjusted_rand_score(labels_true, labels_pred)),
        "homogeneity": float(homogeneity_score(labels_true, labels_pred)),
        "completeness": float(completeness_score(labels_true, labels_pred)),
        "v_measure": float(v_measure_score(labels_true, labels_pred)),
    }
    logger.info(
        "External validation (vs provincia): ARI=%.3f  hom=%.3f  comp=%.3f  v=%.3f",
        metrics["adjusted_rand_score"],
        metrics["homogeneity"],
        metrics["completeness"],
        metrics["v_measure"],
    )
    return metrics


# ---------------------------------------------------------------------------
# 3. Circularity caveat — homogeneity vs dominant classe_allerta
# ---------------------------------------------------------------------------


def circularity_caveat(df: pd.DataFrame) -> dict[str, Any]:
    """Compute homogeneity vs dominant classe_allerta per station.

    classe_allerta is derived from PM10 thresholds — the same signal as
    pm10_mean which drives the clustering.  This is NOT an independent
    validation and must be clearly labelled as such.
    """
    raw = pd.read_parquet(_RAW_DATA_PATH, columns=["idstazione", "classe_allerta"])
    raw["idstazione"] = raw["idstazione"].astype(str)
    dominant = (
        raw.groupby("idstazione")["classe_allerta"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={"classe_allerta": "dominant_classe_allerta"})
    )
    merged = df[["idstazione", "cluster"]].merge(dominant, on="idstazione", how="left")

    le = LabelEncoder()
    labels_true = le.fit_transform(merged["dominant_classe_allerta"])
    labels_pred = merged["cluster"].values

    result: dict[str, Any] = {
        "non_independent": True,
        "note": (
            "The alert class is derived from PM10 thresholds — the same signal "
            "that drives pm10_mean in the feature set.  This metric is reported "
            "for completeness only and must NOT be used as evidence of validity."
        ),
        "homogeneity_vs_dominant_classe_allerta": float(
            homogeneity_score(labels_true, labels_pred)
        ),
    }
    logger.info(
        "Circularity caveat (NON-INDEPENDENT) homogeneity vs classe_allerta: %.3f",
        result["homogeneity_vs_dominant_classe_allerta"],
    )
    return result


# ---------------------------------------------------------------------------
# 4. Severity separation — Kruskal-Wallis
# ---------------------------------------------------------------------------


def severity_separation(df: pd.DataFrame) -> dict[str, Any]:
    groups_pm10 = [grp["pm10_mean"].values for _, grp in df.groupby("cluster")]
    groups_pct = [
        grp["pct_critical_days"].values for _, grp in df.groupby("cluster")
    ]

    stat_pm10, p_pm10 = kruskal(*groups_pm10)
    stat_pct, p_pct = kruskal(*groups_pct)

    result = {
        "pm10_mean": {"kruskal_wallis_stat": float(stat_pm10), "p_value": float(p_pm10)},
        "pct_critical_days": {
            "kruskal_wallis_stat": float(stat_pct),
            "p_value": float(p_pct),
        },
    }
    logger.info(
        "Kruskal-Wallis pm10_mean: stat=%.3f p=%.4g | pct_critical_days: stat=%.3f p=%.4g",
        stat_pm10, p_pm10, stat_pct, p_pct,
    )
    return result


# ---------------------------------------------------------------------------
# 5. Cluster interpretation table
# ---------------------------------------------------------------------------


def build_interpretation_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.groupby("cluster")[PROFILE_FEATURES].mean().round(4)
    table.index.name = "cluster"
    table.to_csv(_INTERPRETATION_CSV)
    table.reset_index().to_json(
        _INTERPRETATION_JSON, orient="records", indent=2, force_ascii=False
    )
    logger.info("Cluster interpretation table saved.")
    return table


# ---------------------------------------------------------------------------
# 6. Metrics JSON update
# ---------------------------------------------------------------------------


def update_metrics_json(
    ext_val: dict[str, float],
    circ: dict[str, Any],
    sev: dict[str, Any],
) -> None:
    with open(METRICS_FILE, encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)

    payload["external_validation"] = ext_val
    payload["severity_separation"] = sev
    payload["circularity_caveat"] = circ

    with open(METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("clustering_metrics.json updated with evaluation results.")


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------


def _savefig(name: str) -> None:
    path = Path(PLOTS_DIR) / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def plot_elbow(metrics: dict[str, Any]) -> None:
    ks = sorted(int(k) for k in metrics["kmeans"])
    inertias = [metrics["kmeans"][str(k)]["inertia"] for k in ks]
    best_k = metrics["best_model"]["k"]
    if metrics["best_model"]["algo"] != "kmeans":
        best_k = None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, marker="o", color="#2196F3", linewidth=2)
    if best_k is not None:
        ax.axvline(best_k, color="#FF5722", linestyle="--", label=f"best k={best_k}")
        ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("KMeans Elbow Curve")
    _savefig("elbow.png")


def plot_silhouette_vs_k(metrics: dict[str, Any]) -> None:
    ks = sorted(int(k) for k in metrics["kmeans"])
    sil_km = [metrics["kmeans"][str(k)]["silhouette"] for k in ks]
    sil_agg = [metrics["agglomerative"][str(k)]["silhouette"] for k in ks]
    best_k = metrics["best_model"]["k"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, sil_km, marker="o", color="#2196F3", label="KMeans", linewidth=2)
    ax.plot(ks, sil_agg, marker="s", color="#FF9800", label="Agglomerative (Ward)", linewidth=2)
    ax.axvline(best_k, color="#FF5722", linestyle="--", label=f"best k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette Score vs k")
    ax.legend()
    _savefig("silhouette_vs_k.png")


def plot_dendrogram(linkage_matrix: np.ndarray, station_labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(
        linkage_matrix,
        labels=station_labels,
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=0.7 * linkage_matrix[-1, 2],
        ax=ax,
    )
    ax.set_title("Ward Linkage Dendrogram (67 stations)")
    ax.set_ylabel("Distance")
    ax.set_xlabel("Station")
    plt.tight_layout()
    _savefig("dendrogram.png")


def plot_dbscan_kdistance(X_scaled: np.ndarray, min_samples: int) -> None:
    # k-th nearest neighbour distance for each point (k = min_samples)
    dist_matrix = cdist(X_scaled, X_scaled)
    np.fill_diagonal(dist_matrix, np.inf)
    kth_distances = np.sort(dist_matrix, axis=1)[:, min_samples - 1]
    sorted_distances = np.sort(kth_distances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sorted_distances, color="#2196F3", linewidth=2)
    ax.set_xlabel("Points (sorted by distance)")
    ax.set_ylabel(f"{min_samples}-NN distance")
    ax.set_title(
        f"DBSCAN k-distance graph (min_samples={min_samples})\n"
        "Inflection point → candidate eps"
    )
    _savefig("dbscan_kdistance.png")


def plot_pca_2d(X_scaled: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    unique_labels = sorted(np.unique(labels))
    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"Cluster {lbl}",
            color=_cluster_color(lbl),
            s=70,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var.)")
    ax.set_title(
        f"PCA 2D — clusters\n"
        f"Explained variance: {sum(var_explained):.1%}"
    )
    ax.legend()
    _savefig("pca_2d_clusters.png")


def plot_cluster_profile_heatmap(X_scaled: np.ndarray, labels: np.ndarray) -> None:
    unique_labels = sorted(np.unique(labels))
    mean_profiles = np.array(
        [X_scaled[labels == lbl].mean(axis=0) for lbl in unique_labels]
    )

    fig, ax = plt.subplots(figsize=(10, max(3, len(unique_labels) + 1)))
    im = ax.imshow(mean_profiles, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(PROFILE_FEATURES)))
    ax.set_xticklabels(PROFILE_FEATURES, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels([f"Cluster {lbl}" for lbl in unique_labels])
    plt.colorbar(im, ax=ax, label="Mean standardised value")
    ax.set_title("Cluster Profile Heatmap (mean standardised features)")
    plt.tight_layout()
    _savefig("cluster_profile_heatmap.png")


def plot_pm10_by_cluster_boxplot(df: pd.DataFrame, p_value: float) -> None:
    unique_labels = sorted(df["cluster"].unique())
    data_by_cluster = [df.loc[df["cluster"] == lbl, "pm10_mean"].values for lbl in unique_labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        data_by_cluster,
        tick_labels=[f"Cluster {lbl}" for lbl in unique_labels],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, lbl in zip(bp["boxes"], unique_labels):
        patch.set_facecolor(_cluster_color(lbl))
        patch.set_alpha(0.75)

    significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "n.s."))
    ax.set_ylabel("PM10 mean (µg/m³)")
    ax.set_title(
        f"PM10 distribution by cluster\n"
        f"Kruskal-Wallis p={p_value:.3g} {significance}"
    )
    _savefig("pm10_by_cluster_boxplot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_evaluation() -> None:
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = load_merged()
    scaler = joblib.load(Path(ARTIFACTS_DIR) / "scaler.joblib")
    X_scaled = scaler.transform(df[PROFILE_FEATURES].values)
    labels = df["cluster"].values
    linkage_matrix = np.load(Path(ARTIFACTS_DIR) / "linkage_matrix.npy")

    with open(METRICS_FILE, encoding="utf-8") as fh:
        metrics_payload: dict[str, Any] = json.load(fh)

    # ------------------------------------------------------------------
    # 2. External validation
    # ------------------------------------------------------------------
    ext_val = external_validation(df)

    # ------------------------------------------------------------------
    # 3. Severity separation
    # ------------------------------------------------------------------
    sev = severity_separation(df)

    # ------------------------------------------------------------------
    # 4. Circularity caveat
    # ------------------------------------------------------------------
    circ = circularity_caveat(df)

    # ------------------------------------------------------------------
    # 5. Interpretation table
    # ------------------------------------------------------------------
    build_interpretation_table(df)

    # ------------------------------------------------------------------
    # 6. Update metrics JSON
    # ------------------------------------------------------------------
    update_metrics_json(ext_val, circ, sev)

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    plot_elbow(metrics_payload)
    plot_silhouette_vs_k(metrics_payload)
    plot_dendrogram(linkage_matrix, df["nomestazione"].tolist())
    plot_dbscan_kdistance(X_scaled, DBSCAN_MIN_SAMPLES)
    plot_pca_2d(X_scaled, labels)
    plot_cluster_profile_heatmap(X_scaled, labels)
    plot_pm10_by_cluster_boxplot(df, sev["pm10_mean"]["p_value"])

    logger.info("Evaluation complete — all artifacts in %s", ARTIFACTS_DIR)


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
