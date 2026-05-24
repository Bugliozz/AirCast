"""Fit clustering models on 67-station profiles and select the best.

Pipeline
--------
1. Load ``station_profiles.parquet`` produced by ``profiles.py``.
2. Scale features with RobustScaler fitted on all 67 rows.
3. KMeans on K_RANGE — inertia, silhouette, calinski_harabasz, davies_bouldin.
4. AgglomerativeClustering (Ward) on K_RANGE — same metrics + linkage matrix.
5. DBSCAN on DBSCAN_EPS_GRID — silhouette (non-noise only), n_clusters, n_noise.
6. Select best (algo, k) by silhouette; assign labels to 67 stations.
7. Persist artefacts: scaler.joblib, kmeans_best.joblib, agglomerative_best.joblib,
   station_clusters.csv, clustering_metrics.json.

Usage::

    python -m step_6_clustering.cluster
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

from step_6_clustering.config import (
    AGGLOMERATIVE_PARAMS,
    ARTIFACTS_DIR,
    CLUSTERS_FILE,
    DBSCAN_EPS_GRID,
    DBSCAN_MIN_SAMPLES,
    K_RANGE,
    KMEANS_PARAMS,
    METRICS_FILE,
    PROFILE_FEATURES,
    PROFILES_FILE,
    RANDOM_STATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_SENTINEL_SILHOUETTE = -2.0  # below valid range [-1, 1]

# A cluster smaller than this fraction of N/k is considered degenerate.
# At N=67, k=2 this means each cluster must have ≥ 8 stations.
_MIN_CLUSTER_FRAC = 0.25


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_interpretable(labels: np.ndarray, n_total: int, k: int) -> bool:
    """Return False if any cluster is too small to be interpretable.

    Threshold: smallest cluster < 25% of the expected balanced size (N/k),
    floored at 3 stations absolute minimum.
    """
    expected = n_total / k
    min_size = max(3, round(expected * _MIN_CLUSTER_FRAC))
    counts = np.bincount(labels[labels >= 0])
    return bool(counts.min() >= min_size)


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Return silhouette score, or _SENTINEL_SILHOUETTE if n_clusters < 2."""
    unique = set(labels) - {-1}
    if len(unique) < 2:
        return _SENTINEL_SILHOUETTE
    mask = labels != -1
    if mask.sum() < 2:
        return _SENTINEL_SILHOUETTE
    return float(silhouette_score(X[mask], labels[mask]))


def _fit_kmeans_grid(
    X: np.ndarray,
    k_range: range,
    params: dict,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    n = len(X)
    for k in k_range:
        km = KMeans(n_clusters=k, **params)
        labels = km.fit_predict(X)
        interpretable = _is_interpretable(labels, n, k)
        results[k] = {
            "inertia": float(km.inertia_),
            "silhouette": _safe_silhouette(X, labels),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "cluster_sizes": np.bincount(labels).tolist(),
            "interpretable": interpretable,
        }
        logger.info(
            "KMeans k=%d | sil=%.3f | ch=%.1f | db=%.3f | inertia=%.1f | sizes=%s%s",
            k,
            results[k]["silhouette"],
            results[k]["calinski_harabasz"],
            results[k]["davies_bouldin"],
            results[k]["inertia"],
            results[k]["cluster_sizes"],
            "" if interpretable else " [DEGENERATE]",
        )
    return results


def _fit_agglomerative_grid(
    X: np.ndarray,
    k_range: range,
    params: dict,
) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    results: dict[int, dict[str, Any]] = {}
    linkage_matrix = scipy_linkage(X, method=params["linkage"])
    n = len(X)
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, **params)
        labels = agg.fit_predict(X)
        interpretable = _is_interpretable(labels, n, k)
        results[k] = {
            "silhouette": _safe_silhouette(X, labels),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "cluster_sizes": np.bincount(labels).tolist(),
            "interpretable": interpretable,
        }
        logger.info(
            "Agglomerative k=%d | sil=%.3f | ch=%.1f | db=%.3f | sizes=%s%s",
            k,
            results[k]["silhouette"],
            results[k]["calinski_harabasz"],
            results[k]["davies_bouldin"],
            results[k]["cluster_sizes"],
            "" if interpretable else " [DEGENERATE]",
        )
    return results, linkage_matrix


def _fit_dbscan_grid(
    X: np.ndarray,
    eps_grid: list[float],
    min_samples: int,
) -> dict[float, dict[str, Any]]:
    results: dict[float, dict[str, Any]] = {}
    for eps in eps_grid:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        sil = _safe_silhouette(X, labels)
        results[eps] = {
            "silhouette": sil,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }
        logger.info(
            "DBSCAN eps=%.2f | n_clusters=%d | n_noise=%d | sil=%.3f",
            eps,
            n_clusters,
            n_noise,
            sil,
        )
    return results


def _select_best(
    kmeans_results: dict[int, dict[str, Any]],
    agg_results: dict[int, dict[str, Any]],
) -> tuple[str, int]:
    """Select (algo, k) maximising silhouette, excluding degenerate partitions.

    Interpretability guard: partitions where any cluster has fewer than 25% of
    the expected balanced size (floor 3 stations) are skipped.  If all
    candidates are degenerate — unlikely but possible — falls back to the
    global best silhouette without the guard.
    """
    best_algo, best_k, best_sil = "kmeans", -1, _SENTINEL_SILHOUETTE

    for k, metrics in kmeans_results.items():
        if metrics["interpretable"] and metrics["silhouette"] > best_sil:
            best_sil = metrics["silhouette"]
            best_algo, best_k = "kmeans", k

    for k, metrics in agg_results.items():
        if metrics["interpretable"] and metrics["silhouette"] > best_sil:
            best_sil = metrics["silhouette"]
            best_algo, best_k = "agglomerative", k

    if best_k == -1:
        # All partitions were degenerate — fall back to unconstrained best
        logger.warning("All partitions are degenerate — falling back to best silhouette without interpretability guard.")
        for k, metrics in kmeans_results.items():
            if metrics["silhouette"] > best_sil:
                best_sil = metrics["silhouette"]
                best_algo, best_k = "kmeans", k
        for k, metrics in agg_results.items():
            if metrics["silhouette"] > best_sil:
                best_sil = metrics["silhouette"]
                best_algo, best_k = "agglomerative", k

    logger.info(
        "Best model: %s k=%d (silhouette=%.3f)", best_algo, best_k, best_sil
    )
    return best_algo, best_k


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_clustering() -> None:
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load profiles
    # ------------------------------------------------------------------
    profiles = pd.read_parquet(PROFILES_FILE)
    assert len(profiles) == 67, f"Expected 67 stations, got {len(profiles)}"
    X_raw = profiles[PROFILE_FEATURES].values
    logger.info("Loaded %d station profiles with %d features.", *X_raw.shape)

    # ------------------------------------------------------------------
    # 2. Scale — RobustScaler on full population (no train/test split)
    # ------------------------------------------------------------------
    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw)
    joblib.dump(scaler, Path(ARTIFACTS_DIR) / "scaler.joblib")
    logger.info("Scaler saved.")

    # ------------------------------------------------------------------
    # 3. KMeans grid
    # ------------------------------------------------------------------
    logger.info("--- KMeans grid ---")
    kmeans_results = _fit_kmeans_grid(X, K_RANGE, KMEANS_PARAMS)

    # ------------------------------------------------------------------
    # 4. Agglomerative (Ward) grid + linkage matrix
    # ------------------------------------------------------------------
    logger.info("--- Agglomerative (Ward) grid ---")
    agg_results, linkage_matrix = _fit_agglomerative_grid(
        X, K_RANGE, AGGLOMERATIVE_PARAMS
    )
    np.save(Path(ARTIFACTS_DIR) / "linkage_matrix.npy", linkage_matrix)
    logger.info("Linkage matrix saved.")

    # ------------------------------------------------------------------
    # 5. DBSCAN grid (comparison only — expectations are low at N=67)
    # ------------------------------------------------------------------
    logger.info("--- DBSCAN grid ---")
    dbscan_results = _fit_dbscan_grid(X, DBSCAN_EPS_GRID, DBSCAN_MIN_SAMPLES)

    # ------------------------------------------------------------------
    # 6. Select best (algo, k) and refit final models
    # ------------------------------------------------------------------
    best_algo, best_k = _select_best(kmeans_results, agg_results)

    km_best = KMeans(n_clusters=best_k, **KMEANS_PARAMS)
    km_best.fit(X)
    joblib.dump(km_best, Path(ARTIFACTS_DIR) / "kmeans_best.joblib")

    agg_best = AgglomerativeClustering(
        n_clusters=best_k, **AGGLOMERATIVE_PARAMS
    )
    agg_labels_best = agg_best.fit_predict(X)
    joblib.dump(agg_best, Path(ARTIFACTS_DIR) / "agglomerative_best.joblib")

    if best_algo == "kmeans":
        final_labels = km_best.labels_
    else:
        final_labels = agg_labels_best

    # ------------------------------------------------------------------
    # 7. station_clusters.csv
    # ------------------------------------------------------------------
    clusters_df = pd.DataFrame(
        {
            "idstazione": profiles["idstazione"].values,
            "cluster": final_labels,
        }
    )
    clusters_df.to_csv(CLUSTERS_FILE, index=False)
    logger.info("station_clusters.csv saved — label distribution:\n%s", clusters_df["cluster"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------------
    # 8. clustering_metrics.json
    # ------------------------------------------------------------------
    metrics_payload: dict[str, Any] = {
        "n_stations": int(len(profiles)),
        "best_model": {
            "algo": best_algo,
            "k": best_k,
            "silhouette": (
                kmeans_results[best_k]["silhouette"]
                if best_algo == "kmeans"
                else agg_results[best_k]["silhouette"]
            ),
        },
        "kmeans": {str(k): v for k, v in kmeans_results.items()},
        "agglomerative": {str(k): v for k, v in agg_results.items()},
        "dbscan": {str(eps): v for eps, v in dbscan_results.items()},
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, ensure_ascii=False)
    logger.info("clustering_metrics.json saved.")

    logger.info("Done — all artefacts in %s", ARTIFACTS_DIR)


def main() -> None:
    run_clustering()


if __name__ == "__main__":
    main()
