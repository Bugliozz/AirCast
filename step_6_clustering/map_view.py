"""Folium cluster map for Step 6 — one coloured marker per station.

Builds a static HTML map centred on Lombardia with one ``CircleMarker`` per
station.  Each marker is coloured by cluster assignment and carries a popup
showing station name, cluster label, mean PM10, and province.

Usage::

    python -m step_6_clustering.map_view
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from step_6_clustering.config import (
    ARTIFACTS_DIR,
    CLUSTERS_FILE,
    MAP_FILE,
    META_COLS,
    PROFILE_FEATURES,
    PROFILES_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_LOMBARDY_CENTER = (45.65, 9.75)
_DEFAULT_ZOOM = 8

# Categorical palette — up to 8 clusters (matches K_RANGE 2–8)
_CLUSTER_PALETTE = [
    "#2196F3",  # 0 — blue
    "#FF5722",  # 1 — deep orange
    "#4CAF50",  # 2 — green
    "#9C27B0",  # 3 — purple
    "#FF9800",  # 4 — amber
    "#00BCD4",  # 5 — cyan
    "#F44336",  # 6 — red
    "#8BC34A",  # 7 — light green
]


def _cluster_color(label: int) -> str:
    return _CLUSTER_PALETTE[int(label) % len(_CLUSTER_PALETTE)]


def build_cluster_map(profiles: pd.DataFrame, labels: np.ndarray) -> folium.Map:
    """Return a Folium map with one CircleMarker per station coloured by cluster.

    Parameters
    ----------
    profiles:
        Station profile DataFrame.  Must contain ``nomestazione``, ``provincia``,
        ``lat``, ``lng``, ``idstazione``, and ``pm10_mean``.
    labels:
        1-D integer array of cluster assignments, aligned row-by-row with
        ``profiles``.
    """
    df = profiles.copy()
    df["cluster"] = labels

    fmap = folium.Map(
        location=list(_LOMBARDY_CENTER),
        zoom_start=_DEFAULT_ZOOM,
        tiles="OpenStreetMap",
    )

    unique_clusters = sorted(df["cluster"].unique())
    for cluster_id in unique_clusters:
        color = _cluster_color(cluster_id)
        fg = folium.FeatureGroup(name=f"Cluster {cluster_id}", show=True)
        for _, row in df[df["cluster"] == cluster_id].iterrows():
            popup_html = (
                f"<strong>{row['nomestazione']}</strong>"
                f"<br/>Provincia: {row['provincia']}"
                f"<br/>Cluster: <b>{int(row['cluster'])}</b>"
                f"<br/>PM10 medio: <b>{row['pm10_mean']:.1f}</b> &mu;g/m&sup3;"
            )
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=row["nomestazione"],
            ).add_to(fg)
        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    n_clusters = len(unique_clusters)
    logger.info(
        "Built cluster map: %d stations across %d clusters.", len(df), n_clusters
    )
    return fmap


def run_map() -> None:
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    profiles = pd.read_parquet(PROFILES_FILE)
    clusters = pd.read_csv(CLUSTERS_FILE)
    clusters["idstazione"] = clusters["idstazione"].astype(str)
    profiles["idstazione"] = profiles["idstazione"].astype(str)
    merged = profiles.merge(clusters, on="idstazione")

    assert len(merged) == 67, f"Expected 67 stations after merge, got {len(merged)}"

    labels = merged["cluster"].values
    fmap = build_cluster_map(merged, labels)
    fmap.save(MAP_FILE)
    logger.info("Saved %s", MAP_FILE)


def main() -> None:
    run_map()


if __name__ == "__main__":
    main()
