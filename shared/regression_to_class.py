"""Utilities for PM10-to-alert-class and hybrid alert decisions."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

ALERT_THRESHOLDS = (20.0, 35.0, 50.0)
CLASS_NAMES = ("verde", "giallo", "arancio", "rosso")


def pm10_to_alert_class(
    pm10_values: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] = ALERT_THRESHOLDS,
) -> np.ndarray:
    """Map PM10 values to ordinal alert classes using fixed thresholds."""
    threshold_arr = np.asarray(thresholds, dtype=float)
    class_values = np.digitize(np.asarray(pm10_values, dtype=float), threshold_arr)
    return np.clip(class_values, 0, len(threshold_arr)).astype(int)


def distance_to_nearest_threshold(
    pm10_values: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] = ALERT_THRESHOLDS,
) -> np.ndarray:
    """Return each PM10 value's absolute distance to the nearest alert threshold."""
    values = np.asarray(pm10_values, dtype=float).reshape(-1)
    threshold_arr = np.asarray(thresholds, dtype=float).reshape(1, -1)
    return np.min(np.abs(values.reshape(-1, 1) - threshold_arr), axis=1)


def order_class_probabilities(
    probabilities: np.ndarray,
    classes: Optional[Sequence[int]] = None,
    *,
    n_classes: int = len(CLASS_NAMES),
) -> np.ndarray:
    """Return probability columns ordered as alert classes 0..n_classes-1."""
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    if classes is None:
        classes = list(range(proba.shape[1]))

    ordered = np.zeros((proba.shape[0], n_classes), dtype=float)
    for source_idx, class_value in enumerate(classes):
        class_idx = int(class_value)
        if 0 <= class_idx < n_classes:
            ordered[:, class_idx] = proba[:, source_idx]
    return ordered


def hybrid_regression_classifier_decision(
    pm10_hat: Sequence[float] | np.ndarray,
    proba_cls: np.ndarray,
    *,
    delta: float,
    p_threshold: float,
    thresholds: Sequence[float] = ALERT_THRESHOLDS,
) -> Dict[str, np.ndarray]:
    """Apply the threshold-zone hybrid rule.

    Outside the threshold zone the regressor-induced class is kept. Inside the
    zone, disagreement is resolved with the prudential rule: choose the more
    severe class only when the classifier probability for that severe class is
    at least ``p_threshold``; otherwise keep the regressor class.
    """
    pm10_arr = np.asarray(pm10_hat, dtype=float).reshape(-1)
    proba_arr = np.asarray(proba_cls, dtype=float)
    if proba_arr.ndim != 2:
        raise ValueError("proba_cls must be a 2D array.")
    if proba_arr.shape[0] != pm10_arr.shape[0]:
        raise ValueError("pm10_hat and proba_cls must have the same row count.")
    if proba_arr.shape[1] < len(thresholds) + 1:
        raise ValueError("proba_cls must contain one column for each alert class.")

    class_reg = pm10_to_alert_class(pm10_arr, thresholds=thresholds)
    class_cls = np.argmax(proba_arr, axis=1).astype(int)
    distance = distance_to_nearest_threshold(pm10_arr, thresholds=thresholds)

    in_threshold_zone = distance <= float(delta)
    discordant = class_cls != class_reg
    class_severe = np.maximum(class_reg, class_cls)
    row_idx = np.arange(pm10_arr.shape[0])
    p_severe = proba_arr[row_idx, class_severe]

    prudential_mask = (
        in_threshold_zone
        & discordant
        & (p_severe >= float(p_threshold))
    )
    class_final = class_reg.copy()
    class_final[prudential_mask] = class_severe[prudential_mask]
    escalation_mask = prudential_mask & (class_final > class_reg)

    return {
        "class_final": class_final.astype(int),
        "class_reg": class_reg.astype(int),
        "class_cls": class_cls.astype(int),
        "distance_to_threshold": distance.astype(float),
        "in_threshold_zone": in_threshold_zone.astype(bool),
        "discordant": discordant.astype(bool),
        "class_severe": class_severe.astype(int),
        "p_severe": p_severe.astype(float),
        "prudential_mask": prudential_mask.astype(bool),
        "escalation_mask": escalation_mask.astype(bool),
    }
