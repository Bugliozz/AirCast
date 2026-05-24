"""Tests for hybrid support-classifier calibration helpers."""

import numpy as np

from step_5_classification.calibrate import (
    CLASSIFIER_CANDIDATES,
    compute_ece_per_class,
    rank_hybrid_classifier_candidates,
    select_hybrid_calibration_candidates,
)


def test_rank_hybrid_candidates_uses_only_admitted_classifiers() -> None:
    metrics_doc = {
        "models": {
            "xgboost": {"f1_macro": 0.64},
            "random_forest": {"f1_macro": 0.63},
            "logistic_regression": {"f1_macro": 0.62},
            "catboost_station": {"f1_macro": 0.99},
        }
    }

    ranking = rank_hybrid_classifier_candidates(metrics_doc)

    assert [item["model_name"] for item in ranking] == list(CLASSIFIER_CANDIDATES)
    assert ranking[0]["model_name"] == "xgboost"


def test_select_hybrid_candidates_adds_runner_up_only_inside_tie_band() -> None:
    close_ranking = [
        {"model_name": "xgboost", "f1_macro": 0.641},
        {"model_name": "random_forest", "f1_macro": 0.637},
        {"model_name": "logistic_regression", "f1_macro": 0.620},
    ]
    far_ranking = [
        {"model_name": "xgboost", "f1_macro": 0.641},
        {"model_name": "random_forest", "f1_macro": 0.630},
    ]

    assert [
        item["model_name"] for item in select_hybrid_calibration_candidates(close_ranking)
    ] == ["xgboost", "random_forest"]
    assert [
        item["model_name"] for item in select_hybrid_calibration_candidates(far_ranking)
    ] == ["xgboost"]


def test_compute_ece_per_class_returns_all_classes() -> None:
    y_true = np.array([0, 1, 2, 3])
    probabilities = np.array(
        [
            [0.90, 0.05, 0.03, 0.02],
            [0.10, 0.75, 0.10, 0.05],
            [0.05, 0.10, 0.80, 0.05],
            [0.02, 0.08, 0.10, 0.80],
        ]
    )

    ece = compute_ece_per_class(y_true, probabilities, n_bins=5)

    assert set(ece) == {"verde", "giallo", "arancio", "rosso"}
    assert all(0.0 <= value <= 1.0 for value in ece.values())
