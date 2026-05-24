import numpy as np

from shared.regression_to_class import (
    hybrid_regression_classifier_decision,
    pm10_to_alert_class,
)


def test_pm10_to_alert_class_uses_fixed_thresholds() -> None:
    values = np.array([19.9, 20.0, 34.9, 35.0, 49.9, 50.0])

    assert pm10_to_alert_class(values).tolist() == [0, 1, 1, 2, 2, 3]


def test_hybrid_rule_escalates_only_near_threshold_with_confidence() -> None:
    pm10_hat = np.array([19.0, 17.0, 42.0])
    proba = np.array(
        [
            [0.20, 0.80, 0.00, 0.00],
            [0.20, 0.80, 0.00, 0.00],
            [0.10, 0.10, 0.20, 0.60],
        ]
    )

    decision = hybrid_regression_classifier_decision(
        pm10_hat,
        proba,
        delta=2.0,
        p_threshold=0.70,
    )

    assert decision["class_reg"].tolist() == [0, 0, 2]
    assert decision["class_final"].tolist() == [1, 0, 2]
    assert decision["escalation_mask"].tolist() == [True, False, False]


def test_hybrid_rule_never_de_escalates() -> None:
    pm10_hat = np.array([51.0])
    proba = np.array([[0.00, 0.00, 0.90, 0.10]])

    decision = hybrid_regression_classifier_decision(
        pm10_hat,
        proba,
        delta=2.0,
        p_threshold=0.70,
    )

    assert decision["class_reg"].tolist() == [3]
    assert decision["class_final"].tolist() == [3]
