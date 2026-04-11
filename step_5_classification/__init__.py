"""Step 5 — Classification module.

Trains and evaluates classification models to predict daily PM10 alert class
(verde/giallo/arancio/rosso) at each monitoring station.

Models:
- Logistic Regression with ElasticNet penalty (linear baseline)
- Random Forest Classifier (ensemble baseline)
- XGBoost Classifier (gradient boosting, non-linear)

Usage::

    from step_5_classification.train import run_training
    from step_5_classification.evaluate import run_evaluation
"""
