"""Step 4 — Regression module.

Trains and evaluates regression models to predict daily PM10 (ug/m3)
at each monitoring station.

Models:
- ElasticNet (linear baseline with log-transformed target)
- XGBoost Regressor (gradient boosting, non-linear)
- Random Forest Regressor (ensemble baseline)

Usage::

    from step_4_regression.train import run_training
    from step_4_regression.evaluate import run_evaluation
"""
