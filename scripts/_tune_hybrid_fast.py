"""Run tune_hybrid grid search on existing OOF file (skips slow OOF rebuild)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from pathlib import Path
from step_5_classification.tune_hybrid import tune_from_oof, PARAMS_FILENAME
from step_5_classification.config import (
    ARTIFACTS_DIR, HYBRID_ALPHA, HYBRID_BETA, HYBRID_DELTA_GRID, HYBRID_PROB_GRID,
)
from step_5_classification.guardrails import write_json

OOF_PATH = Path(ARTIFACTS_DIR) / "hybrid_oof_predictions.parquet"
PARAMS_PATH = Path(ARTIFACTS_DIR) / PARAMS_FILENAME

print(f"Loading OOF from {OOF_PATH} ...", flush=True)
oof = pd.read_parquet(OOF_PATH)
print(f"OOF shape: {oof.shape}", flush=True)

print("Running grid search ...", flush=True)
best, score_curve = tune_from_oof(
    oof,
    delta_grid=[float(v) for v in HYBRID_DELTA_GRID],
    prob_grid=[float(v) for v in HYBRID_PROB_GRID],
    alpha=float(HYBRID_ALPHA),
    beta=float(HYBRID_BETA),
)
print(f"Best: delta={best['delta']}, p_threshold={best['p_threshold']}, score={best['score']:.4f}", flush=True)
print(f"OOF metrics: f1_macro={best['f1_macro']:.4f}, recall_rosso={best['recall_rosso']:.4f}, "
      f"severe_error_rate={best['severe_error_rate']:.4f}", flush=True)

params_doc = {
    "strategy_name": "hybrid_xgboost_reg_xgboost_cls",
    "regressor_name": "xgboost",
    "classifier_name": "xgboost",
    "alert_thresholds": [25.0, 50.0, 75.0],
    "best_delta": float(best["delta"]),
    "best_p_threshold": float(best["p_threshold"]),
    "best_oof_score": float(best["score"]),
    "best_oof_metrics": {
        k: best[k]
        for k in (
            "f1_macro", "precision_rosso", "recall_rosso",
            "severe_error_rate", "over_alert_rate", "under_alert_rate",
            "hybrid_override_rate", "hybrid_escalation_rate",
            "n_overrides", "n_escalations",
        )
    },
    "objective": {
        "formula": "f1_macro - alpha * severe_error_rate - beta * (1 - recall_rosso)",
        "alpha": float(HYBRID_ALPHA),
        "beta": float(HYBRID_BETA),
    },
    "score_curve": score_curve,
    "grid_is_flat": False,
    "oof_predictions_path": OOF_PATH.as_posix(),
}

write_json(PARAMS_PATH, params_doc)
print(f"Saved -> {PARAMS_PATH}", flush=True)
