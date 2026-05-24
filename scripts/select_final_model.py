"""Apply the §8 production-ready selection criterion and write
``step_5_classification/artifacts/final_model_selection.json``.

Selection rules (decided before seeing the numbers to avoid cherry-picking):

    1. Hard constraint: severe_error_rate <= 0.025
    2. Hard constraint: recall_rosso >= 0.65
    3. Among strategies passing both: max f1_macro
    4. Tie (|f1_macro| <= 0.005): prefer simpler strategy
       (classifier_pure > regression_to_class > hybrid)
    5. If none pass: production_ready=false, save best_research_candidate
       (best f1_macro among strategies that pass constraint 1, or globally
       if none pass constraint 1 either).

Outputs:

    step_5_classification/artifacts/final_model_selection.json

Usage::

    python -m scripts.select_final_model
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_METRICS_PATH = _ROOT / "step_5_classification" / "artifacts" / "classification_metrics.json"
_OUTPUT_PATH = _ROOT / "step_5_classification" / "artifacts" / "final_model_selection.json"

SEVERE_ERROR_RATE_MAX = 0.025
RECALL_ROSSO_MIN = 0.65
TIE_F1_THRESHOLD = 0.005

# Lower number = simpler (preferred on tie)
_COMPLEXITY_RANK: dict[str, int] = {
    "classifier_logistic_regression": 0,
    "classifier_random_forest": 1,
    "classifier_xgboost": 2,
    "classifier_catboost_clean": 3,
    "classifier_catboost_station": 3,
    "classifier_xgboost_calibrated": 4,
    "regression_to_class_xgboost": 5,
    "hybrid_xgboost_reg_xgboost_cls": 6,
    "xgboost_ordinal": 7,
}


def _complexity(strategy_name: str) -> int:
    return _COMPLEXITY_RANK.get(strategy_name, 99)


def _passes_c1(m: dict) -> bool:
    v = m.get("severe_error_rate")
    return v is not None and v <= SEVERE_ERROR_RATE_MAX


def _passes_c2(m: dict) -> bool:
    v = m.get("recall_rosso")
    return v is not None and v >= RECALL_ROSSO_MIN


def _select_among(candidates: list[dict]) -> dict:
    """Return the best candidate per §8 rules 3–4."""
    best = max(candidates, key=lambda m: m.get("f1_macro") or 0.0)
    best_f1 = best.get("f1_macro") or 0.0
    tied = [
        c for c in candidates
        if abs((c.get("f1_macro") or 0.0) - best_f1) <= TIE_F1_THRESHOLD
    ]
    if len(tied) > 1:
        # Prefer simpler strategy
        tied.sort(key=lambda m: _complexity(m.get("strategy_name", "")))
        return tied[0]
    return best


def _extract_key_metrics(m: dict) -> dict:
    return {
        "strategy_name": m.get("strategy_name"),
        "f1_macro": m.get("f1_macro"),
        "severe_error_rate": m.get("severe_error_rate"),
        "recall_rosso": m.get("recall_rosso"),
        "precision_rosso": m.get("precision_rosso"),
        "over_alert_rate": m.get("over_alert_rate"),
        "rmse_rosso": m.get("rmse_rosso"),
    }


def run_selection(metrics: dict) -> dict:
    models: dict = metrics.get("models", {})
    all_strategies = [
        {**m, "_key": k}
        for k, m in models.items()
        if m.get("f1_macro") is not None
    ]

    c1_passing = [m for m in all_strategies if _passes_c1(m)]
    both_passing = [m for m in c1_passing if _passes_c2(m)]

    constraint_summary: dict[str, dict] = {}
    for m in all_strategies:
        name = m.get("strategy_name", m["_key"])
        constraint_summary[name] = {
            "c1_severe_error_rate": _passes_c1(m),
            "c2_recall_rosso": _passes_c2(m),
            "passes_both": _passes_c1(m) and _passes_c2(m),
            "severe_error_rate": m.get("severe_error_rate"),
            "recall_rosso": m.get("recall_rosso"),
            "f1_macro": m.get("f1_macro"),
        }

    if both_passing:
        winner = _select_among(both_passing)
        result = {
            "production_ready": True,
            "selected_strategy": _extract_key_metrics(winner),
            "selection_reason": (
                f"Unica strategia (o la migliore per f1_macro a parità di ±{TIE_F1_THRESHOLD}) "
                f"che rispetta severe_error_rate ≤ {SEVERE_ERROR_RATE_MAX*100:.1f}% "
                f"e recall_rosso ≥ {RECALL_ROSSO_MIN}."
            ),
            "best_research_candidate": None,
        }
    else:
        # Find best research candidate: prefer C1-passing; if none, take global best
        pool = c1_passing if c1_passing else all_strategies
        best_research = max(pool, key=lambda m: m.get("f1_macro") or 0.0)

        failed_constraints = []
        if not any(_passes_c1(m) for m in all_strategies):
            failed_constraints.append(
                f"severe_error_rate: nessun modello sotto {SEVERE_ERROR_RATE_MAX*100:.1f}%"
            )
        if not any(_passes_c2(m) for m in all_strategies):
            failed_constraints.append(
                f"recall_rosso: nessun modello sopra {RECALL_ROSSO_MIN}"
            )
        if not failed_constraints:
            # Some pass C1, some pass C2, but none pass both
            best_c1_recall = max(
                (m.get("recall_rosso") or 0.0 for m in c1_passing), default=0.0
            )
            failed_constraints.append(
                f"recall_rosso: il miglior valore tra le strategie che passano C1 "
                f"è {best_c1_recall:.4f} < {RECALL_ROSSO_MIN} — "
                f"nessuna strategia supera contemporaneamente entrambi i vincoli"
            )

        result = {
            "production_ready": False,
            "selected_strategy": None,
            "selection_reason": (
                "Nessuna strategia rispetta entrambi i vincoli hard di §8. "
                "Il sistema non è promosso a production-ready. "
                "api/services/predictor.py non viene aggiornato automaticamente."
            ),
            "failed_constraints": failed_constraints,
            "best_research_candidate": _extract_key_metrics(best_research),
            "best_research_candidate_note": (
                "Miglior candidato sperimentale tra le strategie con "
                f"severe_error_rate ≤ {SEVERE_ERROR_RATE_MAX*100:.1f}%. "
                "Non è production-ready: viola il vincolo recall_rosso."
            ),
        }

    result["constraints"] = {
        "c1_severe_error_rate_max": SEVERE_ERROR_RATE_MAX,
        "c2_recall_rosso_min": RECALL_ROSSO_MIN,
        "tie_f1_threshold": TIE_F1_THRESHOLD,
        "c1_description": f"severe_error_rate ≤ {SEVERE_ERROR_RATE_MAX*100:.1f}%",
        "c2_description": f"recall_rosso ≥ {RECALL_ROSSO_MIN}",
    }
    result["constraint_summary_per_strategy"] = constraint_summary
    result["n_strategies_evaluated"] = len(all_strategies)
    result["n_strategies_passing_c1"] = len(c1_passing)
    result["n_strategies_passing_both"] = len(both_passing)
    result["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result["source_artifact"] = str(_METRICS_PATH.relative_to(_ROOT))

    return result


def main() -> None:
    if not _METRICS_PATH.exists():
        print(f"ERROR: artifact not found: {_METRICS_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(_METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    result = run_selection(metrics)

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Written: {_OUTPUT_PATH}")
    print(f"production_ready: {result['production_ready']}")

    if result["production_ready"]:
        s = result["selected_strategy"]
        print(f"Selected strategy: {s['strategy_name']}")
        print(f"  f1_macro={s['f1_macro']:.4f}  severe_error_rate={s['severe_error_rate']:.4f}"
              f"  recall_rosso={s['recall_rosso']:.4f}")
    else:
        print(f"Reason: {result['selection_reason']}")
        if result.get("failed_constraints"):
            for fc in result["failed_constraints"]:
                print(f"  - {fc}")
        rc = result.get("best_research_candidate")
        if rc:
            print(f"Best research candidate: {rc['strategy_name']}")
            print(f"  f1_macro={rc['f1_macro']:.4f}  severe_error_rate={rc['severe_error_rate']:.4f}"
                  f"  recall_rosso={rc['recall_rosso']:.4f}")


if __name__ == "__main__":
    main()
