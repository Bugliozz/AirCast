"""Generate the §7 comparison table for all classification strategies.

Reads ``step_5_classification/artifacts/classification_metrics.json``
and prints a Markdown table with the §8 constraint evaluation for every
strategy found in the artifact.

Usage::

    python -m scripts.compare_classification_strategies
    # or with output to file:
    python -m scripts.compare_classification_strategies > comparison_table.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_METRICS_PATH = _ROOT / "step_5_classification" / "artifacts" / "classification_metrics.json"

SEVERE_ERROR_RATE_MAX = 0.025
RECALL_ROSSO_MIN = 0.65

_PHASE_MAP = {
    "classifier_logistic_regression": "—",
    "classifier_random_forest": "—",
    "classifier_xgboost": "—",
    "classifier_catboost_clean": "—",
    "classifier_catboost_station": "—",
    "classifier_xgboost_calibrated": "2",
    "regression_to_class_xgboost": "1",
    "hybrid_xgboost_reg_xgboost_cls": "2",
    "xgboost_ordinal": "3",
}

_DISPLAY_ORDER = [
    "classifier_logistic_regression",
    "classifier_random_forest",
    "classifier_xgboost",
    "classifier_catboost_clean",
    "classifier_catboost_station",
    "classifier_xgboost_calibrated",
    "regression_to_class_xgboost",
    "hybrid_xgboost_reg_xgboost_cls",
    "xgboost_ordinal",
]


def _constraint_tag(severe: float, recall: float) -> str:
    c1 = severe <= SEVERE_ERROR_RATE_MAX
    c2 = recall >= RECALL_ROSSO_MIN
    if c1 and c2:
        return "PASS"
    tags = []
    if not c1:
        tags.append("C1-FAIL")
    if not c2:
        tags.append("C2-FAIL")
    return " ".join(tags)


def _fmt(value: float | None, pct: bool = False, decimals: int = 4) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:.{decimals}f}"


def build_rows(metrics: dict) -> list[dict]:
    models: dict = metrics.get("models", {})
    rows = []

    order = _DISPLAY_ORDER + sorted(
        k for k in models if k not in _DISPLAY_ORDER
    )
    seen = set()
    for key in order:
        if key in seen or key not in models:
            continue
        seen.add(key)
        m = models[key]
        strategy = m.get("strategy_name", key)
        severe = m.get("severe_error_rate")
        recall = m.get("recall_rosso")
        f1 = m.get("f1_macro")
        rmse_rosso = m.get("rmse_rosso")
        rows.append({
            "key": key,
            "strategy": strategy,
            "phase": _PHASE_MAP.get(strategy, _PHASE_MAP.get(key, "—")),
            "f1_macro": f1,
            "severe_error_rate": severe,
            "recall_rosso": recall,
            "precision_rosso": m.get("precision_rosso"),
            "over_alert_rate": m.get("over_alert_rate"),
            "rmse_rosso": rmse_rosso,
            "constraints": _constraint_tag(severe or 1.0, recall or 0.0),
        })
    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        "| Strategia | Fase | f1_macro | severe_error_rate | recall_rosso |"
        " precision_rosso | over_alert_rate | RMSE_rosso | Vincoli §8 |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|:---:|"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| `{r['strategy']}` | {r['phase']} |"
            f" {_fmt(r['f1_macro'])} |"
            f" {_fmt(r['severe_error_rate'], pct=True)} |"
            f" {_fmt(r['recall_rosso'])} |"
            f" {_fmt(r['precision_rosso'])} |"
            f" {_fmt(r['over_alert_rate'], pct=True)} |"
            f" {_fmt(r['rmse_rosso'], decimals=2)} |"
            f" {r['constraints']} |"
        )


def print_legend() -> None:
    print()
    print(f"Vincoli hard S8: C1 = severe_error_rate <= {SEVERE_ERROR_RATE_MAX*100:.1f}%,"
          f" C2 = recall_rosso >= {RECALL_ROSSO_MIN}")
    print("PASS = entrambi rispettati. C1-FAIL / C2-FAIL = vincolo fallito.")


def main() -> None:
    if not _METRICS_PATH.exists():
        print(f"ERROR: artifact not found: {_METRICS_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(_METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    rows = build_rows(metrics)
    print_table(rows)
    print_legend()

    passing = [r for r in rows if r["constraints"] == "✅"]
    if passing:
        best = max(passing, key=lambda r: r["f1_macro"] or 0.0)
        print(f"\n**Production-ready:** `{best['strategy']}` (f1_macro={best['f1_macro']:.4f})")
    else:
        best_research = max(rows, key=lambda r: r["f1_macro"] or 0.0)
        print(f"\n**Nessuna strategia rispetta entrambi i vincoli §8.**")
        print(f"Best research candidate: `{best_research['strategy']}`"
              f" (f1_macro={best_research['f1_macro']:.4f},"
              f" recall_rosso={best_research['recall_rosso']:.4f})")


if __name__ == "__main__":
    main()
