"""End-to-end workflow: load -> split -> compare -> train -> evaluate -> decide -> save."""
import json

import pandas as pd

from . import config
from .data_loader import load_raw_data, split_features_target
from .decision_rules import decide_batch
from .evaluation import classification_metrics, plot_evaluation
from .explain import global_importance, top_risk_factors
from .model import (
    ModelSettings,
    choose_threshold,
    compare_models,
    load_model,
    oof_probabilities,
    save_model,
    split_data,
    train_final,
)

OUTCOME_ORDER = ["LOW", "MEDIUM", "HIGH", "INSUFFICIENT"]


def system_metrics(decisions: pd.DataFrame, y_true: pd.Series) -> dict:
    """Evaluate the full decision process (ML + rules), not just the classifier.

    escalation_recall - share of actual difficulty cases sent to MEDIUM/HIGH/INSUFFICIENT
    review_workload   - share of all applications sent beyond standard review
    rate_by_outcome   - observed difficulty rate per outcome (should rise LOW -> HIGH)
    """
    escalated = decisions["outcome"] != "LOW"
    rate = y_true.groupby(decisions["outcome"]).agg(["mean", "count"])
    return {
        "escalation_recall": float(escalated[y_true == 1].mean()),
        "review_workload": float(escalated.mean()),
        "low_band_difficulty_rate": float(y_true[~escalated].mean()) if (~escalated).any() else None,
        "rate_by_outcome": {k: {"difficulty_rate": round(float(rate.loc[k, "mean"]), 4),
                                "count": int(rate.loc[k, "count"])}
                            for k in OUTCOME_ORDER if k in rate.index},
    }


def run_version(settings: ModelSettings, verbose: bool = True) -> dict:
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    comparison = compare_models(X_train, y_train, settings)
    best = comparison.iloc[0]["model"]

    threshold = settings.threshold
    if settings.tune_threshold:
        oof = oof_probabilities(best, X_train, y_train, settings)
        threshold = choose_threshold(y_train.values, oof, settings.fn_cost, settings.fp_cost)
    medium_from, high_from = threshold, max(settings.high_risk_probability, threshold)

    pipe = train_final(best, X_train, y_train, settings)
    proba = pipe.predict_proba(X_test)[:, 1]
    ml = {k: v if isinstance(v, int) else round(float(v), 4)
          for k, v in classification_metrics(y_test.values, proba, threshold).items()}
    plot_evaluation(y_test.values, proba, threshold, settings.name)

    test_apps = df.loc[X_test.index]
    decisions = decide_batch(test_apps, proba, medium_from, high_from)
    decisions.insert(0, config.ID_COLUMN, test_apps[config.ID_COLUMN])
    decisions["top_risk_factors"] = top_risk_factors(pipe, X_test)
    decisions["actual_outcome"] = y_test
    system = system_metrics(decisions, y_test)

    bundle = {"pipeline": pipe, "settings": settings, "model_name": best, "threshold": threshold,
              "medium_from": medium_from, "high_from": high_from}
    save_model(bundle, settings)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.RESULTS_DIR / f"{settings.name}_model_comparison.csv", index=False)
    decisions.to_csv(config.RESULTS_DIR / f"{settings.name}_test_decisions.csv", index=False)
    global_importance(pipe).round(4).to_csv(config.RESULTS_DIR / f"{settings.name}_feature_importance.csv",
                                            header=["weight"])
    result = {"version": settings.name, "selected_model": best, "threshold": threshold,
              "bands": {"medium_from": medium_from, "high_from": high_from},
              "ml_metrics": ml,
              "system_metrics": system}
    (config.RESULTS_DIR / f"{settings.name}_metrics.json").write_text(json.dumps(result, indent=2))

    if verbose:
        print(f"\n=== {settings.name} ===")
        print("Cross-validated model comparison (training data):")
        print(comparison.to_string(index=False))
        print(f"Selected: {best} | threshold {threshold:.2f} | bands LOW<{medium_from:.2f}<=MEDIUM<{high_from:.2f}<=HIGH")
        print("Test-set ML metrics:", ml)
        print("Decision-process metrics:", json.dumps(system))
    return result


def compare_versions(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        m, s = r["ml_metrics"], r["system_metrics"]
        rows.append({"version": r["version"], "model": r["selected_model"], "threshold": r["threshold"],
                     "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"],
                     "recall": m["recall"], "precision": m["precision"], "specificity": m["specificity"],
                     "f1": m["f1"], "false_negatives": m["fn"], "false_positives": m["fp"],
                     "system_escalation_recall": round(s["escalation_recall"], 4),
                     "system_review_workload": round(s["review_workload"], 4)})
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "version_comparison.csv", index=False)
    return table


def score_applications(applications: pd.DataFrame, version_name: str) -> pd.DataFrame:
    """Score new applications with a saved model and apply the decision rules."""
    bundle = load_model(version_name)
    missing = set(config.FEATURES) - set(applications.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    proba = bundle["pipeline"].predict_proba(applications[config.FEATURES])[:, 1]
    out = decide_batch(applications, proba, bundle["medium_from"], bundle["high_from"])
    out["top_risk_factors"] = top_risk_factors(bundle["pipeline"], applications[config.FEATURES])
    if config.ID_COLUMN in applications:
        out.insert(0, config.ID_COLUMN, applications[config.ID_COLUMN])
    return out
