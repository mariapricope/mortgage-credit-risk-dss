"""Explain individual predictions with the factors that pushed risk up the most"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

READABLE = {
    "annual_income_gbp": "annual income",
    "requested_loan_gbp": "loan size",
    "loan_to_value_pct": "loan-to-value",
    "monthly_debt_commitments_gbp": "existing monthly debt",
    "debt_to_income_ratio": "debt-to-income ratio",
    "credit_score": "credit score",
    "employment_years": "employment length",
    "savings_gbp": "savings",
    "missed_payments_24m": "missed payments (24m)",
    "credit_utilisation_pct": "credit utilisation",
    "previous_default_flag": "previous default",
    "term_years": "mortgage term",
    "employment_type": "employment type",
}


def _readable(name: str) -> str:
    name = name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        return f"{READABLE.get(name[len('missingindicator_'):], name)} missing"
    for field, label in READABLE.items():
        if name.startswith(field + "_"):
            return f"{label} = {name[len(field) + 1:]}"
    return READABLE.get(name, name)


def feature_names(pipeline: Pipeline) -> list:
    return [_readable(n) for n in pipeline.named_steps["preprocess"].get_feature_names_out()]


def global_importance(pipeline: Pipeline) -> pd.Series:
    model = pipeline.named_steps["model"]
    values = model.coef_[0] if hasattr(model, "coef_") else model.feature_importances_
    return pd.Series(values, index=feature_names(pipeline)).sort_values(key=np.abs, ascending=False)


def top_risk_factors(pipeline: Pipeline, X: pd.DataFrame, n: int = 3) -> list:
    """Return, for each row, the n features that increased the risk score most."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        top = ", ".join(global_importance(pipeline).index[:n])
        return [f"(global) {top}"] * len(X)
    Z = pipeline.named_steps["preprocess"].transform(X)
    contrib = Z * model.coef_[0]
    names = np.array(feature_names(pipeline))
    out = []
    for row in contrib:
        idx = [i for i in np.argsort(row)[::-1][:n] if row[i] > 0]
        out.append(", ".join(names[idx]) if idx else "no strong risk-increasing factors")
    return out
