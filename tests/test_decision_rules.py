import math

import pandas as pd
import pytest

from mortgage_dss.decision_rules import decide, ml_band

MEDIUM_FROM, HIGH_FROM = 0.14, 0.50

BASE = {
    "credit_score": 720, "employment_years": 6.0, "savings_gbp": 15000, "credit_utilisation_pct": 30.0,
    "loan_to_value_pct": 80.0, "debt_to_income_ratio": 0.30, "previous_default_flag": 0,
    "missed_payments_24m": 0,
}


def app(**changes) -> pd.Series:
    return pd.Series({**BASE, **changes})


DECISION_TABLE = [
    ("T01 clean, low score", {}, 0.05, "LOW", []),
    ("T02 clean, medium score", {}, 0.30, "MEDIUM", []),
    ("T03 clean, high score", {}, 0.70, "HIGH", []),
    ("T04 boundary = medium_from", {}, 0.14, "MEDIUM", []),
    ("T05 boundary = high_from", {}, 0.50, "HIGH", []),
    ("T06 LTV > 90 escalates LOW", {"loan_to_value_pct": 92.0}, 0.05, "MEDIUM", ["R3_HIGH_LTV"]),
    ("T07 LTV = 90 does not fire", {"loan_to_value_pct": 90.0}, 0.05, "LOW", []),
    ("T08 DTI > 0.50 escalates LOW", {"debt_to_income_ratio": 0.60}, 0.05, "MEDIUM", ["R4_HIGH_DTI"]),
    ("T09 credit score < 550", {"credit_score": 520}, 0.05, "HIGH", ["R5_LOW_CREDIT_SCORE"]),
    ("T10 default + missed payment", {"previous_default_flag": 1, "missed_payments_24m": 1}, 0.05, "HIGH",
     ["R6_ADVERSE_HISTORY"]),
    ("T11 default alone", {"previous_default_flag": 1}, 0.05, "LOW", []),
    ("T12 three missed payments", {"missed_payments_24m": 3}, 0.05, "HIGH", ["R7_MISSED_PAYMENTS"]),
    ("T13 credit score missing", {"credit_score": math.nan}, 0.05, "MEDIUM", ["R2_CREDIT_SCORE_MISSING"]),
    ("T14 two key fields missing", {"credit_score": math.nan, "savings_gbp": math.nan}, 0.05, "INSUFFICIENT",
     ["R1_KEY_DATA_MISSING"]),
    ("T15 rule never lowers ML band", {"loan_to_value_pct": 92.0}, 0.80, "HIGH", ["R3_HIGH_LTV"]),
    ("T16 several rules, most cautious wins", {"loan_to_value_pct": 92.0, "credit_score": 500}, 0.05, "HIGH",
     ["R3_HIGH_LTV", "R5_LOW_CREDIT_SCORE"]),
]


@pytest.mark.parametrize("case, changes, proba, expected, rules", DECISION_TABLE, ids=[c[0] for c in DECISION_TABLE])
def test_decision_table(case, changes, proba, expected, rules):
    d = decide(app(**changes), proba, MEDIUM_FROM, HIGH_FROM)
    assert d.outcome == expected
    assert d.fired_rules == rules


def test_reasons_always_given():
    d = decide(app(loan_to_value_pct=95.0), 0.30, MEDIUM_FROM, HIGH_FROM)
    assert d.reasons[0].startswith("ML risk score 0.30")
    assert any("Loan-to-value" in r for r in d.reasons)


def test_insufficient_lists_missing_fields():
    d = decide(app(credit_score=None, employment_years=None), 0.10, MEDIUM_FROM, HIGH_FROM)
    assert set(d.missing_key_fields) == {"credit_score", "employment_years"}
    assert d.ml_band is None


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError):
        ml_band(bad, MEDIUM_FROM, HIGH_FROM)


def test_outcomes_are_not_approve_or_reject():
    for _, changes, proba, _, _ in DECISION_TABLE:
        text = decide(app(**changes), proba, MEDIUM_FROM, HIGH_FROM).outcome_text.lower()
        assert "approve" not in text and "reject" not in text
