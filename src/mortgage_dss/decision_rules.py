"""Rule engine. Combine the ML risk probability with documented policy rules"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd

from . import config

P = config.POLICY


def _is_missing(value) -> bool:
    return value is None or pd.isna(value)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    min_outcome: str
    condition: Callable[[pd.Series], bool]


RULES: List[Rule] = [
    Rule("R2_CREDIT_SCORE_MISSING", "Credit score not available - verify credit file",
         "MEDIUM", lambda a: _is_missing(a["credit_score"])),
    Rule("R3_HIGH_LTV", f"Loan-to-value above {P['max_ltv_pct']:.0f}%",
         "MEDIUM", lambda a: a["loan_to_value_pct"] > P["max_ltv_pct"]),
    Rule("R4_HIGH_DTI", f"Debt-to-income ratio above {P['max_dti']:.2f} (affordability stress)",
         "MEDIUM", lambda a: a["debt_to_income_ratio"] > P["max_dti"]),
    Rule("R5_LOW_CREDIT_SCORE", f"Credit score below {P['min_credit_score']}",
         "HIGH", lambda a: not _is_missing(a["credit_score"]) and a["credit_score"] < P["min_credit_score"]),
    Rule("R6_ADVERSE_HISTORY", "Previous default combined with recent missed payments",
         "HIGH", lambda a: a["previous_default_flag"] == 1 and a["missed_payments_24m"] >= 1),
    Rule("R7_MISSED_PAYMENTS", f"{P['high_missed_payments']} or more missed payments in last 24 months",
         "HIGH", lambda a: a["missed_payments_24m"] >= P["high_missed_payments"]),
]


@dataclass
class Decision:
    outcome: str
    outcome_text: str
    probability: float
    ml_band: Optional[str]
    fired_rules: List[str]
    reasons: List[str]
    missing_key_fields: List[str]

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "outcome_text": self.outcome_text,
            "probability": round(self.probability, 4),
            "ml_band": self.ml_band,
            "fired_rules": "; ".join(self.fired_rules),
            "reasons": "; ".join(self.reasons),
            "missing_key_fields": "; ".join(self.missing_key_fields),
        }


def ml_band(probability: float, medium_from: float, high_from: float) -> str:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be in [0, 1], got {probability}")
    if probability >= high_from:
        return "HIGH"
    if probability >= medium_from:
        return "MEDIUM"
    return "LOW"


def decide(application: pd.Series, probability: float, medium_from: float, high_from: float) -> Decision:
    """Apply the decision process to one application and its ML probability."""
    missing = [f for f in config.KEY_FIELDS if _is_missing(application.get(f))]

    if len(missing) >= P["max_missing_key_fields"]:
        return Decision("INSUFFICIENT", config.OUTCOMES["INSUFFICIENT"], float(probability), None,
                        ["R1_KEY_DATA_MISSING"], [f"Missing key fields: {', '.join(missing)}"], missing)

    band = ml_band(probability, medium_from, high_from)
    outcome = band
    reasons = [f"ML risk score {probability:.2f} -> {band} band"]
    fired = []
    for rule in RULES:
        if rule.condition(application):
            fired.append(rule.rule_id)
            reasons.append(rule.description)
            if config.SEVERITY[rule.min_outcome] > config.SEVERITY[outcome]:
                outcome = rule.min_outcome

    return Decision(outcome, config.OUTCOMES[outcome], float(probability), band, fired, reasons, missing)


def decide_batch(applications: pd.DataFrame, probabilities, medium_from: float, high_from: float) -> pd.DataFrame:
    rows = [decide(row, p, medium_from, high_from).as_dict()
            for (_, row), p in zip(applications.iterrows(), probabilities)]
    return pd.DataFrame(rows, index=applications.index)
