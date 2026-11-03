'''Configuration variables for the mortgage credit risk DSS'''
from pathlib import Path

# ----- Paths -----

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mortgage_credit_risk_dataset.xlsx"
DATA_SHEET = "Mortgage_Data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ----- Schema -----

ID_COLUMN = "application_id"
TARGET = "repayment_difficulty_12m"

EXPECTED_COLUMNS = [
    ID_COLUMN,
    "annual_income_gbp",
    "property_value_gbp",
    "deposit_gbp",
    "requested_loan_gbp",
    "loan_to_value_pct",
    "monthly_debt_commitments_gbp",
    "debt_to_income_ratio",
    "credit_score",
    "employment_type",
    "employment_years",
    "savings_gbp",
    "missed_payments_24m",
    "credit_utilisation_pct",
    "previous_default_flag",
    "term_years",
    "interest_rate_pct",
    "application_channel",
    TARGET,
]

CATEGORY_LEVELS = {
    "employment_type": ["Employed", "Self-employed", "Contractor"],
    "application_channel": ["Broker", "Online", "Branch"],
}

# ----- Features -----

# Excluded fields and the reason:
#   application_id      - identifier
#   deposit_gbp         - deposit = property_value - requested_loan (redundant)
#   property_value_gbp  - information already captured by LTV and loan size
#   interest_rate_pct   - lender-set price that may encode a prior risk
#   application_channel - business-process field
NUMERIC_FEATURES = [
    "annual_income_gbp",
    "requested_loan_gbp",
    "loan_to_value_pct",
    "monthly_debt_commitments_gbp",
    "debt_to_income_ratio",
    "credit_score",
    "employment_years",
    "savings_gbp",
    "missed_payments_24m",
    "credit_utilisation_pct",
    "previous_default_flag",
    "term_years",
]
CATEGORICAL_FEATURES = ["employment_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fields the rule engine treats as essential for a reliable assessment
KEY_FIELDS = ["credit_score", "employment_years", "savings_gbp", "credit_utilisation_pct"]

# ----- Decision logic -----

# Probability of repayment difficulty
RISK_BANDS = {"low_max": 0.20, "high_min": 0.50}

# Policy rules
POLICY = {
    "max_ltv_pct": 90.0,          # LTV above this -> escalate
    "max_dti": 0.45,              # affordability ratio above this -> escalate
    "min_credit_score": 550,      # score below this -> escalate
    "max_missed_payments": 2,     # >= this many missed payments in 24m -> escalate
    "max_missing_key_fields": 2,  # >= this many key fields missing -> insufficient data
}

OUTCOMES = {
    "LOW": "Low risk - proceed with standard underwriter review",
    "MEDIUM": "Medium risk - enhanced review: request further evidence",
    "HIGH": "High risk - refer to senior underwriter",
    "INSUFFICIENT": "Insufficient data - obtain missing information before assessment",
}
