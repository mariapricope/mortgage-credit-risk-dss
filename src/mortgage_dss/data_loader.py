"""Load the supplied mortgage dataset and validate its structure."""
from pathlib import Path

import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def load_raw_data(path: Path = config.RAW_DATA_PATH, sheet: str = config.DATA_SHEET) -> pd.DataFrame:
    """Read the mortgage data sheet from the Excel workbook and validate it."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Check columns, identifiers, target values and category levels."""
    missing = set(config.EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")

    if df[config.ID_COLUMN].duplicated().any():
        raise DataValidationError("Duplicate application_id values found")

    if df[config.TARGET].isna().any():
        raise DataValidationError("Target contains missing values")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise DataValidationError("Target must be binary (0/1)")

    for col, levels in config.CATEGORY_LEVELS.items():
        unknown = set(df[col].dropna().unique()) - set(levels)
        if unknown:
            raise DataValidationError(f"Unexpected values in {col}: {sorted(unknown)}")


def split_features_target(df: pd.DataFrame):
    """Return (X, y) using only the configured model features."""
    return df[config.FEATURES].copy(), df[config.TARGET].astype(int).copy()
