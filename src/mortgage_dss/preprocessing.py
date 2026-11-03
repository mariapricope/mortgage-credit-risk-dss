"""Preprocessing pipeline: imputation, scaling and categorical encoding.

Everything is wrapped in a scikit-learn ColumnTransformer so the same steps
fitted on the training data are applied unchanged to test data and to new
applications (prevents data leakage from the test set).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(add_missing_indicators: bool = False) -> ColumnTransformer:
    """Create the preprocessing transformer.

    add_missing_indicators: append a 0/1 column per numeric field that had
    missing values during training, so the model can learn whether
    "missing" itself carries risk information.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=add_missing_indicators)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(
            categories=[config.CATEGORY_LEVELS[c] for c in config.CATEGORICAL_FEATURES],
            handle_unknown="ignore",
        )),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])
