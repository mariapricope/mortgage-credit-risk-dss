import numpy as np
import pytest

from mortgage_dss import config
from mortgage_dss.data_loader import load_raw_data, split_features_target
from mortgage_dss.explain import top_risk_factors
from mortgage_dss.model import V2, split_data, train_final


@pytest.fixture(scope="module")
def fitted():
    X, y = split_features_target(load_raw_data())
    X_train, X_test, y_train, _ = split_data(X, y)
    return train_final("logistic_regression", X_train, y_train, V2), X_test


def test_missing_reason_only_when_value_missing(fitted):
    """Regression test: a complete application must never get an 'X missing' reason."""
    pipe, X_test = fitted
    complete = X_test.dropna()
    for reason in top_risk_factors(pipe, complete):
        assert "missing" not in reason


def test_missing_reason_reported_when_missing(fitted):
    pipe, X_test = fitted
    row = X_test.dropna().head(1).copy()
    row["credit_score"] = np.nan
    assert "credit score missing" in top_risk_factors(pipe, row, n=len(config.FEATURES))[0]


def test_low_credit_score_appears_as_factor(fitted):
    pipe, X_test = fitted
    row = X_test.dropna().head(1).copy()
    row["credit_score"] = 480
    assert top_risk_factors(pipe, row)[0].startswith("credit score")
