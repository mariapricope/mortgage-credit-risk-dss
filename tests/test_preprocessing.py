import numpy as np
import pandas as pd
import pytest

from mortgage_dss import config
from mortgage_dss.data_loader import load_raw_data, split_features_target
from mortgage_dss.preprocessing import build_preprocessor


@pytest.fixture(scope="module")
def X():
    return split_features_target(load_raw_data())[0]


def test_no_missing_after_preprocessing(X):
    Z = build_preprocessor().fit_transform(X)
    assert not np.isnan(Z).any()


def test_output_width(X):
    Z = build_preprocessor().fit_transform(X)
    assert Z.shape == (len(X), len(config.NUMERIC_FEATURES) + 3)


def test_missing_indicators_added(X):
    pre = build_preprocessor(add_missing_indicators=True).fit(X)
    names = pre.get_feature_names_out()
    for field in config.KEY_FIELDS:
        assert f"num__missingindicator_{field}" in names


def test_excluded_fields_not_used():
    for field in ["application_id", "interest_rate_pct", "application_channel", "deposit_gbp",
                  config.TARGET]:
        assert field not in config.FEATURES


def test_unseen_category_does_not_crash(X):
    pre = build_preprocessor().fit(X)
    new = X.head(1).copy()
    new["employment_type"] = "Retired"
    Z = pre.transform(new)
    assert Z[0, -3:].sum() == 0


def test_scaling_fitted_on_training_only(X):
    train, test = X.iloc[:600], X.iloc[600:]
    pre = build_preprocessor().fit(train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_[0], train["annual_income_gbp"].mean())
    assert not np.allclose(scaler.mean_[0], pd.concat([train, test])["annual_income_gbp"].mean())
