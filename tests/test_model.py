import numpy as np
import pytest

from mortgage_dss import config
from mortgage_dss.data_loader import load_raw_data, split_features_target
from mortgage_dss.model import V1, V2, choose_threshold, split_data, train_final


@pytest.fixture(scope="module")
def splits():
    X, y = split_features_target(load_raw_data())
    return split_data(X, y)


def test_split_is_stratified(splits):
    _, _, y_train, y_test = splits
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert len(y_test) == int(800 * config.TEST_SIZE)


def test_no_overlap_between_train_and_test(splits):
    X_train, X_test, _, _ = splits
    assert set(X_train.index).isdisjoint(X_test.index)


@pytest.mark.parametrize("settings", [V1, V2], ids=["v1", "v2"])
def test_probabilities_valid(splits, settings):
    X_train, X_test, y_train, _ = splits
    proba = train_final("logistic_regression", X_train, y_train, settings).predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_beats_chance(splits):
    from sklearn.metrics import roc_auc_score
    X_train, X_test, y_train, y_test = splits
    proba = train_final("logistic_regression", X_train, y_train, V1).predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, proba) > 0.65


def test_lower_credit_score_increases_risk(splits):
    """Sanity check of learned behaviour: everything else equal, a worse credit score means higher risk."""
    X_train, X_test, y_train, _ = splits
    pipe = train_final("logistic_regression", X_train, y_train, V1)
    row = X_test.head(1).copy()
    good, bad = row.copy(), row.copy()
    good["credit_score"], bad["credit_score"] = 800, 550
    assert pipe.predict_proba(bad)[0, 1] > pipe.predict_proba(good)[0, 1]


def test_missing_values_handled_at_prediction(splits):
    X_train, X_test, y_train, _ = splits
    pipe = train_final("logistic_regression", X_train, y_train, V2)
    row = X_test.head(1).copy()
    row[config.KEY_FIELDS] = np.nan
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1


def test_threshold_follows_costs():
    y = np.array([0] * 80 + [1] * 20)
    proba = np.linspace(0, 1, 100)
    equal = choose_threshold(y, proba, fn_cost=1, fp_cost=1)
    fn_heavy = choose_threshold(y, proba, fn_cost=10, fp_cost=1)
    assert fn_heavy <= equal
