import pandas as pd
import pytest

from mortgage_dss import config
from mortgage_dss.data_loader import DataValidationError, load_raw_data, split_features_target, validate


@pytest.fixture(scope="module")
def df():
    return load_raw_data()


def test_dataset_shape(df):
    assert df.shape == (800, 19)


def test_target_is_binary(df):
    assert set(df[config.TARGET].unique()) == {0, 1}


def test_split_excludes_id_and_target(df):
    X, y = split_features_target(df)
    assert config.ID_COLUMN not in X.columns
    assert config.TARGET not in X.columns
    assert len(X) == len(y)


def test_missing_column_raises(df):
    with pytest.raises(DataValidationError):
        validate(df.drop(columns=["credit_score"]))


def test_duplicate_id_raises(df):
    dup = pd.concat([df, df.head(1)])
    with pytest.raises(DataValidationError):
        validate(dup)


def test_unknown_category_raises(df):
    bad = df.copy()
    bad.loc[0, "employment_type"] = "Unemployed"
    with pytest.raises(DataValidationError):
        validate(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nope.xlsx")
