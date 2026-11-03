"""Train, compare, select and persist the credit-risk classification model."""
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor


@dataclass
class ModelSettings:
    """Settings that distinguish one system version from another."""
    name: str
    class_weight: object = None          # None or balanced
    add_missing_indicators: bool = False
    threshold: float = 0.5               # probability cut-off for the 0/1 class
    tune_threshold: bool = False         # choose threshold by misclassification cost
    fn_cost: float = 1.0                 # cost of missing a difficulty case
    fp_cost: float = 1.0                 # cost of flagging a good applicant
    high_risk_probability: float = 0.70  # probability at or above which the ML band is HIGH


# v1: plain baseline with default 0.5 cut-off.
# v2: improvements applied in section D of the report - missing-value indicators, threshold tuning, and cost-sensitive evaluation.
V1 = ModelSettings(name="v1_baseline")
V2 = ModelSettings(
    name="v2_improved",
    add_missing_indicators=True,
    tune_threshold=True,
    fn_cost=5.0,
    fp_cost=1.0,
    high_risk_probability=0.50,
)
VERSIONS = {"v1": V1, "v2": V2}


def candidate_models(settings: ModelSettings) -> dict:
    cw = settings.class_weight
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight=cw,
                                                  random_state=config.RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                                class_weight=cw, random_state=config.RANDOM_STATE),
        # GradientBoosting has no class_weight; balanced sample weights are passed at fit time.
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                        random_state=config.RANDOM_STATE),
    }


def make_pipeline(estimator, settings: ModelSettings) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(settings.add_missing_indicators)),
        ("model", estimator),
    ])


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified hold-out split so both sets keep the ~22% difficulty rate."""
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


def _fit_params(name: str, y: pd.Series, settings: ModelSettings) -> dict:
    if name == "gradient_boosting" and settings.class_weight == "balanced":
        weights = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
        return {"model__sample_weight": weights}
    return {}


def compare_models(X_train, y_train, settings: ModelSettings) -> pd.DataFrame:
    """Cross-validate each candidate on the training data only."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, est in candidate_models(settings).items():
        scores = cross_validate(
            make_pipeline(est, settings), X_train, y_train, cv=cv,
            scoring=["roc_auc", "average_precision", "recall", "precision", "f1"],
            params=_fit_params(name, y_train, settings),
        )
        rows.append({
            "model": name,
            "cv_roc_auc": scores["test_roc_auc"].mean(),
            "cv_roc_auc_std": scores["test_roc_auc"].std(),
            "cv_pr_auc": scores["test_average_precision"].mean(),
            "cv_recall@0.5": scores["test_recall"].mean(),
            "cv_precision@0.5": scores["test_precision"].mean(),
            "cv_f1@0.5": scores["test_f1"].mean(),
        })
    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).round(4)


def choose_threshold(y_true, proba, fn_cost: float, fp_cost: float) -> float:
    """Pick the probability cut-off that minimises total misclassification cost."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [fn_cost * ((proba < t) & (y_true == 1)).sum() + fp_cost * ((proba >= t) & (y_true == 0)).sum()
             for t in grid]
    return float(grid[int(np.argmin(costs))])


def oof_probabilities(name: str, X_train, y_train, settings: ModelSettings) -> np.ndarray:
    """Out-of-fold probabilities on the training set (used for threshold tuning,
    so the test set is never used to choose the threshold)."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    proba = np.zeros(len(y_train))
    for tr, va in cv.split(X_train, y_train):
        pipe = make_pipeline(candidate_models(settings)[name], settings)
        pipe.fit(X_train.iloc[tr], y_train.iloc[tr], **_fit_params(name, y_train.iloc[tr], settings))
        proba[va] = pipe.predict_proba(X_train.iloc[va])[:, 1]
    return proba


def train_final(name: str, X_train, y_train, settings: ModelSettings) -> Pipeline:
    pipe = make_pipeline(candidate_models(settings)[name], settings)
    pipe.fit(X_train, y_train, **_fit_params(name, y_train, settings))
    return pipe


def save_model(bundle: dict, settings: ModelSettings):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{settings.name}.joblib"
    joblib.dump(bundle, path)
    return path


def load_model(name: str) -> dict:
    return joblib.load(config.MODELS_DIR / f"{name}.joblib")
