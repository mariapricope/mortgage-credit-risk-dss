"""Model evaluation: multiple metrics, confusion matrix, ROC / PR / calibration plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def classification_metrics(y_true, proba, threshold: float) -> dict:
    """Threshold-dependent and threshold-independent metrics"""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def plot_evaluation(y_true, proba, threshold: float, label: str) -> None:
    """Save confusion matrix, ROC, PR and calibration plots for one model version"""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = (proba >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["No difficulty", "Difficulty"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {label} (t={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{label}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=axes[0], name=label)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=axes[1], name=label)
    axes[1].axhline(np.mean(y_true), color="k", ls="--", lw=1, label="base rate")
    axes[1].set_title("Precision-recall curve")
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", label=label)
    axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed difficulty rate")
    axes[2].set_title("Calibration")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{label}_curves.png", dpi=150)
    plt.close(fig)
