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


def bootstrap_comparison(y_true, proba_a, thr_a, proba_b, thr_b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap of the test set: 95% intervals for each version and for the
    difference B - A. With only ~35 positive test cases, point estimates alone
    could overstate an improvement."""
    rng = np.random.default_rng(seed)
    y_true, proba_a, proba_b = map(np.asarray, (y_true, proba_a, proba_b))
    metrics = {
        "roc_auc": lambda y, p, t: roc_auc_score(y, p),
        "pr_auc": lambda y, p, t: average_precision_score(y, p),
        "recall": lambda y, p, t: recall_score(y, (p >= t).astype(int), zero_division=0),
        "precision": lambda y, p, t: precision_score(y, (p >= t).astype(int), zero_division=0),
        "brier": lambda y, p, t: brier_score_loss(y, p),
    }
    samples = {k: {"a": [], "b": [], "diff": []} for k in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        for k, f in metrics.items():
            a, b = f(y_true[idx], proba_a[idx], thr_a), f(y_true[idx], proba_b[idx], thr_b)
            samples[k]["a"].append(a)
            samples[k]["b"].append(b)
            samples[k]["diff"].append(b - a)

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    return {k: {"a_95ci": ci(s["a"]), "b_95ci": ci(s["b"]), "diff_95ci": ci(s["diff"]),
                "share_b_better": round(float(np.mean(np.array(s["diff"]) < 0 if k == "brier"
                                                       else np.array(s["diff"]) > 0)), 3)}
            for k, s in samples.items()}


def plot_version_comparison(table, path) -> None:
    cols = ["roc_auc", "pr_auc", "recall", "precision", "specificity", "f1",
            "system_escalation_recall", "system_review_workload"]
    data = table.set_index("version")[cols].T
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data.plot.bar(ax=ax, color=["#8C8C8C", "#4C72B0"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("v1 baseline vs v2 improved (test set)")
    ax.tick_params(axis="x", rotation=30)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
