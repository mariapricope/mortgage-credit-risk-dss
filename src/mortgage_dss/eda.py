"""Summary tables and figures used in the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config
from .data_loader import load_raw_data


def summarise(df: pd.DataFrame) -> dict:
    """Return the key facts about the dataset as a dictionary."""
    target_rate = df[config.TARGET].mean()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "target_positive": int(df[config.TARGET].sum()),
        "target_rate": round(float(target_rate), 4),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "deposit_identity_max_error": float(
            (df.property_value_gbp - df.requested_loan_gbp - df.deposit_gbp).abs().max()
        ),
    }


def target_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.groupby(column)[config.TARGET].agg(rate="mean", count="count").round(3)


def make_figures(df: pd.DataFrame) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Target balance
    fig, ax = plt.subplots(figsize=(5, 4))
    df[config.TARGET].value_counts().sort_index().plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["0 = no difficulty", "1 = difficulty"], rotation=0)
    ax.set_ylabel("Applications")
    ax.set_title("Target class balance")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_target_balance.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap of numeric fields
    numeric = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Correlation matrix (numeric fields)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    missing.plot.barh(ax=ax, color="#8172B2")
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values by field")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_missing_values.png", dpi=150)
    plt.close(fig)

    # Key numeric distributions by outcome
    cols = ["credit_score", "debt_to_income_ratio", "loan_to_value_pct", "annual_income_gbp"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x=config.TARGET, y=col, ax=ax)
        ax.set_xlabel("repayment difficulty (0/1)")
    fig.suptitle("Key features by observed outcome")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_features_by_outcome.png", dpi=150)
    plt.close(fig)

    # Outcome rate by categorical / count fields
    cats = ["employment_type", "application_channel", "missed_payments_24m", "previous_default_flag"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.ravel(), cats):
        target_rate_by(df, col)["rate"].plot.bar(ax=ax, color="#55A868")
        ax.set_ylabel("Difficulty rate")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_rates_by_category.png", dpi=150)
    plt.close(fig)


def run() -> dict:
    df = load_raw_data()
    summary = summarise(df)
    make_figures(df)
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
