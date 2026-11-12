"""Command-line entry point for the mortgage credit-risk decision-support prototype.

Usage:
    python main.py                       # run everything: EDA, v1, v2, comparison
    python main.py eda                   # data summary and figures
    python main.py train --version v2    # train and evaluate one version
    python main.py score --input data/sample/new_applications.csv [--version v2]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from mortgage_dss import config, eda  # noqa: E402
from mortgage_dss.model import VERSIONS  # noqa: E402
from mortgage_dss.pipeline import compare_versions, run_version, score_applications  # noqa: E402

DISCLAIMER = ("NOTE: outcomes are decision-support recommendations for a human underwriter. "
              "They are not automated approvals or rejections.")


def cmd_eda(_):
    print(json.dumps(eda.run(), indent=2))
    print(f"Figures saved to {config.FIGURES_DIR}")


def cmd_train(args):
    run_version(VERSIONS[args.version])


def cmd_score(args):
    path = Path(args.input)
    apps = pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)
    settings = VERSIONS[args.version]
    if not (config.MODELS_DIR / f"{settings.name}.joblib").exists():
        print(f"No saved {settings.name} model - training it first...")
        run_version(settings, verbose=False)
    result = score_applications(apps, settings.name)
    out = Path(args.output) if args.output else config.RESULTS_DIR / f"scored_{path.stem}.csv"
    result.to_csv(out, index=False)
    for _, r in result.iterrows():
        print(f"\n{r.get(config.ID_COLUMN, '')}: {r['outcome']} (risk score {r['probability']:.2f})")
        print(f"  {r['outcome_text']}")
        print(f"  Reasons: {r['reasons']}")
        print(f"  Main risk factors: {r['top_risk_factors']}")
    print(f"\n{DISCLAIMER}\nSaved to {out}")


def cmd_all(_):
    cmd_eda(None)
    results = [run_version(s) for s in VERSIONS.values()]
    print("\n=== Version comparison (test set) ===")
    print(compare_versions(results).T.to_string(header=False))
    print(f"\nResults saved to {config.RESULTS_DIR}\n{DISCLAIMER}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("eda", help="data summary and figures").set_defaults(func=cmd_eda)
    p = sub.add_parser("train", help="train and evaluate one version")
    p.add_argument("--version", choices=VERSIONS, default="v2")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("score", help="score new applications from a CSV/XLSX file")
    p.add_argument("--input", required=True)
    p.add_argument("--version", choices=VERSIONS, default="v2")
    p.add_argument("--output")
    p.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    (args.func if args.command else cmd_all)(args)


if __name__ == "__main__":
    main()
