"""Render the system flow diagram (docs/design/flow_diagram.png) for the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config

OUT = config.PROJECT_ROOT / "docs" / "design" / "flow_diagram.png"

# (key, x, y, text, colour)
NODES = [
    ("in", 0.5, 0.95, "Mortgage application\n(17 applicant fields)", "#DDE7F3"),
    ("val", 0.5, 0.84, "Validate schema & categories", "#DDE7F3"),
    ("miss", 0.5, 0.72, "R1: >= 2 key fields missing?", "#FBE5C8"),
    ("ins", 0.97, 0.72, "INSUFFICIENT\nobtain missing data", "#F4CCCC"),
    ("pre", 0.5, 0.60, "Preprocess: median impute,\nmissing indicators, scale, one-hot", "#DDE7F3"),
    ("ml", 0.5, 0.48, "Logistic regression\np = P(repayment difficulty)", "#D9EAD3"),
    ("band", 0.5, 0.36, "ML band\nLOW < 0.14 <= MEDIUM < 0.50 <= HIGH", "#D9EAD3"),
    ("rules", 0.5, 0.24, "Policy rules R2-R7\n(each sets a minimum outcome)", "#FBE5C8"),
    ("final", 0.5, 0.12, "Final outcome = most cautious of\nML band and fired rules", "#FBE5C8"),
    ("out", 0.5, 0.01, "Outcome + reasons + top risk factors\n-> human underwriter decides", "#E6D5F0"),
]
EDGES = [("in", "val", ""), ("val", "miss", ""), ("miss", "ins", "yes"), ("miss", "pre", "no"),
         ("pre", "ml", ""), ("ml", "band", ""), ("band", "rules", ""), ("rules", "final", ""),
         ("final", "out", "")]


def draw(path=OUT):
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.05, 1.0)
    ax.axis("off")
    pos = {}
    for key, x, y, text, colour in NODES:
        w, h = (0.25, 0.07) if key == "ins" else (0.52, 0.07)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01",
                                    fc=colour, ec="#444444", lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
        pos[key] = (x, y, w, h)
    for a, b, label in EDGES:
        xa, ya, wa, ha = pos[a]
        xb, yb, wb, hb = pos[b]
        if abs(ya - yb) < 1e-6:
            start, end = (xa + wa / 2, ya), (xb - wb / 2, yb)
        else:
            start, end = (xa, ya - ha / 2), (xb, yb + hb / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#444444"))
        if label:
            dx, dy = (-0.02, 0.012) if start[1] == end[1] else (0.02, 0.0)
            ax.text((start[0] + end[0]) / 2 + dx, (start[1] + end[1]) / 2 + dy, label, fontsize=8,
                    color="#9C2A00")
    ax.set_title("Mortgage credit-risk decision-support flow (v2)", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(draw())
