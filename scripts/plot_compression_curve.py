#!/usr/bin/env python3
"""Plot held-out quality against trainable parameter count for every scored arm.

Reads the committed shared-population record so the figure can never drift from
the numbers in the report. The released reference is drawn as a horizontal line
because it is the comparison baseline, not a point on our compression sweep.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "released-reference"
FULL, LOW = "#c2571a", "#2a5d9f"
# Label offsets are hand-placed per panel because the arms cluster tightly at both ends.
# (dx points, dy points, horizontal alignment)
SWEEP = [
    ("A128fixed/min-ce", "A  width 128, full readout", FULL, "o",
     {"cross_entropy": (-11, 5, "right"), "top1_accuracy": (-11, -2, "right")}),
    ("B32fixed/min-ce", "B  width 32, full readout", FULL, "o",
     {"cross_entropy": (-11, -10, "right"), "top1_accuracy": (-11, 6, "right")}),
    ("reconstruction/min-ce", "our reconstruction, full readout", "#8a8a8a", "s",
     {"cross_entropy": (-11, -4, "right"), "top1_accuracy": (-11, -10, "right")}),
    ("E32rank128fixed/min-ce", "E  rank 128", LOW, "o",
     {"cross_entropy": (10, 2, "left"), "top1_accuracy": (10, -3, "left")}),
    ("G32rank64fixed/min-ce", "G  rank 64", LOW, "o",
     {"cross_entropy": (2, -14, "center"), "top1_accuracy": (10, 3, "left")}),
    ("H32rank32fixed/min-ce", "H  rank 32", LOW, "o",
     {"cross_entropy": (10, 4, "left"), "top1_accuracy": (10, 3, "left")}),
]


def millions(value, _pos):
    return f"{value / 1e6:g}M" if value >= 1e6 else f"{value / 1e6:.1f}M"


def panel(axes, rows, baseline, key, label, better):
    for name, text, color, marker, offsets in SWEEP:
        row = rows[name]
        x, y = row["parameters"], row["audit"][key]
        axes.scatter(x, y, s=80, color=color, marker=marker, zorder=4,
                     edgecolor="white", linewidth=1.2)
        dx, dy, ha = offsets[key]
        axes.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                      fontsize=8.5, color=color, ha=ha, zorder=5)
    line = baseline["audit"][key]
    axes.axhline(line, color="#111111", linestyle="--", linewidth=1.2, zorder=2)
    axes.annotate("released reference", (.035, line), xycoords=("axes fraction", "data"),
                  textcoords="offset points", xytext=(0, 5), fontsize=8.5, color="#111111")
    axes.set_xscale("log")
    axes.set_xlim(1.4e6, 1.5e8)
    axes.xaxis.set_major_formatter(FuncFormatter(millions))
    axes.set_xticks([2e6, 5e6, 1e7, 2e7, 5e7, 1e8])
    axes.set_xlabel("trainable parameters (log scale)")
    axes.set_ylabel(label)
    axes.set_title(better, fontsize=9.5, loc="left", color="#555555")
    axes.grid(True, alpha=.25, linewidth=.6)
    axes.margins(y=.20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/records/compression-curve-v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/05-shared-population/figures/compression-curve.png")
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    rows = {r["label"]: r for r in record["summary"]}
    baseline = rows[BASELINE]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    panel(left, rows, baseline, "cross_entropy", "audit cross-entropy (nats)", "lower is better")
    panel(right, rows, baseline, "top1_accuracy", "audit top-1 accuracy", "higher is better")
    right.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v * 100:.0f}%"))
    figure.legend(handles=[
        Line2D([], [], color=FULL, marker="o", linestyle="", markersize=8,
               label="full 50,578,432-parameter readout"),
        Line2D([], [], color=LOW, marker="o", linestyle="", markersize=8,
               label="factorized low-rank readout"),
        Line2D([], [], color="#111111", linestyle="--", label=f"released reference "
               f"({baseline['parameters']:,} parameters)")],
        loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, -.01))
    stories = record["protocol"]["populations"]["audit"]["stories"]
    figure.suptitle(f"Held-out quality versus model size — {stories} stories, "
                    f"{baseline['audit']['tokens']:,} next-token targets, identical connectome",
                    fontsize=12, y=.975)
    figure.tight_layout(rect=(0, .055, 1, .945))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    figure.savefig(args.output.with_suffix(".svg"))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
