#!/usr/bin/env python3
"""Plot the readout x corpus-size factorial from the committed record.

The left panel is the finding: the gap between a full and a low-rank readout mostly
disappears once the corpus is ten times larger. The right panel is why that matters
less than it looks — neither large-corpus arm has converged.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FULL, LOW, REF = "#c2571a", "#2a5d9f", "#111111"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "experiments/records/corpus10k-v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/06-corpus-size/figures/corpus-size.png")
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    r = record["results"]
    ce = lambda k: r[k]["audit"]["cross_entropy"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.2, 5.0))
    x = [0, 1]
    labels = ["1,000 stories", "10,000 stories"]

    full = [ce("B32fixed-1k"), ce("J32full-10k")]
    low = [ce("G32rank64-1k"), ce("I32rank64-10k")]
    reference = ce("released-reference") if "released-reference" in r else 3.988231

    left.plot(x, full, "o-", color=FULL, linewidth=2.4, markersize=10, label="full readout (51.3M)")
    left.plot(x, low, "o-", color=LOW, linewidth=2.4, markersize=10, label="rank 64 (3.96M)")
    left.axhline(reference, color=REF, linestyle="--", linewidth=1.2)
    left.annotate("released reference, 52.76M", (0.97, reference), xycoords=("axes fraction", "data"),
                  textcoords="offset points", xytext=(0, 6), fontsize=8.5, color=REF, ha="right")

    # The gap is the whole point; draw it at both ends.
    for i, (a, b) in enumerate(zip(full, low)):
        left.annotate("", xy=(i, a), xytext=(i, b),
                      arrowprops=dict(arrowstyle="<->", color="#888", linewidth=1.2))
        left.annotate(f"{a - b:.3f} nats", (i, (a + b) / 2), textcoords="offset points",
                      xytext=(14 if i == 0 else -14, 0), fontsize=10.5, fontweight="bold",
                      color="#333", ha="left" if i == 0 else "right", va="center")
    for i, value in enumerate(full):
        left.annotate(f"{value:.3f}", (i, value), textcoords="offset points",
                      xytext=(0, 12), ha="center", fontsize=9, color=FULL)
    for i, value in enumerate(low):
        left.annotate(f"{value:.3f}", (i, value), textcoords="offset points",
                      xytext=(0, -18), ha="center", fontsize=9, color=LOW)

    left.set_xticks(x)
    left.set_xticklabels(labels)
    left.set_xlim(-0.35, 1.35)
    left.margins(y=.16)
    left.set_ylabel("audit cross-entropy (nats)")
    left.set_title("The readout gap is mostly a small-data artifact", fontsize=11, loc="left")
    left.legend(frameon=False, fontsize=9, loc="upper right")
    left.grid(True, axis="y", alpha=.25, linewidth=.6)

    passes = [14.05, 1.40]
    bars = right.bar([0, 1], passes, color=["#5a5a66", "#2a5d9f"], width=.5)
    for bar, value in zip(bars, passes):
        right.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                       textcoords="offset points", xytext=(0, 5), ha="center",
                       fontsize=11, fontweight="bold")
    right.set_xticks([0, 1])
    right.set_xticklabels(labels)
    right.set_ylabel("passes over the training corpus")
    right.set_title("…but neither large-corpus arm converged", fontsize=11, loc="left")
    right.set_ylim(0, 21)
    right.grid(True, axis="y", alpha=.25, linewidth=.6)
    right.annotate("same 16,800 updates in every arm", (0.5, 19.2), ha="center", fontsize=9,
                   color="#666")

    figure.suptitle("Readout rank × corpus size — 200 held-out stories, 45,059 targets, "
                    "16,800 updates each", fontsize=11.5, y=.98)
    figure.tight_layout(rect=(0, 0, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    figure.savefig(args.output.with_suffix(".svg"))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
