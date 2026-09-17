#!/usr/bin/env python3
"""Plot the randomised-graph control from the committed record.

The left panel is the finding at the scale the claim lives at: how much of this model's
advantage over the released reference survives when the connectome is rewired at random.
The right panel is the same contrast magnified, with its confidence interval, because the
effect is real and far too small to see on the left.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MEASURED, SHUFFLED, REF = "#2a5d9f", "#a03050", "#111111"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path,
                        default=ROOT / "experiments/records/graph-control-v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/07-graph-control/figures/graph-control.png")
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    scores = record["scores"]
    ce = lambda k: scores[k]["audit"]["cross_entropy"]
    reference, shuffled, measured = (ce("released-reference"), ce("K32rank64-10k-shuffled"),
                                     ce("I32rank64-10k"))

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.2, 5.0),
                                         gridspec_kw={"width_ratios": [1.15, 1]})

    bars = [("released\nreference\n52.76M", reference, REF, None),
            ("randomly\nrewired graph\n3.96M", shuffled, SHUFFLED, shuffled - reference),
            ("measured\nconnectome\n3.96M", measured, MEASURED, measured - reference)]
    for index, (label, value, colour, delta) in enumerate(bars):
        left.bar(index, value, width=.6, color=colour, alpha=.9)
        left.text(index, value + .07, f"{value:.4f}", ha="center", fontsize=11.5,
                  fontweight="bold")
        if delta is not None:
            left.text(index, value / 2, f"{delta:+.3f}\nnats", ha="center", va="center",
                      fontsize=11, color="white", fontweight="bold")
    left.axhline(reference, color=REF, linestyle="--", linewidth=1.2, alpha=.55)
    left.set_xticks(range(len(bars)))
    left.set_xticklabels([b[0] for b in bars], fontsize=10)
    left.set_ylabel("audit cross-entropy (nats), lower is better")
    left.set_ylim(0, reference * 1.20)
    left.set_title("Rewiring the fly at random costs almost nothing", fontsize=12.5,
                   fontweight="bold")
    left.grid(axis="y", alpha=.25)
    left.set_axisbelow(True)

    audit = record["graph_contrast"]["audit"]
    delta, (low, high) = audit["delta_cross_entropy"], audit["ce_95_ci"]
    right.axvline(0, color="#999999", linestyle="--", linewidth=1.2)
    right.errorbar([delta], [0], xerr=[[delta - low], [high - delta]], fmt="o", color=SHUFFLED,
                   markersize=12, capsize=7, linewidth=2.4, capthick=2.4)
    right.text(delta, .22, f"+{delta:.4f} nats\n[{low:+.4f}, {high:+.4f}]", ha="center",
               fontsize=11, fontweight="bold", color=SHUFFLED)
    right.text(0, -.30, "no effect", ha="center", fontsize=10, color="#777777")
    right.set_xlim(-.022, .116)
    right.set_ylim(-.55, .55)
    right.set_yticks([])
    right.axvspan(-.10, .10, color="#2a5d9f", alpha=.06)
    right.axvline(.10, color="#2a5d9f", linestyle=":", linewidth=1.5)
    right.text(.097, -.44, "declared ±0.10 nat\nequivalence margin", ha="right", fontsize=9.5,
               color="#2a5d9f", va="center")
    right.set_xlabel("rewired minus measured, audit cross-entropy (nats)")
    right.set_title("The same contrast, magnified ~100×", fontsize=12.5, fontweight="bold")
    right.grid(axis="x", alpha=.25)
    right.set_axisbelow(True)

    figure.suptitle("Stage 7 — the fly's wiring is worth 1% of the model's advantage",
                    fontsize=14, fontweight="bold", y=.98)
    figure.tight_layout(rect=(0, .075, 1, .95))
    figure.text(.5, .050, "200 held-out stories, 45,059 next-token targets; paired whole-story "
                          "bootstrap, 10,000 resamples, seed 1729.",
                ha="center", fontsize=9, color="#555555")
    figure.text(.5, .018, "Both arms: rank 64, 10,000 stories, 16,800 updates, seed 42. The graphs "
                          "differ only in which neuron connects to which — every degree, weight "
                          "and interface is held fixed.",
                ha="center", fontsize=9, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
