#!/usr/bin/env python3
"""Plot a unified arm's training: what each task learned, and how the shared head split.

The left panel is the finding the two-head arm structurally cannot show. Both tasks read
one 2,992-way head, and nothing tells the model which half its answer belongs in except the
settled state of the brain. Leakage -- the probability mass landing in the other task's
range -- starts at the uniform value 1024/2992 and has to be driven down by the model
itself, so the null is drawn rather than described.

The right panel is the ordinary training view, kept alongside because a collapse in leakage
would mean little if either task were failing to learn.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANGUAGE, CHESS, NULL = "#2a5d9f", "#a03050", "#777777"
TOKENS, MOVES = 1024, 1968


def smooth(values, window):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, required=True, help="An arm's unified.jsonl")
    parser.add_argument("--title", default="One brain, two tasks, one output space")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.record.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No records in {args.record}")
    updates = np.array([r["updates"] for r in rows], dtype=float)
    window = min(args.window, max(1, len(rows) // 4))
    axis = updates[window - 1:] if len(rows) >= window else updates

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    null = TOKENS / (TOKENS + MOVES)
    left.axhline(null, color=NULL, linestyle="--", linewidth=1.3)
    left.text(axis[-1], null * 1.08, f"uniform null  {null:.3f}", ha="right", fontsize=9.5,
              color=NULL)
    left.plot(axis, smooth([r["chess_leakage"] for r in rows], window), color=CHESS,
              linewidth=2.2, label="board → words")
    left.plot(axis, smooth([r["language_leakage"] for r in rows], window), color=LANGUAGE,
              linewidth=2.2, label="story → moves")
    left.set_yscale("log")
    left.set_xlabel("update")
    left.set_ylabel("probability mass in the other task's range")
    left.set_title("The model partitions a shared output space", fontsize=12.5, fontweight="bold")
    left.legend(frameon=False, fontsize=10, loc="lower left")
    left.grid(alpha=.25, which="both")
    left.set_axisbelow(True)

    right.plot(axis, smooth([r["language_loss"] for r in rows], window), color=LANGUAGE,
               linewidth=2.2, label="language, next token")
    right.plot(axis, smooth([r["chess_policy_loss"] for r in rows], window), color=CHESS,
               linewidth=2.2, label="chess, Stockfish's move")
    right.set_xlabel("update")
    right.set_ylabel("training cross-entropy (nats)")
    right.set_title("Neither task is being starved", fontsize=12.5, fontweight="bold")
    right.legend(frameon=False, fontsize=10)
    right.grid(alpha=.25)
    right.set_axisbelow(True)

    figure.suptitle(args.title, fontsize=14, fontweight="bold", y=.98)
    figure.tight_layout(rect=(0, .055, 1, .95))
    figure.text(.5, .022, f"Curves are {window}-update running means of the training batches. "
                          f"One head over 2,992 outputs: 1,024 byte-level tokens and 1,968 chess "
                          f"moves. Nothing marks which half an answer belongs in.",
                ha="center", fontsize=9, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    print(f"wrote {args.output} from {len(rows):,} updates")


if __name__ == "__main__":
    main()
