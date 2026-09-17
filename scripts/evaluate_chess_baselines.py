#!/usr/bin/env python3
"""Score the trivial chess baselines a trained model has to beat.

Two numbers matter before any training happens, because they bound what a result means:

    random legal move        what guessing scores
    most common legal move   what a network that has learned nothing about the position
                             scores, by leaning entirely on the move prior

The ChessFly reference reports 3% and 13% on these. Reproducing them is how we check that
our corpus, move vocabulary, mirroring and legality masks agree with the reference before
comparing any trained model against it.

The value baseline is the constant predictor: whatever mean absolute error you get by
ignoring the board entirely and always saying "the side to move is doing about average".
"""
import argparse
import json
from pathlib import Path

import numpy as np


def legality(arrays, split):
    offsets = arrays[f"legal_offsets_{split}"]
    indices = arrays[f"legal_indices_{split}"].astype(np.int64)
    return offsets, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", default="audit", choices=("validation", "test", "audit"))
    parser.add_argument("--vocabulary", type=int, default=1968)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arrays = np.load(args.corpus / "corpus.npz")
    train = arrays["span_train"]
    span = arrays[f"span_{args.split}"]
    moves, values = arrays["moves"].astype(np.int64), arrays["values"]
    targets = moves[span[0]:span[1]]
    offsets, indices = legality(arrays, args.split)
    count = offsets.size - 1
    if count != targets.size:
        raise SystemExit(f"{count} legality rows for {targets.size} positions")

    # The move prior comes from the training split only; using the scored split would
    # leak the answer into the baseline it is supposed to bound.
    prior = np.bincount(moves[train[0]:train[1]], minlength=args.vocabulary).astype(np.float64)

    rng = np.random.default_rng(args.seed)
    random_hits = common_hits = 0.0
    legal_counts = np.empty(count, dtype=np.int64)
    target_always_legal = True
    for position in range(count):
        legal = indices[offsets[position]:offsets[position + 1]]
        legal_counts[position] = legal.size
        if legal.size == 0:
            continue
        target = targets[position]
        target_always_legal &= bool((legal == target).any())
        # Expected accuracy of a uniform draw, rather than one sampled draw.
        random_hits += float((legal == target).sum()) / legal.size
        scores = prior[legal]
        best = scores.max()
        tied = legal[scores == best]
        common_hits += float((tied == target).sum()) / tied.size

    predicted = np.full(targets.size, float(values[train[0]:train[1]].mean()))
    truth = values[span[0]:span[1]]
    # Positions the engine scored as a forced mate are near-terminal: a fifth of them have
    # five or fewer legal moves, which lifts the expected accuracy of guessing sharply.
    # Report them apart so the headline baseline is not quietly driven by them.
    forced = (truth == 0.0) | (truth == 1.0)

    report = {
        "corpus": str(args.corpus), "split": args.split, "positions": int(count),
        "mean_legal_moves": float(legal_counts.mean()),
        "target_move_always_legal": bool(target_always_legal),
        "random_legal_move_top1_expected": random_hits / count,
        "random_legal_move_top1_one_over_mean_legal": float(1.0 / legal_counts.mean()),
        "random_legal_move_top1_expected_excluding_forced_mates":
            float((1.0 / legal_counts[~forced]).mean()) if (~forced).any() else None,
        "random_legal_move_top1_one_over_mean_legal_excluding_forced_mates":
            float(1.0 / legal_counts[~forced].mean()) if (~forced).any() else None,
        "forced_mate_fraction": float(forced.mean()),
        "most_common_legal_move_top1": common_hits / count,
        "constant_value_mae": float(np.abs(predicted - truth).mean()),
        "constant_value_prediction": float(predicted[0]),
        "reference_chessfly": {"random_legal_move_top1": 0.03,
                               "most_common_legal_move_top1": 0.13,
                               "trained_top1": 0.304, "trained_value_mae": 0.081},
        "note": "The prior is built from the training split only. Ties on the prior are "
                "scored as a uniform draw among the tied moves rather than by array order. "
                "Two random-move figures are reported because they answer different "
                "questions: the expected accuracy of a uniform draw is E[1/n], while the "
                "familiar 'one in about thirty legal moves' is 1/E[n]. E[1/n] is the larger "
                "of the two and is the one a trained model actually has to beat. The "
                "ChessFly reference's 3% corresponds to 1/E[n] excluding forced mates.",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
