#!/usr/bin/env python3
"""Held-out accuracy on every task a unified arm was trained on, from one checkpoint.

The trainer's running log reports accuracy on the batch it just trained on, which is the
number that flatters. `evaluate_unified_quality.py` scores language properly but says
nothing about the other tasks, and the chess and sentiment figures only appear in
`unified-results.json` once the run has finished. So mid-run there is no way to ask the
question a reader actually asks -- how good is it at each task, on data it has not seen.

This answers that for any checkpoint, including `latest.pt` of a run still in progress.
Each task is scored with the evaluator the trainer itself uses at the end, so a number
printed here and a number in the final report mean the same thing.

Every accuracy is reported against its own floor, because they are not comparable to each
other: next-token prediction over 1,024 classes, a move over 1,968 with a legality mask
available, and a binary label whose corpus is not balanced.

Those floors are **recomputed from the split being scored**, never written down here. A
hardcoded floor is a number with no owner: it survives a corpus rebuild, a split change or
a plain mistake, and a wrong one silently rescales every accuracy a reader compares against
it. This module had exactly that bug -- a sentiment "majority class" constant of 72.48%
against a split whose real majority class is 50.92%, which made a 19-point win read as a
narrow miss. Deriving it from the labels costs one pass over the split and cannot drift.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_unified_arms import rebuild
from evaluate_ngxson_quality import aggregate
from evaluate_unified_quality import score

DEFAULT_CHESS_BASELINES = ROOT / "experiments/records/chess-baselines-v1.json"


def majority_class(rows):
    """The accuracy of always answering with the commonest label in this split."""
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    total = sum(counts.values())
    if not total:
        raise SystemExit("Cannot derive a floor from an empty split")
    return {"rows": total, "label_counts": counts, "majority_class_accuracy": max(counts.values()) / total}


def chess_floors(split, path=DEFAULT_CHESS_BASELINES):
    """Read the measured chess baselines rather than restating them."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Missing measured chess baselines: {path}. Pass --chess-baselines.")
    measured = json.loads(path.read_text())
    if measured.get("split") != split:
        raise SystemExit(f"Chess baselines were measured on {measured.get('split')!r}, "
                         f"not the {split!r} split being scored; they are not the floor here")
    return {"commonest_legal_move": measured["most_common_legal_move_top1"],
            "uniform_legal_draw_expected": measured["random_legal_move_top1_expected"],
            "constant_value_mae": measured["constant_value_mae"],
            "source": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--audit-data", type=Path, required=True)
    parser.add_argument("--chess-corpus", type=Path, required=True)
    parser.add_argument("--sentiment-corpus", type=Path)
    parser.add_argument("--split", default="audit", choices=("validation", "audit"))
    parser.add_argument("--chess-limit", type=int, default=10000)
    parser.add_argument("--chess-baselines", type=Path, default=DEFAULT_CHESS_BASELINES,
                        help="Measured chess floors; must have been measured on --split")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import train_unified

    model, unified, plasticity = rebuild(args.config, args.model, args.groups,
                                         args.run, args.checkpoint, args.device)
    saved = json.loads((args.run / "status.json").read_text())
    report = {"run": str(args.run), "checkpoint": args.checkpoint, "split": args.split,
              "updates": saved.get("updates"), "plasticity": plasticity, "tasks": {}}

    rows = json.loads(args.audit_data.read_text())[args.split if args.split != "audit" else "audit"]
    full, language, _ = score(unified, model.brain, rows, args.device, unified.tokens)
    narrow, wide = aggregate(language), aggregate(full)
    report["tasks"]["language"] = {
        "stories": narrow["stories"], "targets": narrow["tokens"],
        "top1_accuracy": narrow["top1_accuracy"],
        "cross_entropy_language_range": narrow["cross_entropy"],
        "cross_entropy_full_output_space": wide["cross_entropy"],
        "leakage_cost_nats": wide["cross_entropy"] - narrow["cross_entropy"],
        "floor": {"classes": unified.tokens, "uniform": 1.0 / unified.tokens}}

    chess = train_unified.ChessBatches(args.chess_corpus, args.device, 32, 42)
    report["tasks"]["chess"] = {
        **train_unified.evaluate_chess(unified, chess, args.split, args.device, args.chess_limit),
        "floor": chess_floors(args.split, args.chess_baselines)}

    if args.sentiment_corpus:
        sentiment = train_unified.SentimentBatches(args.sentiment_corpus, args.device, 32, 42)
        report["tasks"]["sentiment"] = {
            **train_unified.evaluate_sentiment(unified, sentiment, args.split, args.device),
            "floor": majority_class(sentiment.split(args.split))}

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
