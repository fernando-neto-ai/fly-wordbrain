#!/usr/bin/env python3
"""Paired comparison of the graph control against the arm it controls.

The compression-curve evaluator compares every arm to the released reference, which is the
right baseline for "is this model good". It is the wrong baseline for "does the wiring
matter": that question is a contrast between two arms that differ only in their graph.

So this recomputes the same paired whole-story bootstrap — same estimator, same 10,000
resamples, same seed — on the K-minus-I pair, from the per-story records the evaluation
already wrote. Nothing is re-scored, so the numbers cannot drift from the published ones.

The +/-0.10 nat and +/-0.02 accuracy equivalence margins are the ones declared in the
original protocol for the released-reference comparison. They are reused here, not invented
for this result; a two-sided equivalence verdict against them says the shuffled graph is
indistinguishable from the measured one *within those pre-existing margins*, not that the
two are identical.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_ngxson_quality import paired_comparison


def load(directory, label):
    path = directory / (label.replace("/", "_") + ".json")
    if not path.is_file():
        raise SystemExit(f"Missing per-story records for {label}: {path}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Evaluation output directory")
    parser.add_argument("--control", default="K32rank64-10k-shuffled")
    parser.add_argument("--reference-arm", default="I32rank64-10k")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    control, measured = load(args.results, args.control), load(args.results, args.reference_arm)
    rebuild = control["meta"].get("graph_control_rebuild")
    if not rebuild:
        raise SystemExit(f"{args.control} carries no graph-control receipt; it is not a control arm")

    report = {
        "question": "Does the fly's specific wiring contribute anything beyond its degree structure?",
        "control_arm": args.control, "measured_arm": args.reference_arm,
        "direction": f"{args.control} minus {args.reference_arm}; positive means the shuffled graph is worse",
        "graph_control": {"fraction_rewired": rebuild["changed"]["fraction_rewired"],
                          "preserved": rebuild["preserved"]},
        "equivalence_margins": {"cross_entropy_nats": 0.10, "accuracy": 0.02,
                                "provenance": "declared in the original protocol for the "
                                              "released-reference comparison, reused here"},
    }
    for split in ("validation", "audit"):
        comparison = paired_comparison(control[split]["per_story"], measured[split]["per_story"])
        comparison["direction"] = report["direction"]
        report[split] = {
            "control": control[split]["summary"], "measured": measured[split]["summary"],
            "paired": comparison,
        }

    audit = report["audit"]["paired"]
    low, high = audit["ce_95_ci"]
    report["reading"] = (
        f"On the held-out audit population the shuffled graph is "
        f"{audit['delta_cross_entropy']:+.4f} nats [{low:+.4f}, {high:+.4f}] and "
        f"{audit['delta_accuracy'] * 100:+.2f} accuracy points against the measured "
        f"connectome. " + (
            "The interval excludes zero, so the specific wiring is carrying signal."
            if low > 0 or high < 0 else
            "The interval contains zero, so this run does not detect any contribution from "
            "the specific wiring beyond its degree structure."))

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
