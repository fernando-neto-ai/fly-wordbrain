#!/usr/bin/env python3
"""Give every arm coordinates on the real axes, so comparability is a query not a memory.

Arm names grew one at a time and each encodes a different subset of what distinguishes it.
`W32threetasksbounded` does not say its rank; `Y32threetasksrank128bounded` does.
`U32unifiedfixed` says neither task count nor rank. Letters ran out and were reused -- `C`
is both `C128bounded` and `C32chessonly`, `T` is both `T32threetasks` and
`T32threetaskspooled`. Renaming published arms is not an option: their directories, run
receipts and model cards point at the existing names.

So the names stay and the registry carries the meaning. Every arm gets its coordinates --
tasks, rank, plasticity, weights, corpus, budget -- read out of its own config rather than
restated here, and `differences` computes which arms differ in exactly one axis. That is
the question that actually matters when reading a contrast, and answering it from memory
is how a four-task arm was built with three-task weights.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments/configs"

# The axes an arm can differ on. Anything not here is not a designed difference.
AXES = ("tasks", "readout_rank", "plasticity", "leak", "graph", "weights", "corpus", "max_updates",
        "d_embed", "history_length", "seed", "sentiment_pooling", "task_cue")

# Config keys that are prose, provenance or measurement rather than a designed knob. Any
# key outside this set and outside AXES is an UNMODELLED difference: the first version of
# this registry had no `graph` axis and duly reported J32fullfixed10k against the shuffled
# -graph control K32rank64shuffled10k as a clean readout_rank contrast. A missing axis does
# not make a difference disappear, it makes it invisible.
NOT_A_KNOB = {
    "schema_version", "configuration_role", "training_host", "device", "branch",
    "experiment_id", "claim_scope", "isolates", "budget_note", "note", "notes",
    "reference_repository", "reference_revision", "reference_weights_sha256",
    "dataset_sha256", "groups_path", "groups_sha256", "connectorch_revision", "trainer",
    "initialization", "optimizer", "checkpoint_selection", "environment", "split_stories",
    "trainable_parameters", "plasticity_detail", "training", "unified", "data_provenance",
    # Prose, status and prerequisites.
    "comparison_baseline", "prerequisites", "test_policy", "initialization_note",
    "execution_status", "architecture", "output_relative",
    # Derived counts. Each follows from an axis -- encoder and readout sizes from d_embed
    # and readout_rank, the edge terms from plasticity -- so they cannot differ
    # independently, and listing them as axes would report the same difference twice.
    "encoder_parameters", "readout_parameters", "readout_architecture", "neuron_gains",
    "edge_gain_parameters", "edge_multiplier_bounds",
    # Read by the `corpus` axis under its own name.
    "dataset_path",
}


def unmodelled(config):
    """Config keys that vary between arms but sit on no axis, so differences hide."""
    return sorted(k for k in config
                  if k not in NOT_A_KNOB and k not in AXES and k != "graph_control")


def coordinates(config):
    training = config.get("training", {})
    unified = config.get("unified", {})
    space = unified.get("output_space", {})
    tasks = unified.get("tasks")
    weights = unified.get("weights", {})
    return {
        "tasks": tuple(tasks) if tasks else ("language",),
        "readout_rank": training.get("readout_rank"),
        "plasticity": training.get("plasticity"),
        "leak": training.get("leak", "fixed"),
        "weights": {k: float(v) for k, v in sorted(weights.items())} or None,
        "corpus": config.get("dataset_path"),
        "max_updates": training.get("max_updates"),
        "d_embed": training.get("d_embed"),
        "history_length": training.get("history_length"),
        "seed": training.get("seed"),
        "sentiment_pooling": unified.get("sentiment_pooling"),
        "task_cue": unified.get("task_cue"),
        # The measured connectome unless an arm declares a control. Stated explicitly so a
        # rewired arm can never read as a clean contrast against a measured-graph one.
        "graph": (config.get("graph_control", {}).get("mode", "measured")
                  if isinstance(config.get("graph_control"), dict) else "measured"),
        "output_width": space.get("total"),
    }


def load():
    arms = {}
    for path in sorted(CONFIGS.glob("*.json")):
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(config, dict) or "training" not in config:
            continue
        arms[path.stem] = coordinates(config)
    return arms


def differences(a, b):
    """Which axes two arms differ on. One axis means the contrast is clean."""
    return sorted(axis for axis in AXES if a.get(axis) != b.get(axis))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/records/arms-registry.json")
    parser.add_argument("--against", help="list every arm's distance from this one")
    args = parser.parse_args()

    arms = load()
    stray = {name: unmodelled(json.loads((CONFIGS / f"{name}.json").read_text()))
             for name in arms}
    stray = {k: v for k, v in stray.items() if v}
    if stray:
        print("UNMODELLED config keys -- add an axis or add them to NOT_A_KNOB:")
        for name, keys in sorted(stray.items()):
            print(f"  {name}: {', '.join(keys)}")
        raise SystemExit(1)
    registry = {
        "note": "Coordinates are read from each arm's config, never restated here. "
                "An arm pair differing on exactly one axis is a clean contrast.",
        "axes": list(AXES),
        "arms": {name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}
                 for name, c in arms.items()},
    }
    clean = {}
    for a in arms:
        for b in arms:
            if a < b:
                d = differences(arms[a], arms[b])
                if len(d) == 1:
                    clean.setdefault(d[0], []).append([a, b])
    registry["clean_contrasts"] = clean
    args.output.write_text(json.dumps(registry, indent=2) + "\n")

    print(f"{len(arms)} arms -> {args.output.relative_to(ROOT)}")
    if args.against:
        if args.against not in arms:
            raise SystemExit(f"Unknown arm {args.against}. Known: {', '.join(sorted(arms))}")
        print(f"\ndistance from {args.against}:")
        for name in sorted(arms):
            if name == args.against:
                continue
            d = differences(arms[args.against], arms[name])
            mark = "  <- clean contrast" if len(d) == 1 else ""
            print(f"  {name:<34} {len(d)} axes: {', '.join(d) or 'identical'}{mark}")
    else:
        print("\nclean single-axis contrasts:")
        for axis, pairs in sorted(clean.items()):
            for a, b in pairs:
                print(f"  {axis:<18} {a}  vs  {b}")


if __name__ == "__main__":
    main()
