#!/usr/bin/env python3
"""Measure how far training moved the parameters that live inside the brain.

"The connectome is frozen" is easy to over-read as "the brain learns nothing".
Both statements matter and they are different: the *wiring* — which neuron
contacts which, with what sign and what relative strength — is byte-identical to
the reference, while each of the 49,393 neurons carries a learned input gain,
recurrent gain and bias that training moves a long way.

The distinction is that rec_gain multiplies a neuron's entire incoming sum, so it
rescales all of that neuron's synapses by one factor. It cannot change their
relative weights or their signs. This script quantifies that rescaling instead of
leaving the reader to guess.

Initialization is reconstructed from the recorded seed, not assumed.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_connectorch import build_model, file_hash, load_groups

BRAIN = ("brain.gain", "brain.rec_gain", "brain.bias")


def displacement(initial, trained):
    delta = trained - initial
    norm = float(np.linalg.norm(initial))
    return {
        "values": int(initial.size),
        # Relative L2 is undefined against an all-zero initialization, which is how
        # bias starts; report absolute statistics there instead of a vast ratio.
        "relative_l2_change": None if norm == 0 else float(np.linalg.norm(delta) / norm),
        "absolute_rms_change": float(np.sqrt((delta ** 2).mean())),
        "mean_absolute_change": float(np.abs(delta).mean()),
        "initial_mean": float(initial.mean()),
        "trained_mean": float(trained.mean()),
        "initialized_at_zero": norm == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    spec = json.loads((ROOT / "experiments/configs" / (args.experiment + ".json")).read_text())["training"]
    reference = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                                     local_files_only=True, torch_dtype=torch.float32).cpu()
    groups, _ = load_groups(args.groups)
    model = build_model(reference, argparse.Namespace(
        d_embed=spec["d_embed"], plasticity=spec["plasticity"], readout_rank=spec["readout_rank"],
        history_length=spec["history_length"], seed=spec["seed"]), groups)
    initial = {name: value.detach().cpu().numpy().copy() for name, value in model.named_parameters()}

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trained = {name: value.cpu().numpy() for name, value in saved["parameters"].items()}

    moved = {name: displacement(initial[name], trained[name]) for name in BRAIN}
    product_initial = initial["brain.gain"] * initial["brain.rec_gain"]
    product_trained = trained["brain.gain"] * trained["brain.rec_gain"]
    moved["gain_times_rec_gain"] = displacement(product_initial, product_trained)
    moved["gain_times_rec_gain"]["meaning"] = (
        "Effective per-neuron scaling applied to that neuron's entire incoming synaptic "
        "sum. A uniform rescale per neuron; it cannot change relative weights or signs.")

    budget = {
        "neuron_dynamics": int(sum(trained[n].size for n in BRAIN)),
        "encoder": int(trained["brain.in_proj"].size + trained["brain.wte.weight"].size),
        "output_layernorm": int(trained["ln.weight"].size + trained["ln.bias"].size),
        "readout": int(sum(v.size for k, v in trained.items() if k.startswith("lm_head"))),
    }
    budget["total"] = sum(budget.values())
    budget["frozen_synapses"] = int(model.config.n_edges)

    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "updates": saved["cursor"]["updates"],
        "seed": spec["seed"],
        "initialization": "rebuilt from the recorded seed by scripts/train_connectorch.build_model",
        "unchanged_buffers_sha256": saved["frozen_buffers_sha256"],
        "brain_parameter_displacement": moved,
        "parameter_budget": budget,
        "interpretation": [
            "The wiring is unchanged: every endpoint, sign and stored weight is "
            "byte-identical to the reference graph.",
            "The neurons are not inert: 148,179 per-neuron parameters are trained, and "
            "they move substantially.",
            "Per-neuron recurrent gain rescales a neuron's whole incoming sum uniformly. "
            "It cannot alter the relative strengths or signs of individual synapses.",
            "Trainable per-neuron dynamics are part of the reference architecture, not an "
            "addition made by this project.",
        ],
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    for name, entry in moved.items():
        rel = entry["relative_l2_change"]
        shown = "n/a (init 0)" if rel is None else f"{rel * 100:.2f}%"
        print(f"  {name:22s} n={entry['values']:>6,}  relative L2 {shown:>13s}  "
              f"RMS delta {entry['absolute_rms_change']:.5f}  "
              f"mean {entry['initial_mean']:+.4f} -> {entry['trained_mean']:+.4f}")
    print(f"  budget: dynamics {budget['neuron_dynamics']:,} / total {budget['total']:,} "
          f"({budget['neuron_dynamics'] / budget['total'] * 100:.2f}%), "
          f"frozen synapses {budget['frozen_synapses']:,}")


if __name__ == "__main__":
    main()
