#!/usr/bin/env python3
"""Assemble a Hugging Face repository for a graph-control arm.

A control arm is packaged differently from an ordinary one in exactly one way that matters:
its weights are bound to a graph that does not exist in any dataset. Loading them onto the
measured connectome would silently produce a model that was never trained, so the manifest
carries the control receipt, the rewired graph's digest, and the seed and script needed to
rebuild that graph — and the frozen-buffer digests are checked against the receipt here
rather than being copied through on trust.

The connectome itself is still not duplicated. The published dataset holds the measured
graph; this arm's graph is a deterministic function of that dataset plus one integer.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="The arm's results directory, holding best.pt and the receipts")
    parser.add_argument("--audit-record", type=Path,
                        default=ROOT / "experiments/records/graph-control-v1.json")
    parser.add_argument("--stage", default="07-graph-control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")

    spec = json.loads((ROOT / "experiments/configs" / (args.experiment + ".json")).read_text())
    selectors = json.loads((args.run_dir / "selected-checkpoints.json").read_text())["selectors"]
    control = json.loads((args.run_dir / "graph-control.json").read_text())
    status = json.loads((args.run_dir / "status.json").read_text())
    audit = json.loads(args.audit_record.read_text())

    if status.get("updates") != spec["training"]["max_updates"]:
        raise SystemExit(f"Arm stopped at {status.get('updates')} of {spec['training']['max_updates']}")
    if control["changed"]["duplicate_edges_created"] or control["changed"]["self_loops_created"]:
        raise SystemExit("Control graph is not simple; refusing to publish")
    if not all(control["preserved"].values()):
        raise SystemExit(f"Control graph did not preserve degrees: {control['preserved']}")

    args.output.mkdir(parents=True)
    exported = {}
    for criterion, filename, name in (("minimum_validation_ce", "best.pt", "min-ce"),
                                      ("maximum_validation_accuracy", "best-accuracy.pt",
                                       "max-accuracy")):
        source = args.run_dir / filename
        found = digest(source)
        if found != selectors[criterion]["sha256"]:
            raise SystemExit(f"{criterion}: {found} does not match the selector receipt")
        saved = torch.load(source, map_location="cpu", weights_only=False)
        # The weights only mean anything against the graph they were trained on. The receipt
        # is taken on the reference model, before the cell-type grouping installs
        # node_type_index and edge_group_index, so the checkpoint legitimately carries two
        # buffers the receipt does not. Every buffer the receipt does assert must match.
        recorded = control["frozen_buffers_sha256"]
        mismatched = [k for k, v in recorded.items() if saved["frozen_buffers_sha256"].get(k) != v]
        if mismatched:
            raise SystemExit(f"{criterion} was trained on a different graph than the receipt "
                             f"records; differing buffers: {', '.join(sorted(mismatched))}")
        if saved["frozen_buffers_sha256"]["brain.w_indices"] != control["control_graph_sha256"]["w_indices"]:
            raise SystemExit(f"{criterion} is not bound to the control graph")
        tensors = {k: v.contiguous() for k, v in saved["parameters"].items()}
        target = args.output / f"{name}.safetensors"
        save_file(tensors, target, metadata={
            "experiment": args.experiment, "selector": criterion,
            "updates": str(saved["cursor"]["updates"]), "source_checkpoint_sha256": found,
            "graph": "randomised control graph; NOT the measured connectome",
            "graph_w_indices_sha256": control["control_graph_sha256"]["w_indices"],
        })
        exported[criterion] = {
            "file": target.name, "updates": saved["cursor"]["updates"],
            "source_checkpoint_sha256": found, "safetensors_sha256": digest(target),
            "parameters": int(sum(v.numel() for v in tensors.values())),
            "validation": selectors[criterion]["validation"],
            "frozen_buffers_sha256": saved["frozen_buffers_sha256"],
        }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "stage": args.stage,
        "role": "randomised-graph control for " + spec["comparison_baseline"],
        "trainable_parameters": spec["trainable_parameters"],
        "architecture": {
            "neurons": 49393, "edges": 9050172,
            "encoder_width": spec["training"]["d_embed"],
            "readout_rank": spec["training"]["readout_rank"],
            "delay_slots": spec["training"]["history_length"], "vocabulary": 1024,
            "edge_adaptation": spec["training"]["plasticity"],
            "recurrence": "x = 0.1*x + 0.9*tanh(gain*(rec_gain*(W@x) + drive) + bias)",
        },
        "graph_control": {
            "WARNING": "These weights were trained on a RANDOMISED graph. Loading them onto the "
                       "measured connectome gives a model that was never trained and its scores "
                       "are meaningless. Rebuild the control graph first.",
            "mode": control["mode"], "seed": control["seed"],
            "rebuild_with": "scripts/train_connectorch_control.py :: control_indices(offsets, "
                            "source, mode, seed), applied to the fly-connectome-49k graph",
            "measured_graph_w_indices_sha256": control["original_graph_sha256"]["w_indices"],
            "control_graph_w_indices_sha256": control["control_graph_sha256"]["w_indices"],
            "synaptic_weights_unchanged": control["original_graph_sha256"]["w_values"]
                                          == control["control_graph_sha256"]["w_values"],
            "preserved": control["preserved"], "changed": control["changed"],
        },
        "training": {
            "corpus": "10,000 TinyStories (identical corpus and splits to the arm it controls)",
            "corpus_sha256": spec["dataset_sha256"],
            "updates": status["updates"], "passes_over_corpus": 1.83, "budget_capped": True,
            "status": "debug_stopped — the trainer's label for an update cap, not a failure",
            "converged": False, "seed": spec["training"]["seed"],
            "host": "Apple M3 Max, PyTorch MPS",
        },
        "selectors": exported,
        "held_out_scores": audit["scores"],
        "stage7_finding": {
            "question": audit["question"],
            "audit_delta_vs_measured_connectome": audit["graph_contrast"]["audit"],
            "reading": audit["reading"],
        },
        "graph": {"dataset": "fernandofernandes/fly-connectome-49k",
                  "note": "Not duplicated here. This arm's graph is a deterministic function of "
                          "that dataset and the seed recorded above."},
        "license": {"code": "MIT", "weights": "CC-BY-4.0",
                    "attribution": "MaleCNS v1.0 (FlyEM / HHMI Janelia, University of Cambridge, "
                                   "MRC LMB, Google Research) and the ngxson/fly-llm-hf architecture."},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "graph-control.json").write_text(json.dumps(control, indent=2) + "\n")
    shutil.copy(ROOT / "experiments/configs" / (args.experiment + ".json"),
                args.output / "experiment-config.json")
    print(json.dumps({"output": str(args.output), "experiment": args.experiment,
                      "selectors": {k: v["updates"] for k, v in exported.items()},
                      "bytes": sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file())},
                     indent=2))


if __name__ == "__main__":
    main()
