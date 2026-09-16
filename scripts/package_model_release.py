#!/usr/bin/env python3
"""Assemble a Hugging Face model repository for one trained arm.

Ships the learned parameters as safetensors for both retained selectors, the raw
float32 arrays the browser demo streams, and a manifest that binds everything to
the accepted-stop receipts. The connectome itself is not duplicated here: it lives
once in the fly-connectome-49k dataset, and the manifest records the frozen-buffer
digests that prove which graph these weights were trained on.
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Directory holding <experiment>-best.pt and -best-accuracy.pt")
    parser.add_argument("--web-export", type=Path, help="Exported float32 arrays for the browser")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)

    spec = json.loads((ROOT / "experiments/configs" / (args.experiment + ".json")).read_text())
    receipt = json.loads((ROOT / "experiments/records" / f"{args.experiment}-accepted-early-stop.json").read_text())
    selectors = receipt["selected_checkpoints"].get("selectors", receipt["selected_checkpoints"])

    exported = {}
    for criterion, filename in (("minimum_validation_ce", "best.pt"),
                                ("maximum_validation_accuracy", "best-accuracy.pt")):
        source = args.checkpoint_dir / f"{args.experiment}-{filename}"
        found = digest(source)
        expected = selectors[criterion]["sha256"]
        if found != expected:
            raise SystemExit(f"{criterion}: checkpoint hash {found} does not match receipt {expected}")
        saved = torch.load(source, map_location="cpu", weights_only=False)
        tensors = {name: value.contiguous() for name, value in saved["parameters"].items()}
        name = "min-ce" if criterion == "minimum_validation_ce" else "max-accuracy"
        target = args.output / f"{name}.safetensors"
        save_file(tensors, target, metadata={
            "experiment": args.experiment, "selector": criterion,
            "updates": str(saved["cursor"]["updates"]), "source_checkpoint_sha256": found,
        })
        exported[criterion] = {
            "file": target.name, "selector": criterion,
            "updates": saved["cursor"]["updates"],
            "source_checkpoint_sha256": found,
            "safetensors_sha256": digest(target),
            "validation": selectors[criterion]["validation"],
            "frozen_buffers_sha256": saved["frozen_buffers_sha256"],
            "parameters": int(sum(v.numel() for v in tensors.values())),
        }

    web = None
    if args.web_export:
        shutil.copytree(args.web_export, args.output / "web")
        web = json.loads((args.output / "web/manifest.json").read_text())

    comparison = json.loads((ROOT / "experiments/records/compression-curve-v1.json").read_text())
    rows = {r["label"]: r for r in comparison["summary"]}
    scores = {label: {"parameters": row["parameters"], "audit": row["audit"], "validation": row["validation"]}
              for label, row in rows.items() if label.startswith(args.experiment) or label == "released-reference"}

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "architecture": {
            "neurons": 49393, "edges": 9050172, "delay_slots": spec["training"]["history_length"],
            "encoder_width": spec["training"]["d_embed"], "readout_rank": spec["training"]["readout_rank"],
            "vocabulary": 1024, "edge_adaptation": spec["training"]["plasticity"],
            "recurrence": "x = 0.1*x + 0.9*tanh(gain*(rec_gain*(W@x) + drive) + bias)",
        },
        "trainable_parameters": spec["trainable_parameters"],
        "encoder_parameters": spec["encoder_parameters"],
        "readout_parameters": spec["readout_parameters"],
        "graph": {
            "dataset": "fernandofernandes/fly-connectome-49k",
            "note": "The connectome is not duplicated here. The frozen-buffer digests below "
                    "identify the exact graph these weights were trained on; the dataset "
                    "manifest carries the same digests.",
        },
        "selectors": exported,
        "held_out_scores": scores,
        "training": {
            "reference": spec["reference_repository"], "reference_revision": spec["reference_revision"],
            "dataset": "TinyStories, 1,000 training stories (not redistributed)",
            "dataset_sha256": spec["dataset_sha256"],
            "seed": spec["training"]["seed"], "host": "Apple M3 Max, PyTorch MPS",
            "stop": "accepted early stop under a plateau rule; the planned 44-epoch schedule "
                    "was not completed",
        },
        "web_export": None if web is None else {
            "files": list(web["files"]), "config": web["config"],
            "golden_prompts": [entry["prompt"] for entry in web["golden"]],
            "note": "Raw little-endian float32 arrays streamed by the browser demo. The "
                    "manifest's golden trace was verified against PyTorch token-for-token.",
        },
        "license": {"code": "MIT", "weights": "CC-BY-4.0",
                    "attribution": "Derived from the MaleCNS v1.0 connectome (FlyEM / HHMI Janelia, "
                                   "University of Cambridge, MRC LMB, Google Research) and the "
                                   "ngxson/fly-llm-hf architecture."},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy(ROOT / "experiments/configs" / (args.experiment + ".json"), args.output / "experiment-config.json")
    print(json.dumps({"output": str(args.output), "experiment": args.experiment,
                      "parameters": manifest["trainable_parameters"],
                      "selectors": {k: v["updates"] for k, v in exported.items()},
                      "bytes": sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file())}, indent=2))


if __name__ == "__main__":
    main()
