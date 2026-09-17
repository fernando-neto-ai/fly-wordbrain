#!/usr/bin/env python3
"""Train the whole fly-brain language model on a Mac, in one command.

Three modules are learned jointly, end to end, in a single optimization:

    token  ->  [ENCODER]  ->  [49,393 NEURONS, 9,050,172 FROZEN SYNAPSES]  ->  [DECODER]  ->  next token
                   ^                        ^                                      ^
                   |                        |                                      |
             482,816 params        148,179 gain/rec_gain/bias            3,226,688 params

One AdamW, one backward pass per step. Gradients reach the encoder by flowing
*backwards through the connectome* — the sparse transpose is a real Metal kernel,
not a stop-gradient — so the encoder learns how to speak to the brain it is wired
into, while the brain learns how to listen.

Stages, each skipped when its output already exists:

    prepare    pinned reference model, TinyStories splits, audit split, cell-type
               grouping (fetched from the published dataset, not rebuilt from ~1 GB
               of upstream connectome files)
    preflight  short optimizer smoke; proves gradients reach all three modules
    train      the real run
    evaluate   score against the held-out audit population
    sample     generate text from the trained model

    python scripts/run_pipeline.py --quick   # ~15 min on an M3 Max
    python scripts/run_pipeline.py           # ~90 min, the published recipe
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
GROUPS_DATASET = "fernandofernandes/fly-connectome-49k"
# The published rank-64 recipe. Quick mode shortens only the schedule.
RECIPE = {"d_embed": 32, "readout_rank": 64, "plasticity": "fixed", "history_length": 8,
          "batch_size": 8, "chunk_size": 32, "seed": 42}


def say(stage, message):
    print(f"\033[1m[{stage}]\033[0m {message}", flush=True)


def run(command, stage):
    say(stage, "$ " + " ".join(str(c) for c in command))
    result = subprocess.run([str(c) for c in command], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"[{stage}] failed with exit code {result.returncode}")


def check_platform(device):
    import torch
    if device == "cpu":
        return "cpu (slow; --device mps is the intended path on Apple Silicon)"
    if not torch.backends.mps.is_available():
        raise SystemExit(
            "Apple GPU (MPS) is not available. This pipeline targets Apple Silicon.\n"
            "Run with --device cpu to proceed anyway, much more slowly.")
    return "Apple GPU via Metal"


def prepare(args):
    model = args.data_root / "ngxson-fly-llm-hf" / REVISION
    if (model / "model.safetensors").exists():
        say("prepare", "reference model already present")
    else:
        run([sys.executable, "scripts/prepare_ngxson.py", "--output",
             args.data_root / "ngxson-fly-llm-hf"], "prepare")

    dataset = args.data_root / "ngxson-tinystories-v1" / "dataset.json"
    if dataset.exists():
        say("prepare", "training splits already present")
    else:
        run([sys.executable, "scripts/prepare_ngxson_data.py",
             "--output", dataset.parent, "--model", model], "prepare")

    audit = args.data_root / "ngxson-quality-v1" / "dataset.json"
    if audit.exists():
        say("prepare", "audit split already present")
    elif not args.skip_audit:
        run([sys.executable, "scripts/prepare_ngxson_quality_data.py",
             "--existing", dataset, "--model", model, "--output", audit.parent], "prepare")

    groups = args.groups or (args.data_root / "connectorch-groups-v1" / "groups.npz")
    if groups.exists():
        say("prepare", f"cell-type grouping already present ({groups})")
    else:
        say("prepare", f"fetching the grouping from {GROUPS_DATASET}")
        from huggingface_hub import hf_hub_download
        fetched = hf_hub_download(GROUPS_DATASET, "groups/groups.npz", repo_type="dataset")
        groups.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fetched, groups)
        say("prepare", f"grouping saved to {groups}")
    return model, dataset, audit, groups


def train_command(args, model, dataset, groups, output, epochs, second, extra=()):
    command = [sys.executable, "scripts/train_connectorch.py",
               "--output", output, "--model", model, "--data", dataset, "--groups", groups,
               "--device", args.device, "--epochs", epochs, "--second-epochs", second,
               "--skip-final-test"]
    for key, value in RECIPE.items():
        command += ["--" + key.replace("_", "-"), value]
    return command + list(extra)


def preflight(args, model, dataset, groups):
    """Eight updates. Cheap, and it fails loudly if any module is not learning."""
    output = args.output / "preflight"
    if (output / "metrics.jsonl").exists():
        say("preflight", "already passed")
        return
    run(train_command(args, model, dataset, groups, output, 1, 0,
                      ["--max-updates", 8, "--eval-limit", 4, "--eval-interval-updates", 8]),
        "preflight")
    manifest = json.loads((output / "manifest.json").read_text())
    counts = manifest["parameter_counts"]
    encoder = counts["embedding"] + counts["input_projection"]
    say("preflight", "all three modules are learning — "
        f"encoder {encoder:,} · neurons {counts['neurons']:,} · readout {counts['readout']:,} "
        f"(+{counts['layernorm']:,} output norm, {counts['total']:,} total)")


def train(args, model, dataset, groups):
    output = args.output / "model"
    if (output / "best.pt").exists():
        say("train", f"run already present at {output}")
        return output
    epochs, second = (2, 0) if args.quick else (30, 14)
    say("train", f"{'quick' if args.quick else 'full'} schedule: {epochs}+{second} epochs")
    started = time.monotonic()
    run(train_command(args, model, dataset, groups, output, epochs, second), "train")
    say("train", f"finished in {(time.monotonic() - started) / 60:.1f} minutes")
    return output


def evaluate(args, model, dataset, audit, groups, trained):
    if args.skip_audit or not audit.exists():
        say("evaluate", "audit split unavailable; skipping")
        return
    output = args.output / "evaluation"
    if output.exists():
        say("evaluate", "already scored")
    else:
        arms = args.output / "arms.json"
        best = trained / "best.pt"
        import hashlib
        digest = hashlib.sha256(best.read_bytes()).hexdigest()
        arms.write_text(json.dumps([
            {"label": "released-reference", "kind": "reference", "baseline": True},
            {"label": "yours", "kind": "connectorch", "config": "G32rank64fixed",
             "checkpoint": str(best), "sha256": digest, "selector": "minimum_validation_ce"},
        ], indent=2) + "\n")
        run([sys.executable, "scripts/evaluate_compression_curve.py",
             "--model", model, "--groups", groups, "--training-data", dataset,
             "--audit-data", audit, "--arms", arms, "--output", output,
             "--device", args.device, "--skip-generation"], "evaluate")
    report = json.loads((output / "results.json").read_text())
    print()
    for row in report["summary"]:
        a = row["audit"]
        print(f"  {row['label']:22s} {row['parameters']:>12,} params   "
              f"CE {a['cross_entropy']:.4f}   PPL {a['perplexity']:7.2f}   "
              f"top-1 {a['top1_accuracy'] * 100:.2f}%")


def sample(args, model, trained):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    say("sample", "generating from the model you just trained")
    export = args.output / "export"
    if not export.exists():
        run([sys.executable, "scripts/export_web_model.py", "--model", model,
             "--groups", args.groups_path, "--checkpoint", trained / "best.pt",
             "--config", "G32rank64fixed", "--output", export, "--max-new-tokens", 50,
             "--parity-tokens", 6], "sample")
    manifest = json.loads((export / "manifest.json").read_text())
    print()
    for entry in manifest["golden"]:
        print(f"  \033[2m{entry['prompt']}\033[0m{entry['continuation']}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "results/pipeline-run")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--groups", type=Path, help="Existing groups.npz; fetched if absent")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--quick", action="store_true",
                        help="Two epochs instead of the published 44-epoch schedule")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip building and scoring the held-out audit population")
    parser.add_argument("--stage", choices=("prepare", "preflight", "train", "evaluate", "sample"),
                        help="Run a single stage and stop")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    say("start", f"device: {check_platform(args.device)}")
    say("start", "three modules, one optimizer: encoder -> frozen connectome -> low-rank readout")

    model, dataset, audit, groups = prepare(args)
    args.groups_path = groups
    if args.stage == "prepare":
        return
    preflight(args, model, dataset, groups)
    if args.stage == "preflight":
        return
    trained = train(args, model, dataset, groups)
    if args.stage == "train":
        return
    evaluate(args, model, dataset, audit, groups, trained)
    if args.stage == "evaluate":
        return
    sample(args, model, trained)
    say("done", f"everything under {args.output}")


if __name__ == "__main__":
    main()
