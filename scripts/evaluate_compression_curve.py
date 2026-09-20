#!/usr/bin/env python3
"""Score the released reference and every preserved arm on one shared population pair.

Existing per-arm reports replay the 100-story selection-validation split, while
the released-model audit used a separate 200-story population. Those two tables
cannot be read against each other. This evaluator scores the released reference,
our full-size reconstruction and any ConnecTorch arm on *both* splits in a single
run so one comparison table is defensible.

Inference only: no optimizer, no backward pass, and the reserved 100-story test
is never opened. Arms are declared in a JSON file so the exact population, the
exact checkpoints and their hashes stay in the receipt.
"""
import argparse
from datetime import datetime, timezone
import copy
import gc
import json
from pathlib import Path
import platform
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F

from train_connectorch import (atomic_json, batch_tensors, build_model, environment_receipt,
                               file_hash, frozen_hashes, load_groups, parameter_audit,
                               parameters_cpu, restore_parameters, source_receipt, verify_reference)
from evaluate_ngxson_quality import aggregate, paired_comparison
from evaluate_connectorch_quality import PROMPTS, generate, require, training_overlap

SELECTOR_FILES = {"minimum_validation_ce": "best.pt", "maximum_validation_accuracy": "best-accuracy.pt"}


def load_arms(path):
    """Read the declared arm list; every field that changes a number is explicit."""
    arms = json.loads(Path(path).read_text())
    require(isinstance(arms, list) and arms, "Arm specification must be a nonempty list")
    labels = [a["label"] for a in arms]
    require(len(set(labels)) == len(labels), "Duplicate arm labels")
    baselines = [a for a in arms if a.get("baseline")]
    require(len(baselines) == 1, "Exactly one arm must be marked baseline")
    require(baselines[0]["kind"] == "reference", "The baseline must be the released reference")
    for arm in arms:
        require(arm["kind"] in ("reference", "ngxson", "connectorch"), "Unknown arm kind: " + arm["label"])
        if arm["kind"] != "reference":
            require(Path(arm["checkpoint"]).is_file(), "Missing checkpoint: " + arm["label"])
            require(len(arm.get("sha256", "")) == 64, "Every arm must pin its checkpoint SHA256: " + arm["label"])
        if arm["kind"] == "connectorch":
            require((ROOT / "experiments/configs" / (arm["config"] + ".json")).is_file(),
                    "Missing configuration: " + arm["label"])
        control = arm.get("graph_control")
        if control is not None:
            require(arm["kind"] == "connectorch", "Only connectorch arms can carry a graph control")
            require(control.get("mode") in ("shuffle", "zero"),
                    "Unknown graph control mode: " + arm["label"])
            require(isinstance(control.get("seed"), int), "A graph control must pin its seed: " + arm["label"])
    return arms


def connectorch_training_args(config_name):
    """Bind architecture to the committed experiment specification, not to CLI memory."""
    spec = json.loads((ROOT / "experiments/configs" / (config_name + ".json")).read_text())["training"]
    for key in ("d_embed", "plasticity", "readout_rank", "history_length", "seed"):
        require(key in spec, f"Configuration {config_name} does not declare {key}")
    return argparse.Namespace(d_embed=spec["d_embed"], plasticity=spec["plasticity"],
                              readout_rank=spec["readout_rank"],
                              history_length=spec["history_length"], seed=spec["seed"],
                              leak=spec.get("leak", "fixed")), spec


def score(model, rows, label, output, device, batch_size=8, chunk_size=32):
    """Token-weighted CE and top-1 accuracy with per-story records for paired statistics.

    Mirrors the accepted arm evaluator's loop exactly: whole stories, fresh cache
    per batch, chunked TBPTT-shaped forward passes, no gradient anywhere.
    """
    model.eval()
    pad = model.config.pad_token_id
    records = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            inputs, targets, mask = batch_tensors(group, device, pad)
            cache = None
            totals = torch.zeros(len(group), dtype=torch.float64)
            counts = torch.zeros(len(group), dtype=torch.long)
            hits = torch.zeros_like(counts)
            for offset in range(0, inputs.shape[1], chunk_size):
                sl = slice(offset, offset + chunk_size)
                out = model(input_ids=inputs[:, sl], attention_mask=mask[:, sl],
                            cache_params=cache, use_cache=True, return_dict=True)
                cache = out.cache_params
                require(bool(torch.isfinite(out.logits).all()), "Nonfinite logits: " + label)
                target = targets[:, sl]
                valid = target.ne(pad)
                losses = F.cross_entropy(out.logits.flatten(0, 1), target.flatten(),
                                         ignore_index=pad, reduction="none").reshape_as(target)
                totals += losses.cpu().double().sum(dim=1)
                counts += valid.sum(dim=1).cpu()
                hits += ((out.logits.argmax(-1) == target) & valid).sum(dim=1).cpu()
            for i, row in enumerate(group):
                require(int(counts[i]) == len(row["ids"]) - 1, "Lost next-token targets: " + label)
                records.append({"id": row["id"], "tokens": int(counts[i]),
                                "correct": int(hits[i]), "nll_sum": float(totals[i])})
            atomic_json(output / "status.json",
                        {"stage": label, "stories_completed": len(records), "stories_total": len(rows),
                         "seconds": time.monotonic() - started, "partial": aggregate(records)})
    return {"summary": aggregate(records), "per_story": records, "seconds": time.monotonic() - started}


def controlled_reference(reference_cpu, control):
    """A private copy of the reference whose graph is a control arm's randomised one.

    The connectorch model snapshots its graph at construction and refuses every later
    edit, which is the right behaviour: a model is bound to one graph. So the graph has to
    be replaced on the *reference*, before the arm's model is built from it — which is
    exactly where the trainer does it too. The copy is private because every other arm in
    the run is built from this same reference.

    The graph is rebuilt from the declared seed rather than read out of the checkpoint, so
    the frozen-buffer comparison in build_arm is a real verification: it passes only if
    this process independently reproduced, bit for bit, the graph the arm trained on.
    """
    import train_connectorch_control as ctl
    private = copy.deepcopy(reference_cpu)
    receipt = ctl.rewire(private.brain, control["mode"], control["seed"])
    require(not receipt["changed"]["duplicate_edges_created"]
            and not receipt["changed"]["self_loops_created"],
            "Rebuilt control graph is not simple")
    require(all(receipt["preserved"].values()),
            "Rebuilt control graph did not preserve the degree structure")
    return private, receipt


def build_arm(arm, model_dir, reference_cpu, groups, device):
    """Return (model, checkpoint_metadata). Never mutates the shared CPU reference."""
    from transformers import AutoModelForCausalLM
    if arm["kind"] == "connectorch":
        args, spec = connectorch_training_args(arm["config"])
        control, rebuilt = arm.get("graph_control"), None
        if control is not None:
            reference_cpu, rebuilt = controlled_reference(reference_cpu, control)
        model = build_model(reference_cpu, args, groups)
        meta = {"architecture": "connectorch", "declared_training": spec,
                "graph_control": control, "graph_control_rebuild": rebuilt}
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True,
                                                     local_files_only=True, torch_dtype=torch.float32)
        meta = {"architecture": "ngxson_reference"}
    model = model.eval()
    reference_frozen = frozen_hashes(model)
    if arm["kind"] == "reference":
        meta.update({"checkpoint": None, "checkpoint_sha256": None,
                     "frozen_buffers_sha256": reference_frozen, "cursor": None})
        released_audit = parameter_audit(parameters_cpu(model))
    else:
        digest = file_hash(arm["checkpoint"])
        require(digest == arm["sha256"],
                f"{arm['label']} checkpoint hash differs from its accepted-stop receipt")
        saved = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False)
        require(file_hash(arm["checkpoint"]) == digest, "Checkpoint changed during load: " + arm["label"])
        for name, value in saved["frozen_buffers_sha256"].items():
            require(reference_frozen.get(name) == value,
                    f"{arm['label']} was trained on a different frozen buffer: {name}")
        restore_parameters(model, saved["parameters"])
        restored = parameter_audit(parameters_cpu(model))
        for name, entry in saved["parameter_audit"].items():
            require(restored[name]["sha256"] == entry["sha256"],
                    f"{arm['label']} restored parameter differs from its checkpoint: {name}")
        meta.update({"checkpoint": str(arm["checkpoint"]), "checkpoint_sha256": digest,
                     "frozen_buffers_sha256": saved["frozen_buffers_sha256"],
                     "cursor": saved["cursor"],
                     "selected_validation": saved.get("best") if arm.get("selector") == "minimum_validation_ce"
                                            else saved.get("best_accuracy")})
        del saved
        gc.collect()
        released_audit = restored
    model.requires_grad_(False)
    if device == "mps":
        if arm["kind"] == "connectorch":
            model.to(device)
            # Build the derived CSR layout outside inference mode so a later
            # inference-mode generation pass cannot pin an inference tensor.
            with torch.inference_mode(False):
                model.brain.sparse_runtime()
        else:
            from fly_wordbrain.ngxson_mps import enable_mps
            model = enable_mps(model)
    meta["parameter_count"] = sum(p.numel() for p in model.parameters())
    return model, meta, released_audit


def report_markdown(protocol, arms, comparisons):
    lines = ["# Shared-population comparison: released reference and preserved arms", "",
        f"Inference only on {protocol['hostname']} ({protocol['device']}). No optimizer, no backward pass.",
        "The reserved 100-story test was not opened.", "",
        "Both populations are scored for every arm so one table is comparable. The",
        "validation split selected our checkpoints, so it flatters our arms and not the",
        "released model; the audit split selected none of them. Read the audit column",
        "for the selection-unbiased comparison.", "",
        "| Arm | Trainable parameters | Validation CE | Validation acc | Audit CE | Audit PPL | Audit acc |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        v = arm["validation"]["summary"]
        a = arm["audit"]["summary"]
        lines.append(f"| {arm['label']} | {arm['meta']['parameter_count']:,} | {v['cross_entropy']:.6f} | "
                     f"{v['top1_accuracy'] * 100:.4f}% | {a['cross_entropy']:.6f} | {a['perplexity']:.2f} | "
                     f"{a['top1_accuracy'] * 100:.4f}% |")
    lines.extend(["", "## Paired story bootstrap against the released reference", "",
        "Direction is arm minus released; 10,000 paired whole-story resamples, seed 1729.", "",
        "| Arm | Split | ΔCE | ΔCE 95% CI | Δaccuracy | Δaccuracy 95% CI |", "|---|---|---:|---|---:|---|"])
    for label, splits in comparisons.items():
        for split, c in splits.items():
            ci = f"[{c['ce_95_ci'][0]:+.4f}, {c['ce_95_ci'][1]:+.4f}]"
            aci = f"[{c['accuracy_95_ci'][0] * 100:+.3f}, {c['accuracy_95_ci'][1] * 100:+.3f}] pp"
            lines.append(f"| {label} | {split} | {c['delta_cross_entropy']:+.6f} | {ci} | "
                         f"{c['delta_accuracy'] * 100:+.4f} pp | {aci} |")
    lines.extend(["", "A negative ΔCE means the arm assigns higher probability to held-out text than the",
        "released 52,756,661-parameter reference. That is a measurement on this corpus, not",
        "evidence of an anatomical prior: the released model's own training stories are",
        "unknown and may overlap either population.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned reference model directory")
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--audit-data", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    arms = load_arms(args.arms)
    verified = verify_reference(args.model)
    training_data = json.loads(args.training_data.read_text())
    audit_rows = json.loads(args.audit_data.read_text())["audit"]
    validation_rows = training_data["validation"]
    existing = [r for s in ("train", "validation", "test") for r in training_data[s]]
    require(len(audit_rows) == 200, "Audit population must hold 200 stories")
    require(not {r["id"] for r in existing}.intersection(r["id"] for r in audit_rows),
            "Audit population overlaps an existing split by story ID")
    require(not {r["normalized_sha256"] for r in existing}.intersection(
        r["normalized_sha256"] for r in audit_rows), "Audit population overlaps an existing split by text")

    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "training_run": False,
        "reserved_test_evaluated": False, "device": args.device, "threads": args.threads,
        "hostname": socket.gethostname(), "platform": platform.platform(), "torch": torch.__version__,
        "environment": environment_receipt(), "evaluator_sha256": file_hash(Path(__file__)),
        "source_receipt": source_receipt(), "reference_files": verified,
        "arms_spec_sha256": file_hash(args.arms), "arms": arms,
        "training_data_sha256": file_hash(args.training_data),
        "audit_data_sha256": file_hash(args.audit_data),
        "groups_sha256": file_hash(args.groups),
        "populations": {
            "validation": {"stories": len(validation_rows),
                           "role": "selected every retained checkpoint; biased in our arms' favour"},
            "audit": {"stories": len(audit_rows),
                      "role": "200 fresh official-validation stories excluded from all 1,200 existing rows"}},
        "generation": {"prompts": PROMPTS, "max_new_tokens": 80, "greedy": True,
                       "fresh_cache_per_prompt": True, "bos_once": True, "eos_stopping": True},
        "limitations": [
            "The released model's exact training stories are unknown; either population may overlap them.",
            "Validation scores selected our checkpoints and are not a held-out estimate for our arms.",
            "The audit population was revealed by an earlier audit and is no longer untouched.",
            "One seed per arm; no randomized-graph or zero-edge control is included here."]}
    atomic_json(args.output / "protocol.json", protocol)

    reference_cpu = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                                         local_files_only=True, torch_dtype=torch.float32).cpu()
    groups, group_metadata = load_groups(args.groups)
    require(group_metadata["frozen_buffers_sha256"] == frozen_hashes(reference_cpu),
            "Grouping belongs to a different reference graph")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    require((tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (0, 1, 2),
            "Pinned special-token IDs differ")

    results = []
    for arm in arms:
        model, meta, before = build_arm(arm, args.model, reference_cpu, groups, args.device)
        entry = {"label": arm["label"], "spec": arm, "meta": meta}
        for split, rows in (("validation", validation_rows), ("audit", audit_rows)):
            entry[split] = score(model, rows, arm["label"] + "/" + split, args.output, args.device)
            atomic_json(args.output / (arm["label"].replace("/", "_") + ".json"), entry)
        if arm.get("expected_validation_ce") is not None:
            gap = entry["validation"]["summary"]["cross_entropy"] - arm["expected_validation_ce"]
            entry["validation_replay_delta"] = gap
            require(abs(gap) < 1e-4, f"{arm['label']} failed its saved validation replay: {gap}")
        if not args.skip_generation:
            entry["samples"] = generate(model, tokenizer, args.device)
            entry["training_overlap"] = training_overlap(entry["samples"], training_data["train"])
        after = parameter_audit(parameters_cpu(model))
        require(before == after, "Scoring changed parameters: " + arm["label"])
        require(all(p.grad is None and not p.requires_grad for p in model.parameters()),
                "Gradients appeared during inference: " + arm["label"])
        entry["parameters_unchanged"] = True
        entry["frozen_graph_unchanged"] = frozen_hashes(model) == meta["frozen_buffers_sha256"] \
            if meta["frozen_buffers_sha256"] else True
        require(entry["frozen_graph_unchanged"], "Scoring changed the frozen graph: " + arm["label"])
        atomic_json(args.output / (arm["label"].replace("/", "_") + ".json"), entry)
        results.append(entry)
        del model
        gc.collect()

    baseline = next(e for e in results if e["spec"].get("baseline"))
    comparisons = {e["label"]: {s: paired_comparison(e[s]["per_story"], baseline[s]["per_story"])
                                for s in ("validation", "audit")}
                   for e in results if e is not baseline}
    summary = [{"label": e["label"], "parameters": e["meta"]["parameter_count"],
                "validation": e["validation"]["summary"], "audit": e["audit"]["summary"]} for e in results]
    atomic_json(args.output / "results.json",
                {"protocol": protocol, "summary": summary, "comparisons": comparisons,
                 "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    (args.output / "report.md").write_text(report_markdown(protocol, results, comparisons) + "\n")
    print(json.dumps({"completed": True, "output": str(args.output),
                      "summary": [{k: s[k] for k in ("label", "parameters")} for s in summary]}), flush=True)


if __name__ == "__main__":
    main()
