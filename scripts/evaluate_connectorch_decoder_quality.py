#!/usr/bin/env python3
"""Compare accepted E/rank128 and G/rank64 winners on an idle macm3 GPU.

Inference only. Both selectors, full checkpoint-selection validation, six fixed
prompts, training overlap, exact matched budgets, and neuron parameter deltas.
The reserved test is never scored. Existing checksum-bound helpers are imported
unchanged; this file supplies rank-aware identity and cessation verification.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import evaluate_connectorch_quality as q

trainer, require, read_json = q.trainer, q.require, q.read_json
SELECTORS = q.SELECTORS
ARMS = {"E32rank128fixed": 128, "G32rank64fixed": 64}
NEURON_PARAMETERS = ("brain.gain", "brain.rec_gain", "brain.bias")
DISPATCHERS = q.DISPATCHERS | {
    "run_connectorch_decoder_campaign.py", "continue_connectorch_decoder_campaign.py",
    "run_connectorch_rank64_campaign.py", "continue_connectorch_rank64_campaign.py",
}
COMMON_CONFIG = {"epochs": 30, "second_epochs": 14, "batch_size": 8, "chunk_size": 32,
                 "seed": 42, "max_updates": None, "eval_limit": None,
                 "eval_interval_updates": 100, "device": "mps", "d_embed": 32,
                 "plasticity": "fixed", "history_length": 8, "skip_final_test": True}


def utc():
    return datetime.now(timezone.utc).isoformat()


def jsonl(path):
    # U+2028/U+2029 can occur inside JSON strings; only physical LF is a boundary.
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def parse_processes(text):
    records = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        fields = line.strip().split(None, 3)
        require(len(fields) == 4, "Unparseable process inventory")
        records.append(dict(zip(("pid", "ppid", "state", "command"),
                                (int(fields[0]), int(fields[1]), fields[2], fields[3]))))
    return records


def process_guard(records, worker_pids=(), own_pid=None):
    blocked, zombies, held = [], [], []
    for row in records:
        if row["pid"] == own_pid:
            continue
        if "Z" in row["state"]:
            if row["pid"] in worker_pids:
                zombies.append(row)
            continue
        try:
            tokens = shlex.split(row["command"])
        except ValueError:
            tokens = row["command"].split()
        names = {Path(token).name for token in tokens}
        executable = Path(tokens[0]).name.lower() if tokens else ""
        python = executable.startswith("python") or executable.endswith(".py")
        dispatcher = python and bool(names & DISPATCHERS)
        gpu_worker = python and any(name.startswith(("train_", "evaluate_", "compare_ngxson_texts",
                                                     "audit_connectorch", "export_rank128_demo"))
                                    and name.endswith(".py") for name in names)
        if row["pid"] in worker_pids or gpu_worker or (dispatcher and "T" not in row["state"]):
            blocked.append(row)
        elif dispatcher:
            held.append(row)
    require(not blocked, "Idle GPU audit requires exited workers and held dispatchers: " + json.dumps(blocked))
    return {"checked_at_utc": utc(), "workers_exited": True, "exited_zombies": zombies, "held_dispatchers": held}


def process_birth(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], text=True, capture_output=True)
    return " ".join(result.stdout.split()) if result.returncode == 0 else None


def idle_host_guard(worker_refs=()):
    hostname = socket.gethostname()
    require(platform.system() == "Darwin" and hostname.lower().removesuffix(".local") in q.HOST_NAMES,
            "Real evaluation is restricted to the registered macm3 host")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    require(chip.startswith("Apple M3") and torch.backends.mps.is_available(), "Expected available Apple M3 MPS")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", "MPS CPU fallback must be disabled")
    records = parse_processes(subprocess.check_output(["ps", "-axo", "pid=,ppid=,state=,command="], text=True))
    current = {row["pid"]: row for row in records}
    pids = set()
    for ref in worker_refs:
        pid, birth = ref["pid"], ref.get("birth")
        if pid in current and birth and process_birth(pid) != " ".join(birth.split()):
            continue  # PID was reused; any new trainer is still caught by its command.
        pids.add(pid)
    return {"hostname": hostname, "chip": chip, "device": "mps",
            **process_guard(records, pids, os.getpid())}


def expected_counts(rank):
    result = {"embedding": 32768, "input_projection": 450048, "neurons": 148179,
              "edge_gains": 0, "layernorm": 98786, "readout": rank * (49393 + 1024), "other": 0}
    return {**result, "total": sum(result.values())}


def verify_config(manifest, label):
    rank = ARMS[label]
    require(manifest.get("config") == {**COMMON_CONFIG, "readout_rank": rank}, "Wrong full-data fixed-edge architecture: " + label)
    require(manifest.get("debug") is False and manifest.get("parameter_counts") == expected_counts(rank),
            "Wrong parameter inventory or debug arm: " + label)
    require(manifest.get("reference_repository") == "ngxson/fly-llm-hf" and manifest.get("reference_revision") == trainer.REVISION,
            "Different reference checkpoint")


def verify_pair(e, g):
    keys = ("data_sha256", "groups_sha256", "reference_files", "frozen_buffers_sha256", "trainer_sources_sha256",
            "connectorch_source", "assumptions", "planned_phase_updates", "epoch_updates", "split_sizes", "data_provenance")
    for key in keys:
        require(e["manifest"][key] == g["manifest"][key], "E/G identity differs: " + key)
    require({k: v for k, v in e["manifest"]["config"].items() if k != "readout_rank"}
            == {k: v for k, v in g["manifest"]["config"].items() if k != "readout_rank"}, "Recipe differs beyond readout rank")


def verify_cessation(receipt, known_workers):
    cessation = receipt.get("process_cessation", {})
    processes = cessation.get("processes", [])
    require(cessation.get("confirmed") is True and processes, "Missing confirmed process cessation")
    indexed = {}
    for process in processes:
        require(isinstance(process.get("pid"), int) and process["pid"] > 0
                and process.get("alive") is False, "Invalid exited process receipt")
        require(process.get("exited") is not False, "Process receipt explicitly denies exit")
        indexed[process["pid"]] = process
    require(known_workers and known_workers <= indexed.keys(), "Cessation omits a recorded training worker")
    # Every process claimed exited is checked live, including its dispatcher.
    # Other dispatchers may remain held, but an accepted exit must be truthful.
    return [indexed[pid] for pid in sorted(indexed)]


def inspect_arm(directory, label):
    directory = Path(directory).resolve()
    require(directory.name == label, "Arm directory name differs from declared experiment")
    bindings = {}
    def bind(path, digest=None):
        return q.verify_bound(path, digest or trainer.file_hash(path), bindings)
    manifest = read_json(bind(directory / "manifest.json"))
    verify_config(manifest, label)
    accepted = read_json(bind(directory / "accepted-early-stop.json"))
    require(accepted.get("format_version") == 1 and accepted.get("arm") == label
            and accepted.get("status") == "accepted_early_stop"
            and accepted.get("accepted_by") in ("user", "agent_under_standing_user_authorization")
            and accepted.get("full_schedule_completed") is False and accepted.get("test_evaluated") is False,
            "Requires an accepted early stop with reserved test")
    for path, digest in accepted.get("artifact_sha256", {}).items():
        bind(path, digest)
    for name in ("manifest.json", "launch.json", "metrics.jsonl", "selected-checkpoints.json"):
        require(str(directory / name) in accepted.get("artifact_sha256", {}), "Acceptance omits immutable artifact: " + name)
    stop_ref = accepted["stop_receipt"]
    require(Path(stop_ref["path"]).resolve() == directory / "stop-receipt.json", "Stop receipt belongs to another arm")
    stop = read_json(bind(stop_ref["path"], stop_ref["sha256"]))
    require(stop.get("arm") == label and stop.get("process_cessation") == accepted["process_cessation"],
            "Accepted cessation and stop receipt differ")
    for path, digest in stop.get("sources_sha256", {}).items():
        bind(path, digest)
    if accepted.get("decision_receipt"):
        bind(accepted["decision_receipt"]["path"], accepted["decision_receipt"]["sha256"])
    statuses = {name: read_json(bind(directory / name)) for name in ("status.json", "process-status.json")}
    require(all(item.get("test_evaluated") is not True for item in statuses.values())
            and not (directory / "results.json").exists(), "Expected early-stopped arm with no final test result")
    known_workers = {statuses[name][key] for name, key in (("status.json", "pid"), ("process-status.json", "worker_pid"))
                     if isinstance(statuses[name].get(key), int)}
    worker_refs = verify_cessation(accepted, known_workers)
    require(trainer.source_receipt() == manifest["trainer_sources_sha256"], "Current trainer/model sources differ")
    for name, digest in manifest["trainer_sources_sha256"].items():
        bind(ROOT / name, digest)
    require(trainer.connectorch_receipt() == manifest["connectorch_source"], "Installed ConnecTorch differs")
    model_dir = Path(manifest["model_path"])
    require(trainer.verify_reference(model_dir) == manifest["reference_files"], "Pinned reference contents differ")
    for name, receipt in manifest["reference_files"].items():
        bind(model_dir / name, receipt["sha256"])
    for name, digest in manifest["source_sha256"].items():
        bind(ROOT / "scripts/train_connectorch.py" if name == "trainer" else model_dir / name, digest)
    data = read_json(bind(manifest["data_path"], manifest["data_sha256"]))
    require(len(data["train"]) == manifest["split_sizes"]["train"] == 1000
            and len(data["validation"]) == manifest["split_sizes"]["validation"] == 100
            and sum(len(row["ids"]) - 1 for row in data["validation"]) == 21874, "Full validation/training population differs")
    groups, metadata = trainer.load_groups(bind(manifest["groups_path"], manifest["groups_sha256"]))
    require(metadata == manifest["groups_provenance"] and groups.numel() == 49393
            and int(groups.max()) + 1 == manifest["node_types"] == 9161, "Group provenance differs")
    records = [row for row in jsonl(directory / "metrics.jsonl") if row.get("event") == "validation"]
    require(records and records[0]["updates"] == 0, "Validation history lacks initialization")
    for row in records:
        metric, audit = row["validation"], row["model_audit"]
        require(metric["stories"] == 100 and metric["tokens"] == 21874 and 0 <= metric["correct"] <= 21874
                and math.isfinite(metric["cross_entropy"]) and metric["cross_entropy"] >= 0
                and abs(metric["top1_accuracy"] - metric["correct"] / 21874) < 1e-12, "Invalid complete validation record")
        require(audit["frozen_buffers_preserved"] is True and audit["frozen_buffers_sha256"] == manifest["frozen_buffers_sha256"]
                and audit["parameter_counts"] == manifest["parameter_counts"], "Logged model/graph audit differs")
        adaptation = audit["adaptation"]
        require(adaptation["edge_displacement"]["rms"] == 0 and adaptation["edge_multiplier"]["min"] == 1
                and adaptation["edge_multiplier"]["max"] == 1, "Fixed arm changed canonical edge weights")
    native = q.verify_selectors(directory, manifest, records, bindings)
    preserved = accepted.get("selected_checkpoints", {})
    require(preserved.get("format_version") == 1 and set(preserved.get("selectors", {})) == set(SELECTORS), "Missing preserved dual selectors")
    selectors = preserved["selectors"]
    for selector, entry in selectors.items():
        require(Path(entry["path"]).resolve() == directory / "stopped-checkpoints" / SELECTORS[selector], "Unexpected preserved winner path")
        bind(entry["path"], entry["sha256"])
        require(entry.get("frozen_buffers_preserved") is True
                and all(entry[key] == native[selector][key] for key in ("sha256", "cursor", "validation", "frozen_buffers_sha256")),
                "Preserved winner and native selector differ")
    latest = accepted["latest_checkpoint"]
    require(Path(latest["path"]).resolve() == directory / "stopped-checkpoints/latest.pt"
            and latest.get("frozen_buffers_preserved") is True
            and latest["frozen_buffers_sha256"] == manifest["frozen_buffers_sha256"], "Latest preserved graph/path differs")
    bind(latest["path"], latest["sha256"]); bind(directory / "latest.pt", latest["sha256"])
    durable, observed, epochs = (accepted.get(k) for k in ("durable_updates", "observed_updates", "completed_epochs"))
    require(all(isinstance(v, int) for v in (durable, observed, epochs)) and 0 < durable <= observed < sum(manifest["planned_phase_updates"])
            and latest["cursor"]["updates"] == durable == max(row["updates"] for row in records)
            and latest["cursor"]["epoch"] == epochs, "Stop cursor/history budget differs")
    launch = read_json(directory / "launch.json")
    for path, digest in launch.get("sources_sha256", {}).items():
        bind(path, digest)
    training_path = bind(directory / "training.jsonl")
    return {"directory": directory, "manifest": manifest, "bindings": bindings, "data": data, "groups": groups,
            "records": records, "selectors": selectors, "latest": latest, "accepted": accepted, "stop": stop,
            "worker_refs": worker_refs, "launch": launch, "training_records": jsonl(training_path)}


def verify_saved(saved, arm, selector=None):
    if selector is not None:
        q.verify_checkpoint(saved, arm, selector)
        entry = arm["selectors"][selector]
    else:
        entry = arm["latest"]
        require(saved.get("format_version") == 1 and saved["manifest"] == arm["manifest"]
                and saved["cursor"] == entry["cursor"], "Latest checkpoint manifest/cursor differs")
    require(saved["frozen_buffers_sha256"] == entry["frozen_buffers_sha256"]
            and saved["model_audit"]["frozen_buffers_preserved"] is True
            and saved["model_audit"]["frozen_buffers_sha256"] == entry["frozen_buffers_sha256"], "Checkpoint frozen graph differs")
    actual = trainer.parameter_audit(saved["parameters"], arm["manifest"]["initial_parameter_audit"])
    require(actual == saved["parameter_audit"] and all(v["finite"] for v in actual.values()), "Checkpoint parameter hashes/finite audit differs")
    require(all(v.dtype == torch.float32 for v in saved["parameters"].values()), "Checkpoint parameters must be float32")
    return actual


def budget_table(rows, config, max_updates):
    """Reconstruct exact non-padding next-token exposure without tensor/GPU work."""
    table, updates, total = {0: {"tokens": 0, "epoch": 0}}, 0, 0
    require(isinstance(max_updates, int) and max_updates >= 0, "Invalid requested budget")
    if max_updates == 0:
        return table
    for epoch in range(config["epochs"] + config["second_epochs"]):
        for batch, indices in enumerate(trainer.epoch_batches(rows, epoch, config["batch_size"], config["seed"])):
            lengths = [len(rows[i]["ids"]) - 1 for i in indices]
            for offset in range(0, max(lengths), config["chunk_size"]):
                tokens = sum(min(config["chunk_size"], max(0, length - offset)) for length in lengths)
                updates += 1; total += tokens
                table[updates] = {"tokens": total, "chunk_tokens": tokens, "epoch": epoch, "batch": batch, "offset": offset}
                if updates >= max_updates:
                    return table
    require(updates >= max_updates, "Requested budget exceeds planned recipe")
    return table


def verify_training_budget(arm, table):
    rows = arm["training_records"]
    require(rows and len(rows) >= arm["accepted"]["durable_updates"], "Training log omits durable updates")
    for update, row in enumerate(rows, 1):
        require(row["updates"] == update and update in table, "Training updates are not contiguous or exceed observed stop")
        expected = table[update]
        require(all(row[key] == expected[key] for key in ("epoch", "batch", "offset"))
                and row["tokens"] == expected["chunk_tokens"], "Logged data exposure differs from deterministic recipe")
    return {"durable_updates": arm["accepted"]["durable_updates"], "observed_updates": arm["accepted"]["observed_updates"],
            "logged_updates": len(rows), "durable_training_tokens": table[arm["accepted"]["durable_updates"]]["tokens"],
            "logged_training_tokens": table[len(rows)]["tokens"], "verified_every_training_chunk": True}


def compare_scores(e, g):
    delta = q.compare_scores(e, g)
    return {**delta, "direction": "G minus E"}


def compare_history(e, g, table):
    histories = [{row["updates"]: row["validation"] for row in arm["records"]} for arm in (e, g)]
    common = sorted(histories[0].keys() & histories[1].keys())
    require(common and max(common) > 0, "No shared nonzero validation budget")
    rows = [{"updates": step, "training_tokens": table[step]["tokens"], "E": histories[0][step], "G": histories[1][step],
             "comparison": compare_scores(histories[0][step], histories[1][step])} for step in common]
    cap = max(common)
    best = {}
    for label, arm in (("E", e), ("G", g)):
        eligible = [row for row in arm["records"] if row["updates"] <= cap]
        best[label] = {}
        for selector, choose, field in (("minimum_validation_ce", min, "cross_entropy"), ("maximum_validation_accuracy", max, "top1_accuracy")):
            row = choose(eligible, key=lambda r: r["validation"][field])
            best[label][selector] = {"updates": row["updates"], "training_tokens": table[row["updates"]]["tokens"], "validation": row["validation"]}
    return {"common_budget_updates": cap, "common_budget_training_tokens": table[cap]["tokens"],
            "matched_updates": rows, "best_through_common_budget": best,
            "qualification": "Historical scores, not regenerated historical checkpoint texts. Retained winners can use unequal budgets."}


def tensor_difference(base, other):
    require(base.shape == other.shape and base.numel() > 0, "Cannot compare tensor shapes")
    a, b = base.detach().cpu().double().flatten(), other.detach().cpu().double().flatten()
    delta = b - a
    return {"values": delta.numel(), "different_values": int(torch.count_nonzero(delta)),
            "changed_fraction": float((delta != 0).double().mean()), "mean_delta": float(delta.mean()),
            "mean_absolute_delta": float(delta.abs().mean()), "rms_delta": float(delta.square().mean().sqrt()),
            "maximum_absolute_delta": float(delta.abs().max()), "base_rms": float(a.square().mean().sqrt()),
            "other_rms": float(b.square().mean().sqrt()), "relative_l2_delta": float(delta.norm() / a.norm()) if a.norm() > 0 else None}


def neuron_differences(base, other):
    out = {name: tensor_difference(base[name], other[name]) for name in NEURON_PARAMETERS}
    out["gain_times_rec_gain"] = tensor_difference(base["brain.gain"] * base["brain.rec_gain"], other["brain.gain"] * other["brain.rec_gain"])
    return out


def restore_with_initial(reference, arm, saved):
    cfg = SimpleNamespace(**arm["manifest"]["config"])
    reference_hashes = trainer.frozen_hashes(reference)
    require(reference_hashes == arm["manifest"]["groups_provenance"]["frozen_buffers_sha256"], "Reference/group canonical graph differs")
    model = trainer.build_model(reference, cfg, arm["groups"])
    require(trainer.parameter_counts(model) == trainer.expected_parameter_counts(model, cfg, arm["groups"])
            == arm["manifest"]["parameter_counts"], "Actual parameter inventory differs")
    initial = trainer.parameters_cpu(model)
    initial_audit = trainer.parameter_audit(initial)
    require(initial_audit == arm["manifest"]["initial_parameter_audit"], "Reconstructed initialization differs from saved hashes")
    initial_neurons = {name: initial[name].clone() for name in NEURON_PARAMETERS}
    del initial
    require(trainer.frozen_hashes(model) == arm["manifest"]["frozen_buffers_sha256"], "Restored canonical buffers differ")
    trainer.restore_parameters(model, saved["parameters"])
    require(trainer.parameter_audit(trainer.parameters_cpu(model)) == trainer.parameter_audit(saved["parameters"]), "Restored parameter contents differ")
    return model, initial_neurons, initial_audit


def report_markdown(result):
    lines = ["# Rank-64 decoder quality: G versus rank-128 E", "",
             "Sequential MPS inference on idle macm3. Both retained selectors replay all 100 validation stories / 21,874 targets. The reserved test is untouched.", "",
             "| Arm | Selector | Update | Training targets | CE | Accuracy |", "|---|---|---:|---:|---:|---:|"]
    for label, selectors in result["arms"].items():
        for selector, value in selectors.items():
            metric = value["validation"]["summary"]
            lines.append(f"| {label} | {selector} | {value['cursor']['updates']:,} | {value['training_tokens']:,} | {metric['cross_entropy']:.5f} | {metric['top1_accuracy']:.3%} |")
    lines += ["", "## Retained winners: G minus E", ""]
    for selector, difference in result["comparisons"].items():
        lines.append(f"- {selector}: CE {difference['delta_cross_entropy']:+.5f}; accuracy {difference['delta_accuracy'] * 100:+.3f} percentage points.")
    history = result["budget_comparison"]
    lines += ["", f"## Common budget: {history['common_budget_updates']:,} updates / {history['common_budget_training_tokens']:,} training targets", "",
              "| Update | E CE | G CE | E accuracy | G accuracy |", "|---:|---:|---:|---:|---:|"]
    for row in history["matched_updates"]:
        if row['updates'] % 1000 == 0 or row['updates'] == history['common_budget_updates']:
            lines.append(f"| {row['updates']:,} | {row['E']['cross_entropy']:.5f} | {row['G']['cross_entropy']:.5f} | {row['E']['top1_accuracy']:.3%} | {row['G']['top1_accuracy']:.3%} |")
    lines += ["", "Retained winners have their actual budgets above. The JSON also records both best selectors through the common budget; no unavailable historical checkpoint text is inferred.", "",
              "## Unedited greedy continuations", "", "Same six prompts, fresh cache, one BOS, EOS stopping, at most 80 new BPE tokens. Fluency and overlap are descriptive, not a held-out generalization claim.", ""]
    for i, prompt in enumerate(q.PROMPTS):
        lines += [f"### {i + 1}. {prompt}", ""]
        for label, selectors in result["arms"].items():
            for selector, value in selectors.items():
                overlap = value["training_overlap"][i]
                lines += [f"**{label} / {selector}**", "", value["samples"][i]["continuation"], "",
                          f"Longest contiguous training overlap: {overlap['longest_contiguous_match']['words']} whitespace words. Full text is a normalized training prefix: {overlap['full_prompt_plus_continuation_is_normalized_training_prefix']}.", ""]
    lines += ["## Per-neuron parameters", "", "| Selector | Parameter | G−E RMS | G−E max absolute | Changed fraction |", "|---|---|---:|---:|---:|"]
    for selector, differences in result["neuron_comparison"]["G_minus_E"].items():
        for name, values in differences.items():
            lines.append(f"| {selector} | {name} | {values['rms_delta']:.6g} | {values['maximum_absolute_delta']:.6g} | {values['changed_fraction']:.3%} |")
    lines += ["", "The JSON contains each winner’s deltas from its hash-verified initialization. Canonical edge values, connectivity and interfaces are exactly fixed; neuron gains/biases are trainable. These differences do not establish anatomical advantage or compensation causality. One seed, different decoder shapes/RNG consumption, checkpoint-selection validation, and potentially unequal budgets limit interpretation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-e", type=Path, required=True)
    parser.add_argument("--arm-g", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    require(args.threads > 0 and not args.output.exists(), "Positive thread count and a new output directory required")
    host = idle_host_guard()
    # A process lock supplements the repeated live guards; no bound source is edited.
    lock_path = ROOT / "results/.connectorch-decoder-quality.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, host)


def run(args, host):
    torch.set_num_threads(args.threads)
    arms = {"E32rank128fixed": inspect_arm(args.arm_e, "E32rank128fixed"),
            "G32rank64fixed": inspect_arm(args.arm_g, "G32rank64fixed")}
    e, g = arms.values(); verify_pair(e, g)
    worker_refs = [ref for arm in arms.values() for ref in arm["worker_refs"]]
    host = idle_host_guard(worker_refs)
    max_updates = max(max(arm["accepted"]["observed_updates"], len(arm["training_records"])) for arm in arms.values())
    table = budget_table(e["data"]["train"], e["manifest"]["config"], max_updates)
    budget_audits = {label: verify_training_budget(arm, table) for label, arm in arms.items()}
    history = compare_history(e, g, table)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(e["manifest"]["model_path"], trust_remote_code=True, local_files_only=True)
    require((tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (0, 1, 2), "Pinned special-token IDs differ")
    q.validate_tokenized_rows(e["data"]["train"] + e["data"]["validation"], tokenizer)
    bindings = {path: digest for arm in arms.values() for path, digest in arm["bindings"].items()}
    for name in ("scripts/evaluate_connectorch_decoder_quality.py", "scripts/evaluate_connectorch_quality.py",
                 "scripts/evaluate_ngxson_quality.py", "scripts/train_ngxson.py"):
        q.verify_bound(ROOT / name, trainer.file_hash(ROOT / name), bindings)
    args.output.mkdir(parents=True)
    protocol = {"format_version": 1, "created_at_utc": utc(), "training_run": False, "host_guard": host,
                "device": "mps", "torch": torch.__version__, "threads": args.threads,
                "reserved_test_evaluated": False, "validation_used_for_checkpoint_selection": True,
                "generation": {"prompts": q.PROMPTS, "max_new_tokens": 80, "greedy": True, "fresh_cache_per_prompt": True, "bos_once": True, "eos_stopping": True},
                "bindings_sha256": bindings, "training_budget_audits": budget_audits,
                "arms": {label: {"config": arm["manifest"]["config"], "parameter_counts": arm["manifest"]["parameter_counts"],
                    "selectors": arm["selectors"], "acceptance": arm["accepted"], "experiment_git": arm["launch"].get("experiment_git")}
                    for label, arm in arms.items()}}
    trainer.atomic_json(args.output / "protocol.json", protocol)
    outputs, retained_neurons, initial_neurons, initialization = {}, {}, {}, {}
    for label, arm in arms.items():
        idle_host_guard(worker_refs)
        latest = torch.load(arm["latest"]["path"], map_location="cpu", weights_only=False, mmap=True)
        verify_saved(latest, arm); del latest
        outputs[label], retained_neurons[label] = {}, {}
        for selector, entry in arm["selectors"].items():
            idle_host_guard(worker_refs)
            q.verify_bound(entry["path"], entry["sha256"], bindings)
            saved = torch.load(entry["path"], map_location="cpu", weights_only=False, mmap=True)
            verify_saved(saved, arm, selector)
            reference = AutoModelForCausalLM.from_pretrained(arm["manifest"]["model_path"], trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32).cpu()
            model, initial, init_audit = restore_with_initial(reference, arm, saved)
            del reference
            initial_neurons[label] = initial; initialization[label] = init_audit
            retained_neurons[label][selector] = {name: saved["parameters"][name].clone() for name in NEURON_PARAMETERS}
            cursor = dict(saved["cursor"]); del saved
            before = trainer.parameter_audit(trainer.parameters_cpu(model)); graph_before = trainer.frozen_hashes(model)
            model = model.to("mps").eval().requires_grad_(False)
            model.brain.sparse_runtime()  # Version counters must be initialized outside inference_mode.
            model.verify_frozen()
            validation = q.score_validation(model, arm["data"]["validation"], batch_size=COMMON_CONFIG["batch_size"], chunk_size=COMMON_CONFIG["chunk_size"])
            replay = q.verify_replay(validation["summary"], entry["validation"])
            samples = q.generate(model, tokenizer)
            torch.mps.synchronize()
            backend = model.backend_metadata(); counts = backend["sparse_backend"]["kernel_launch_count"]
            require(counts["forward"] > 0 and counts["state_backward"] == counts["edge_backward"] == 0
                    and backend["sparse_backend"]["host_fallback"] is False, "Expected native forward-only Metal execution")
            after = trainer.parameter_audit(trainer.parameters_cpu(model)); graph_after = trainer.frozen_hashes(model)
            require(before == after and graph_before == graph_after, "Evaluation mutated parameters or graph")
            require(all(not v.requires_grad and v.grad is None for v in model.parameters()), "Evaluation created gradients")
            model.verify_frozen()
            outputs[label][selector] = {"cursor": cursor, "training_tokens": table[cursor["updates"]]["tokens"],
                "checkpoint_sha256": entry["sha256"], "validation": validation, "replay": replay, "samples": samples,
                "training_overlap": q.training_overlap(samples, arm["data"]["train"]),
                "parameter_hashes_before": before, "parameter_hashes_after": after,
                "graph_hashes_before": graph_before, "graph_hashes_after": graph_after,
                "neuron_delta_from_initialization": neuron_differences(initial, retained_neurons[label][selector]),
                "initialization_hashes_verified": True, "parameters_and_graph_unchanged": True,
                "zero_backward_kernel_calls": True, "backend": backend}
            trainer.atomic_json(args.output / (label + ".json"), outputs[label])
            trainer.atomic_json(args.output / "status.json", {"status": "running", "completed": label + "/" + selector})
            del model; gc.collect(); torch.mps.empty_cache()
    final_host = idle_host_guard(worker_refs)
    for path, digest in bindings.items():
        q.verify_bound(path, digest, {})
    require(trainer.connectorch_receipt() == e["manifest"]["connectorch_source"], "ConnecTorch changed during audit")
    comparisons = {selector: compare_scores(outputs["E32rank128fixed"][selector]["validation"]["summary"], outputs["G32rank64fixed"][selector]["validation"]["summary"]) for selector in SELECTORS}
    neuron_comparison = {"G_minus_E": {selector: neuron_differences(retained_neurons["E32rank128fixed"][selector], retained_neurons["G32rank64fixed"][selector]) for selector in SELECTORS},
                         "initial_G_minus_E": neuron_differences(initial_neurons["E32rank128fixed"], initial_neurons["G32rank64fixed"]),
                         "initial_parameter_hashes": initialization,
                         "canonical_edges_exactly_fixed": True, "canonical_graph_hashes": e["manifest"]["frozen_buffers_sha256"],
                         "qualification": "Trainable neuron parameters, not rewiring. Retained checkpoints may have unequal budgets; no causal anatomical advantage established."}
    result = {"status": "completed", "protocol": protocol, "arms": outputs, "comparisons": comparisons,
              "budget_comparison": history, "neuron_comparison": neuron_comparison, "final_host_guard": final_host,
              "reserved_test_evaluated": False, "completed_at_utc": utc()}
    trainer.atomic_json(args.output / "results.json", result)
    (args.output / "report.md").write_text(report_markdown(result))
    trainer.atomic_json(args.output / "status.json", {"status": "completed", "comparisons": comparisons})
    print(json.dumps({"completed": True, "output": str(args.output), "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
