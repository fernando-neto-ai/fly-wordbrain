#!/usr/bin/env python3
"""Run the two rank-128 decoder arms serially after accepted B and quality review.

This independent dispatcher leaves the existing trainer and encoder campaigns
unchanged. A supplied, graph-bound CPU/MPS rank-128 preflight must pass before
two eight-update smoke runs; both smokes must pass before either full run.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import sys
import uuid

SELF = Path(__file__).resolve()
ROOT = SELF.parents[1]
spec = importlib.util.spec_from_file_location("decoder_original_campaign", SELF.with_name("run_connectorch_campaign.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ARMS = (("E32rank128fixed", 32, "fixed"), ("F32rank128bounded", 32, "bounded10"))
SELECTORS = {"minimum_validation_ce": "best.pt", "maximum_validation_accuracy": "best-accuracy.pt"}
HOST_NAMES = {"macm3", "fernandos-macbook-pro-2"}
PLANNED_UPDATES = [35877, 16732]
SOURCE_FILES = (*c.SOURCE_FILES, "scripts/run_connectorch_decoder_campaign.py", "scripts/audit_connectorch_decoder.py")


def require(condition, message):
    if not condition:
        raise c.CampaignError(message)


def arm_config(arm, smoke=False):
    require(arm in ARMS, "Unknown decoder arm")
    return {"epochs": 30, "second_epochs": 14, "batch_size": 8, "chunk_size": 32, "seed": 42,
            "d_embed": 32, "plasticity": arm[2], "readout_rank": 128, "history_length": 8,
            "skip_final_test": True, "device": "mps", "eval_interval_updates": 100,
            "max_updates": 8 if smoke else None, "eval_limit": 8 if smoke else None}


def expected_counts(arm):
    result = {"embedding": 32768, "input_projection": 450048, "neurons": 148179,
              "edge_gains": 18322 if arm[2] == "bounded10" else 0, "layernorm": 98786,
              "readout": 128 * (49393 + 1024), "other": 0}
    result["total"] = sum(result.values())
    return result


def process_inventory(text):
    records = []
    for line in text.splitlines():
        fields = line.strip().split(None, 3)
        require(len(fields) == 4, "Could not parse process inventory")
        pid, parent, state, command = fields
        records.append({"pid": int(pid), "ppid": int(parent), "state": state, "command": command})
    return records


def reject_competing_jobs(records, ignored_pids=()):
    blocked = []
    for record in records:
        if record["pid"] in ignored_pids or "Z" in record["state"]:
            continue
        try:
            tokens = shlex.split(record["command"])
        except ValueError:
            tokens = record["command"].split()
        executable = Path(tokens[0]).name.lower() if tokens else ""
        if not (executable.startswith("python") or executable.endswith(".py")):
            continue
        names = {Path(token).name for token in tokens}
        if any(name.endswith(".py") and name.startswith(("train_", "evaluate_", "audit_connectorch", "compare_ngxson_texts", "run_connectorch", "continue_connectorch")) for name in names):
            blocked.append(record)
    require(not blocked, "Competing training/audit/dispatcher process, including suspended controllers: " + json.dumps(blocked))
    return {"no_competing_jobs": True, "ignored_authorized_pids": sorted(ignored_pids)}


def host_guard(manifest=None):
    hostname = socket.gethostname()
    require(platform.system() == "Darwin" and hostname.lower().removesuffix(".local") in HOST_NAMES,
            "Decoder training is restricted to the registered macm3 host")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    require(chip.startswith("Apple M3"), "Expected Apple M3 hardware")
    import torch
    require(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"), "Native Apple MPS shaders required")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", "MPS CPU fallback must be disabled")
    ignored = {os.getpid()}
    launcher = (manifest or {}).get("authorized_launcher", {})
    if c.process_alive(launcher.get("pid"), launcher.get("birth")):
        ignored.add(launcher["pid"])
    processes = process_inventory(subprocess.check_output(["ps", "-axo", "pid=,ppid=,state=,command="], text=True))
    return {"hostname": hostname, "chip": chip, "device": "mps", "checked_at_utc": c.utc(),
            **reject_competing_jobs(processes, ignored)}


def bound(path, digest, bindings):
    path = Path(path).resolve()
    require(path.is_file() and c.file_hash(path) == digest, "Bound artifact missing or changed: " + str(path))
    bindings[str(path)] = digest
    return path


def bound_entry(entry, bindings):
    require(Path(entry["path"]).is_absolute(), "Receipt path must be absolute")
    return bound(entry["path"], entry["sha256"], bindings)


def validation_records(directory):
    # JSONL boundaries are physical LF bytes; Unicode separators can be valid
    # content inside JSON strings and must not split an otherwise valid row.
    rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().split("\n") if line.strip()]
    return [row for row in rows if row.get("event") == "validation"]


def verify_cessation(receipt):
    cessation = receipt.get("process_cessation", {})
    processes = cessation.get("processes", [])
    require(cessation.get("confirmed") is True and bool(processes), "B lacks verified process cessation")
    inventory = {item["pid"]: item for item in process_inventory(subprocess.check_output(["ps", "-axo", "pid=,ppid=,state=,command="], text=True))}
    for process in processes:
        require(isinstance(process.get("pid"), int) and process["pid"] > 0 and process.get("alive") is False,
                "Invalid B process cessation record")
        current = inventory.get(process["pid"])
        same_birth = current is not None and (not process.get("birth") or c.process_birth(process["pid"]) == process["birth"])
        require(not same_birth or "Z" in current["state"], "A recorded B training process is still live")


def baseline_bindings(args):
    directory, bindings = args.baseline_b, {}
    receipt = c.read_json(args.accepted_b_receipt)
    bound(args.accepted_b_receipt, c.file_hash(args.accepted_b_receipt), bindings)
    require(receipt.get("format_version") == 1 and receipt.get("status") == "accepted_early_stop"
            and receipt.get("arm") == "B32fixed"
            and receipt.get("accepted_by") in ("user", "agent_under_standing_user_authorization")
            and receipt.get("full_schedule_completed") is False and receipt.get("test_evaluated") is False,
            "Expected B's accepted early-stop disposition with its test reserved")
    require(args.accepted_b_receipt == directory / "accepted-early-stop.json", "Acceptance belongs to a different B directory")
    for path, digest in receipt["artifact_sha256"].items():
        bound(path, digest, bindings)
    for name in ("manifest.json", "launch.json", "metrics.jsonl", "selected-checkpoints.json"):
        require(str(directory / name) in bindings, "Acceptance omitted B artifact: " + name)
    stop = c.read_json(bound_entry(receipt["stop_receipt"], bindings))
    require(stop.get("process_cessation") == receipt["process_cessation"], "B stop and acceptance cessation records differ")
    verify_cessation(receipt)
    if receipt.get("decision_receipt"):
        bound_entry(receipt["decision_receipt"], bindings)
    baseline = c.read_json(directory / "manifest.json")
    expected = {**arm_config(ARMS[0]), "readout_rank": 0}
    require(all(baseline.get("config", {}).get(key) == value for key, value in expected.items())
            and baseline.get("debug") is False and baseline.get("planned_phase_updates") == PLANNED_UPDATES,
            "B baseline recipe or planned schedule differs")
    require(baseline.get("reference_repository") == "ngxson/fly-llm-hf" and baseline.get("reference_revision") == c.REVISION,
            "B reference identity differs")
    records = validation_records(directory)
    require(bool(records), "B has no validation records")
    selectors = c.verify_selected_checkpoints(directory, records, {"baseline_frozen_buffers_sha256": baseline["frozen_buffers_sha256"]})
    preserved = receipt.get("selected_checkpoints", {})
    require(preserved.get("format_version") == 1 and set(preserved.get("selectors", {})) == set(SELECTORS), "B acceptance lacks both preserved selectors")
    for name, entry in preserved["selectors"].items():
        bound_entry(entry, bindings)
        original = selectors[name]
        bound_entry(original, bindings)
        require(all(entry[key] == original[key] for key in ("sha256", "cursor", "validation", "frozen_buffers_sha256"))
                and entry.get("frozen_buffers_preserved") is True, "Preserved B selector differs: " + name)
    latest = receipt["latest_checkpoint"]
    bound_entry(latest, bindings)
    require(latest.get("frozen_buffers_preserved") is True and latest["frozen_buffers_sha256"] == baseline["frozen_buffers_sha256"], "B latest graph differs")
    require(latest["cursor"]["updates"] == receipt["durable_updates"] == max(row["updates"] for row in records)
            and 0 < receipt["durable_updates"] <= receipt["observed_updates"] < sum(PLANNED_UPDATES)
            and latest["cursor"]["epoch"] == receipt["completed_epochs"] < 44, "B stop counts differ")
    quality = c.read_json(bound(args.quality_results, c.file_hash(args.quality_results), bindings))
    require(quality.get("status") == "completed" and quality.get("reserved_test_evaluated") is False
            and quality.get("protocol", {}).get("training_run") is False
            and quality["protocol"].get("device") == "mps", "B quality audit is incomplete or scored its reserved test")
    require(quality["protocol"]["arms"]["B32fixed"]["directory"] == str(directory), "Quality audit belongs to another B run")
    for path in (directory / "manifest.json", directory / "selected-checkpoints.json", directory / "metrics.jsonl"):
        require(quality["protocol"]["bindings_sha256"].get(str(path)) == bindings[str(path)], "Quality audit used different B artifacts")
    for name, entry in selectors.items():
        measured = quality["arms"]["B32fixed"][name]
        score = measured["validation"]["summary"]
        require(measured["checkpoint_sha256"] == entry["sha256"] and measured["cursor"] == entry["cursor"]
                and measured.get("parameters_and_graph_unchanged") is True and measured.get("zero_backward_kernel_calls") is True
                and measured.get("replay", {}).get("passed") is True
                and measured["graph_hashes_before"] == measured["graph_hashes_after"] == baseline["frozen_buffers_sha256"]
                and all(score[key] == entry["validation"][key] for key in ("stories", "tokens", "correct"))
                and abs(score["cross_entropy"] - entry["validation"]["cross_entropy"]) < 1e-4,
                "B quality selector failed identity/replay checks: " + name)
    summary = {"name": "B32fixed", "status": "accepted_early_stop", "output": str(directory),
               "updates": receipt["durable_updates"], "observed_updates": receipt["observed_updates"], "epochs": receipt["completed_epochs"],
               "full_schedule_completed": False, "test_evaluated": False,
               "selected_checkpoints": selectors, "parameter_counts": baseline["parameter_counts"]}
    return baseline, receipt, summary, bindings


def verify_preflight(parity, manifest):
    """Rank-0 evidence cannot satisfy the decoder-specific preflight."""
    require(parity.get("passed") is True and parity.get("sources_unchanged") is True
            and parity.get("inputs_unchanged") is True and parity.get("connectorch_unchanged") is True
            and parity.get("exact_ir", {}).get("passed") is True and parity.get("final_host_guard", {}).get("passed") is True
            and parity.get("data_file_sha256") == manifest["data_sha256"]
            and parity.get("groups_file_sha256") == manifest["groups_sha256"], "Rank128 parity failed or input hashes differ")
    for arm in ARMS:
        name = arm[0]
        case = parity.get("arms", {}).get(name, {})
        require(case.get("d_embed") == 32 and case.get("readout_rank") == 128
                and case.get("history_length") == 8 and case.get("seed") == 42 and case.get("plasticity") == arm[2]
                and case.get("parameter_counts") == expected_counts(arm), "Missing or incorrect rank128 preflight arm: " + name)
        comparison = parity.get("comparisons", {}).get(name + "_cpu_vs_mps", {})
        require(comparison.get("passed") is True and comparison.get("loss_abs_difference", math.inf) < 1e-4
                and comparison.get("max_gradient_relative_l2", math.inf) < 1e-3
                and comparison.get("cache_ids_and_length_equal") is True, "Rank128 CPU/MPS parity tolerance failed: " + name)
        for device in ("cpu", "mps"):
            audit = parity.get("cases", {}).get(name + "_" + device, {})
            require(audit.get("passed") is True and audit.get("device") == device and audit.get("targets") == 64
                    and audit.get("parameter_count") == expected_counts(arm)["total"]
                    and audit.get("frozen_buffers_unchanged") is True
                    and audit.get("frozen_buffers_sha256") == manifest["baseline_frozen_buffers_sha256"]
                    and audit.get("padding_gradient_zero") is True and audit.get("native_kernel_gate") is True,
                    "Rank128 preflight omitted graph/gradient/kernel evidence: " + name + "/" + device)
            finite = audit.get("parameters_and_gradients_finite", {})
            require(bool(finite) and all(finite.values()) and all(finite.get(key) is True for key in ("lm_head.0.weight", "lm_head.1.weight")), "Rank128 readout gradient missing")
            if arm[2] == "bounded10":
                gradients = audit.get("factorized_gain_gradients", {})
                require(all(gradients.get(key, {}).get("nonzero", 0) > 0 for key in ("brain.edge_theta_source", "brain.edge_theta_destination")), "Bounded gain gradient missing in preflight")
    initialization = parity.get("initialization_equivalence", {})
    require(initialization.get("passed") is True and initialization.get("shared_parameter_hashes_equal") is True
            and initialization.get("zero_adaptive_parameters") is True and initialization.get("logits_equal") is True,
            "Fixed/bounded rank128 arms did not start from equivalent shared weights and outputs")
    require(parity.get("connectorch") == manifest["baseline_connectorch_source"] and bool(parity.get("source_sha256")), "Parity installed package/source receipt differs")
    for relative, digest in parity["source_sha256"].items():
        require(manifest["sources_sha256"].get(str(Path(manifest["root"]) / relative)) == digest, "Preflight used unbound source: " + relative)


def prepare(args):
    host = host_guard()
    baseline, acceptance, summary, baseline_files = baseline_bindings(args)
    require(set(baseline["trainer_sources_sha256"]) == c.TRAINER_SOURCE_FILES, "B training source inventory differs")
    sources = {str(args.root / name): c.file_hash(args.root / name) for name in SOURCE_FILES}
    for name, digest in baseline["trainer_sources_sha256"].items():
        require(sources.get(str(args.root / name)) == digest, "Current trainer differs from accepted B: " + name)
    inputs = {}
    for key in ("data", "groups"):
        bound(baseline[key + "_path"], baseline[key + "_sha256"], inputs)
    for name, receipt in baseline["reference_files"].items():
        bound(Path(baseline["model_path"]) / name, receipt["sha256"], inputs)
    dataset = c.read_json(baseline["data_path"])
    require(len(dataset["train"]) == 1000 and len(dataset["validation"]) == 100
            and sum(len(row["ids"]) - 1 for row in dataset["train"]) == 234796
            and sum(len(row["ids"]) - 1 for row in dataset["validation"]) == 21874, "Expected full reconstructed data split")
    branches = c.read_json(bound(args.experiment_branches, c.file_hash(args.experiment_branches), inputs))
    require(branches.get("repository") == "https://github.com/fernando-neto-ai/fly-wordbrain"
            and set(branches.get("arms", {})) == {arm[0] for arm in ARMS}, "Branch mapping must name exactly E/F")
    for entry in branches["arms"].values():
        commit = entry.get("commit", "")
        require(bool(entry.get("branch")) and len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit), "Invalid branch or commit provenance")
    configs = {}
    for arm in ARMS:
        path = args.root / "experiments/configs" / (arm[0] + ".json")
        bound(path, c.file_hash(path), inputs)
        declared = c.read_json(path)
        require(declared.get("experiment_id") == arm[0] and declared.get("branch") == branches["arms"][arm[0]]["branch"]
                and declared.get("training_host") == "macm3" and declared.get("trainer") == "scripts/train_connectorch.py"
                and declared.get("trainable_parameters") == expected_counts(arm)["total"]
                and declared.get("dataset_sha256") == baseline["data_sha256"] and declared.get("groups_sha256") == baseline["groups_sha256"]
                and all(declared.get("training", {}).get(key) == value for key, value in arm_config(arm).items()),
                "Branch experiment configuration differs from planned decoder command: " + arm[0])
        configs[arm[0]] = {"path": str(path), "sha256": inputs[str(path)]}
    parity_path = bound(args.preflight_receipt, c.file_hash(args.preflight_receipt), inputs)
    parity = c.read_json(parity_path)
    for relative, digest in parity.get("source_sha256", {}).items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Preflight source must be repository relative")
        bound(args.root / path, digest, sources)
    manifest = {"format_version": 1, "launch_id": str(uuid.uuid4()), "created_at_utc": c.utc(),
        "root": str(args.root), "output": str(args.output), "python": args.python, "environment": c.ENVIRONMENT,
        "authorized_launcher": {"pid": os.getpid(), "birth": c.process_birth(os.getpid())}, "host_guard": host,
        "arms": [{"name": arm[0], **arm_config(arm), "parameter_counts": expected_counts(arm)} for arm in ARMS],
        "model": baseline["model_path"], "data": baseline["data_path"], "groups": baseline["groups_path"],
        "data_sha256": baseline["data_sha256"], "groups_sha256": baseline["groups_sha256"],
        "baseline_frozen_buffers_sha256": baseline["frozen_buffers_sha256"], "baseline_connectorch_source": baseline["connectorch_source"],
        "baseline_run": str(args.baseline_b), "baseline_acceptance": {"path": str(args.accepted_b_receipt), "sha256": baseline_files[str(args.accepted_b_receipt)]},
        "baseline_selected_checkpoints": summary["selected_checkpoints"], "baseline_summary": summary,
        "baseline_quality": {"path": str(args.quality_results), "sha256": baseline_files[str(args.quality_results)]},
        "preflight_receipt": {"path": str(parity_path), "sha256": inputs[str(parity_path)]},
        "preflight_connectorch_source": parity.get("connectorch"), "experiment_branches": branches, "experiment_configs": configs,
        "sources_sha256": sources, "input_files_sha256": inputs, "baseline_files_sha256": baseline_files,
        "planned_phase_updates": PLANNED_UPDATES, "validation_stories": 100, "validation_tokens": 21874,
        "smoke_validation_stories": 8, "smoke_validation_tokens": sum(len(row["ids"]) - 1 for row in dataset["validation"][:8]),
        "initialization": "from scratch with seed42; accepted B is a comparison baseline, not resumed weights",
        "stop_policy": "An authorized plateau stop preserves both selectors and latest, halts this dispatcher, and requires a deliberate receipt-verified handoff for remaining arms. No automatic SIGTERM recovery or restart.",
        "final_test_deferred": True, "baseline_full_schedule_completed": False}
    verify_preflight(parity, manifest)
    c.verify_bindings(manifest)
    return manifest


def trainer_command(manifest, arm, output, smoke=False):
    config = arm_config(arm, smoke)
    command = [manifest["python"], "-u", str(Path(manifest["root"]) / "scripts/train_connectorch.py"),
               "--model", manifest["model"], "--data", manifest["data"], "--groups", manifest["groups"], "--output", str(output)]
    for name, value in config.items():
        if value is None:
            continue
        command.append("--" + name.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    return command


def verify_smoke_parameter_updates(directory, recorded, plasticity):
    """Verify both low-rank factors actually receive gradients and change."""
    import torch
    spec = importlib.util.spec_from_file_location("decoder_train_helpers", SELF.with_name("train_connectorch.py"))
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    gradients = c.read_json(directory / "initial-gradients.json")
    names = ["lm_head.0.weight", "lm_head.1.weight"]
    if plasticity == "bounded10":
        names.extend(("brain.edge_theta_source", "brain.edge_theta_destination"))
    for name in names:
        info = gradients.get(name, {})
        require(info.get("present") is True and info.get("finite") is True and info.get("nonzero_values", 0) > 0, "Smoke gradient missing, nonfinite or zero: " + name)
    checkpoint = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
    require(checkpoint["manifest"] == recorded and checkpoint["cursor"]["updates"] == 8, "Smoke latest checkpoint identity differs")
    audit = trainer.parameter_audit(checkpoint["parameters"], recorded["initial_parameter_audit"])
    require(audit == checkpoint["parameter_audit"], "Smoke parameter audit differs from actual values")
    require(all(audit.get(name, {}).get("changed_from_initialization") is True for name in names), "Low-rank or gain parameters did not actually update")
    return {name: {"gradient_nonzero_values": gradients[name]["nonzero_values"], "changed": True} for name in names}


def verify_arm(manifest, arm, directory, smoke=False):
    directory = Path(directory)
    name, width, plasticity = arm
    require(arm in ARMS and not (directory / "failure.json").exists(), "Unknown or failed decoder arm")
    recorded, status = c.read_json(directory / "manifest.json"), c.read_json(directory / "status.json")
    process = c.read_json(directory / "process-status.json")
    require(process.get("status") == "completed" and process.get("exit_code") == 0, "Decoder worker did not exit successfully")
    require(all(recorded.get("config", {}).get(key) == value for key, value in arm_config(arm, smoke).items()), "Decoder arm configuration differs")
    require(recorded.get("parameter_counts") == expected_counts(arm), "Decoder parameter counts differ")
    require(recorded.get("data_sha256") == manifest["data_sha256"] and recorded.get("groups_sha256") == manifest["groups_sha256"], "Decoder data/group identity differs")
    require(recorded.get("planned_phase_updates") == manifest["planned_phase_updates"], "Decoder planned updates differ")
    require(set(recorded.get("trainer_sources_sha256", {})) == c.TRAINER_SOURCE_FILES, "Decoder trainer source inventory differs")
    for relative, digest in recorded["trainer_sources_sha256"].items():
        require(manifest["sources_sha256"].get(str(Path(manifest["root"]) / relative)) == digest, "Decoder used unbound source: " + relative)
    require(recorded.get("connectorch_source") == manifest["preflight_connectorch_source"], "Decoder installed package differs")
    records = validation_records(directory)
    require(bool(records), "Decoder has no validation records")
    for record in records:
        metric = record["validation"]
        require(metric.get("stories") == manifest["smoke_validation_stories" if smoke else "validation_stories"]
                and metric.get("tokens") == manifest["smoke_validation_tokens" if smoke else "validation_tokens"], "Decoder validation coverage differs")
        require(math.isfinite(metric["cross_entropy"]) and metric["cross_entropy"] >= 0
                and 0 <= metric.get("correct", -1) <= metric["tokens"]
                and abs(metric["top1_accuracy"] - metric["correct"] / metric["tokens"]) < 1e-12, "Invalid decoder validation metrics")
        c.graph_audit(record["model_audit"], manifest)
    retained = c.verify_selected_checkpoints(directory, records, manifest)
    if smoke:
        require(status.get("status") == "debug_stopped" and status.get("debug") is True
                and status.get("updates") == records[-1]["updates"] == 8, "Decoder smoke did not stop at exactly eight updates")
        require(status.get("test_evaluated") is False and not (directory / "results.json").exists(), "Decoder smoke accessed final testing")
        updates = verify_smoke_parameter_updates(directory, recorded, plasticity)
        adaptation = records[-1]["model_audit"]["adaptation"]
        if plasticity == "bounded10":
            require(adaptation.get("edge_displacement", {}).get("rms", 0) > 0
                    and all(adaptation.get(key, {}).get("rms", 0) > 0 for key in ("source_type_theta", "destination_type_theta")), "Bounded decoder smoke did not alter effective edges")
        return {"name": name, "passed": True, "updates": 8, "validation": records[-1]["validation"], "parameter_updates": updates,
                "parameter_counts": recorded["parameter_counts"], "selected_checkpoints": retained}
    result = c.read_json(directory / "results.json")
    require(status.get("status") == result.get("status") == "completed" and result.get("debug") is False
            and result.get("epochs") == 44 and result.get("updates") == sum(PLANNED_UPDATES), "Decoder full schedule is incomplete")
    require(result.get("test_evaluated") is False and result.get("test_deferred") is True and "test" not in result, "Decoder accessed its reserved test")
    c.graph_audit(result["checks"], manifest)
    c.graph_audit(result["checks"]["selected_checkpoint_audit"], manifest)
    for selector, result_key in (("minimum_validation_ce", "selected"), ("maximum_validation_accuracy", "selected_accuracy")):
        entry = retained[selector]
        require(result[result_key]["updates"] == entry["cursor"]["updates"]
                and all(result[result_key][key] == entry["validation"][key] for key in ("cross_entropy", "top1_accuracy")), "Decoder final selector differs")
    require(result.get("selected_checkpoints") == retained, "Decoder final selector receipts differ")
    return {"name": name, "d_embed": width, "readout_rank": 128, "history_length": 8, "plasticity": plasticity,
            "updates": result["updates"], "status": "completed", "full_schedule_completed": True,
            "selected_checkpoints": retained, "validation": retained["minimum_validation_ce"]["validation"],
            "accuracy_validation": retained["maximum_validation_accuracy"]["validation"], "parameter_counts": recorded["parameter_counts"],
            "test_evaluated": False, "output": str(directory), "elapsed_seconds": result.get("elapsed_seconds_this_invocation")}


def launch(args):
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with c.lock_file(args.output / "launch.lock"):
            if (args.output / "manifest.json").exists():
                status = c.read_json(args.output / "campaign-status.json")
                launched = c.read_json(args.output / "launch.json")
                if status.get("status") in ("starting", "running") and c.process_alive(launched.get("worker_pid"), launched.get("process_birth")):
                    return {"status": "already_running", "output": str(args.output)}
                return {"status": "completed" if status.get("status") == "completed" else "blocked", "reason": "Existing decoder campaign is never automatically restarted", "output": str(args.output)}
            manifest = prepare(args)
            c.atomic_json(args.output / "manifest.json", manifest)
            argv = [args.python, "-u", str(SELF), "--root", str(args.root), "--baseline-b", str(args.baseline_b),
                    "--accepted-b-receipt", str(args.accepted_b_receipt), "--quality-results", str(args.quality_results),
                    "--preflight-receipt", str(args.preflight_receipt), "--experiment-branches", str(args.experiment_branches),
                    "--output", str(args.output), "--python", args.python, "--worker", "--launch-id", manifest["launch_id"]]
            record = {"command": argv, "cwd": str(args.root), "environment": c.ENVIRONMENT, "created_at_utc": c.utc(),
                      "launch_id": manifest["launch_id"], "manifest_sha256": c.file_hash(args.output / "manifest.json")}
            c.campaign_status(args.output, status="starting", phase="dispatch", launch_id=manifest["launch_id"])
            c.atomic_json(args.output / "launch.json", record)
            with (args.output / "campaign.log").open("a") as log:
                child = subprocess.Popen(argv, cwd=args.root, env={**os.environ, **c.ENVIRONMENT}, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(worker_pid=child.pid, process_birth=c.process_birth(child.pid))
            c.atomic_json(args.output / "launch.json", record)
            return {"status": "launched", "worker_pid": child.pid, "output": str(args.output)}
    except BlockingIOError:
        return {"status": "already_dispatching"}
    except Exception as error:
        report = {"status": "blocked", "error": str(error), "type": type(error).__name__}
        c.atomic_json(args.output / "readiness.json", report)
        return report


def worker(args):
    try:
        with c.lock_file(args.output / "worker.lock"):
            manifest = c.read_json(args.output / "manifest.json")
            launch_record = c.read_json(args.output / "launch.json")
            require(args.launch_id == manifest["launch_id"] and c.read_json(args.output / "campaign-status.json").get("status") == "starting"
                    and c.file_hash(args.output / "manifest.json") == launch_record["manifest_sha256"], "Worker does not match the pending decoder launch")
            c.verify_bindings(manifest)
            host_guard(manifest)
            verify_cessation(c.read_json(manifest["baseline_acceptance"]["path"]))
            previous = {}
            def stop(signum, frame):
                raise c.CampaignError("Decoder campaign interrupted by signal " + str(signum))
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, stop)
            awake = c.start_awake()
            try:
                c.campaign_status(args.output, status="running", phase="preflight", worker_pid=os.getpid())
                parity = c.read_json(manifest["preflight_receipt"]["path"])
                verify_preflight(parity, manifest)
                (args.output / "preflight").mkdir()
                c.atomic_json(args.output / "preflight/parity.json", parity)
                smokes = []
                for arm in ARMS:
                    host_guard(manifest)
                    directory = args.output / "smokes" / arm[0]
                    c.execute_job(manifest, "smoke", arm[0], directory, trainer_command(manifest, arm, directory, smoke=True))
                    smokes.append(verify_arm(manifest, arm, directory, smoke=True))
                c.atomic_json(args.output / "smoke-results.json", {"passed": True, "arms": smokes})
                completed = []
                for arm in ARMS:
                    host_guard(manifest)
                    directory = args.output / "arms" / arm[0]
                    c.execute_job(manifest, "training", arm[0], directory, trainer_command(manifest, arm, directory))
                    completed.append(verify_arm(manifest, arm, directory))
                    c.atomic_json(args.output / "partial-results.json", {"completed_arms": completed, "baseline": manifest["baseline_summary"], "final_test_deferred": True})
                c.verify_bindings(manifest)
                result = {"status": "completed", "arms": completed, "baseline": manifest["baseline_summary"],
                          "preflight_passed": True, "smoke_checks_passed": True, "test_evaluated": False,
                          "finished_at_utc": c.utc(), "qualification": "E/F use equal full schedules; accepted B was early-stopped. One seed, no randomized-connectome control."}
                c.atomic_json(args.output / "results.json", result)
                c.campaign_status(args.output, status="completed", phase="finished", completed_arms=2, test_evaluated=False)
                return result
            finally:
                if awake is not None:
                    awake.terminate()
                    awake.wait()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    except BlockingIOError:
        return {"status": "already_running"}
    except BaseException as error:
        previous = c.read_json(args.output / "campaign-status.json") if (args.output / "campaign-status.json").exists() else {}
        failure = {"status": "failed", "error": str(error), "type": type(error).__name__, "phase": previous.get("phase"), "arm": previous.get("arm"), "finished_at_utc": c.utc()}
        c.atomic_json(args.output / "failure.json", failure)
        c.campaign_status(args.output, **failure)
        return failure


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--baseline-b", type=Path, required=True)
    p.add_argument("--accepted-b-receipt", type=Path, required=True)
    p.add_argument("--quality-results", type=Path, required=True)
    p.add_argument("--preflight-receipt", type=Path, required=True)
    p.add_argument("--experiment-branches", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/connectorch-decoder-v1"))
    p.add_argument("--python", default=sys.executable)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--worker", action="store_true")
    p.add_argument("--launch-id", help=argparse.SUPPRESS)
    return p


def normalized_args(args):
    for name in ("root", "baseline_b", "accepted_b_receipt", "quality_results", "preflight_receipt", "experiment_branches"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output = (args.root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    # Keep the venv symlink path, not its system-Python target.
    args.python = str(Path(args.python).expanduser().absolute())
    require(args.root == ROOT and args.output != args.baseline_b and args.output not in args.baseline_b.parents,
            "Use this staged root and a distinct decoder campaign output")
    return args


def main():
    args = normalized_args(parser().parse_args())
    result = launch(args) if args.launch else worker(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    if result["status"] in ("failed", "blocked"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
