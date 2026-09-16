#!/usr/bin/env python3
"""Inspect H's plateau; --execute stops its exact dispatcher and preserves state.

This one-launch helper never starts another run or performs model inference.
The default is read-only. A signal is permitted only in a coherent, freshly
verified training interval; native failure records remain intact after stopping.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import shlex
import signal
import socket
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_connectorch_campaign as c

REMOTE_ROOT = Path("/Users/fernando/fly_wordbrain_connectorch")
RUN_RELATIVE = "results/connectorch-rank32-v1"
ARM_NAME = "H32rank32fixed"
LAUNCH_ID = "2556b84a-d9f9-49c4-83ee-bc6cd7434258"
MANIFEST_SHA256 = "880b2e9cf957c7b16607a118f7cd058caebab15d7de509d131de5fb20eaadaac"
CONTROLLER, TRAINER = 85176, 85584
CONTROLLER_BIRTH = "Tue Sep 15 21:57:06 2026"
TRAINER_BIRTH = "Tue Sep 15 21:57:20 2026"
SELECTORS = {"minimum_validation_ce": "best.pt", "maximum_validation_accuracy": "best-accuracy.pt"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_launch_bound():
    """Placeholders must never turn into a process or artifact lookup."""
    require(isinstance(LAUNCH_ID, str) and len(LAUNCH_ID) == 36
            and isinstance(MANIFEST_SHA256, str) and len(MANIFEST_SHA256) == 64
            and all(value in '0123456789abcdef' for value in MANIFEST_SHA256)
            and type(CONTROLLER) is int and CONTROLLER > 0
            and type(TRAINER) is int and TRAINER > 0 and CONTROLLER != TRAINER
            and isinstance(CONTROLLER_BIRTH, str) and bool(CONTROLLER_BIRTH)
            and isinstance(TRAINER_BIRTH, str) and bool(TRAINER_BIRTH),
            "H native launch bindings are pending; no inspection or signal is permitted")


def ps(pid, field):
    result = subprocess.run(["ps", "-p", str(pid), "-o", field + "="], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def exited(pid, birth):
    state = ps(pid, "state")
    return state is None or "Z" in state or ps(pid, "lstart") != birth


def host_guard():
    require(ROOT == REMOTE_ROOT and platform.system() == "Darwin", "Run this helper from the native macm3 staged root")
    hostname = socket.gethostname().lower().removesuffix(".local")
    require(hostname in ("macm3", "fernandos-macbook-pro-2"), "Unexpected stop host")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    require(chip.startswith("Apple M3"), "Expected native Apple M3 host")
    return {"hostname": hostname, "chip": chip, "checked_at_utc": c.utc(), "gpu_operations": False}


def verify_identity(run, arm, launch, process):
    require(launch.get("launch_id") == LAUNCH_ID and launch.get("worker_pid") == CONTROLLER
            and launch.get("process_birth") == CONTROLLER_BIRTH, "Dispatcher launch identity differs")
    require(process.get("worker_pid") == TRAINER and process.get("runner_pid") == CONTROLLER
            and process.get("status") == "running", "Trainer launch identity/status differs")
    require(ps(CONTROLLER, "lstart") == CONTROLLER_BIRTH and ps(TRAINER, "lstart") == TRAINER_BIRTH,
            "Exact process birth identity no longer matches")
    require(ps(TRAINER, "ppid") == str(CONTROLLER), "Trainer parent is not the expected dispatcher")
    require(not any(flag in (ps(CONTROLLER, "state") or "") + (ps(TRAINER, "state") or "") for flag in ("T", "Z")),
            "Processes must be running, not suspended or exited")
    for pid, command in ((CONTROLLER, launch["command"]), (TRAINER, process["command"])):
        observed = shlex.split(ps(pid, "command") or "")
        require(observed[1:] == command[1:], "Exact process arguments changed: " + str(pid))
    require("run_connectorch_rank32_campaign.py" in launch["command"][2]
            and LAUNCH_ID in launch["command"] and str(run) in launch["command"], "Unexpected controller command")
    command = process["command"]
    require(str(arm) in command and command[command.index("--readout-rank") + 1] == "32"
            and command[command.index("--d-embed") + 1] == "32"
            and command[command.index("--plasticity") + 1] == "fixed", "Unexpected H trainer architecture")
    state = c.read_json(run / "campaign-status.json")
    require(state.get("status") == "running" and state.get("phase") == "training"
            and state.get("arm") == ARM_NAME and state.get("child_pid") == TRAINER,
            "Campaign is not running the expected H training job")


def validation_rows(arm, manifest):
    rows = [json.loads(line) for line in (arm / "metrics.jsonl").read_text().split("\n") if line.strip()]
    rows = [row for row in rows if row.get("event") == "validation"]
    require(bool(rows) and all(left["updates"] <= right["updates"] for left, right in zip(rows, rows[1:])),
            "Validation history is empty or out of order")
    for row in rows:
        metric = row["validation"]
        require(metric.get("stories") == 100 and metric.get("tokens") == 21874
                and 0 <= metric.get("correct", -1) <= 21874
                and math.isfinite(metric["cross_entropy"]) and metric["cross_entropy"] >= 0
                and abs(metric["top1_accuracy"] - metric["correct"] / 21874) < 1e-12,
                "Invalid full-validation coverage or metric")
        c.graph_audit(row["model_audit"], manifest)
    return rows


def plateau(rows, status):
    update = rows[-1]["updates"]
    previous = [row for row in rows if row["updates"] <= update - 2000]
    epochs = [row for row in rows if row.get("reason") == "epoch_end"]
    completed = max((row["epoch"] for row in epochs), default=0)
    require(status["epoch"] == completed, "Live completed epoch count differs from durable epoch history")
    require(previous and completed >= 8, "Plateau requires at least eight completed epochs and a 2,000-update history")
    best_ce = lambda items: min(row["validation"]["cross_entropy"] for row in items)
    best_acc = lambda items: max(row["validation"]["top1_accuracy"] for row in items)
    ce_gain, accuracy_gain = best_ce(previous) - best_ce(rows), 100 * (best_acc(rows) - best_acc(previous))
    require(0 <= ce_gain < .10 and 0 <= accuracy_gain < .5,
            "Plateau no longer met: CE gain=" + str(ce_gain) + ", accuracy pp gain=" + str(accuracy_gain))
    windows = []
    for low, high in ((update - 2000, update - 1000), (update - 1000, update)):
        items = [row for row in rows if low < row["updates"] <= high and row.get("reason") == "update_interval"]
        require(len(items) == 10, "Expected ten periodic validations per comparison window")
        windows.append({"start_exclusive": low, "end_inclusive": high, "count": len(items),
                        "mean_ce": statistics.mean(row["validation"]["cross_entropy"] for row in items),
                        "median_ce": statistics.median(row["validation"]["cross_entropy"] for row in items),
                        "mean_accuracy": statistics.mean(row["validation"]["top1_accuracy"] for row in items)})
    return {"decision_at_utc": c.utc(), "decision_validation_update": update, "trailing_updates": 2000,
            "minimum_ce_improvement": ce_gain, "maximum_accuracy_improvement_percentage_points": accuracy_gain,
            "completed_epochs": completed, "criterion": {"minimum_completed_epochs": 8,
                "minimum_ce_improvement": .10, "maximum_accuracy_improvement_percentage_points": .5},
            "validation_windows": windows,
            "last_completed_epoch_training": [{"epoch": row["epoch"], **row["training_epoch_so_far"]} for row in epochs[-3:]],
            "qualification": "Operational diminishing returns; later training or scheduled learning-rate changes could still improve results."}


def safe_interval(status, last_update):
    return (type(TRAINER) is int and TRAINER > 0
            and status.get("status") == "running" and status.get("pid") == TRAINER and status.get("activity") == "training"
            and 5 <= status["updates"] % 100 <= 45 and status["batch"] < 115
            and 0 <= status["updates"] - last_update <= 45)


def not_ready(reason, observed):
    print(json.dumps({"status": "not_ready_no_signal", "reason": reason, "observed": observed}), flush=True)


def load_and_verify(path, recorded, manifest, expected_sha=None, entry=None, selector=None, latest_update=None):
    import torch
    import train_connectorch as trainer
    require(not path.is_symlink(), "Checkpoint must not be a symlink")
    with path.open("rb") as handle:
        before_stat = os.fstat(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        require(expected_sha is None or digest == expected_sha, "Checkpoint byte hash differs: " + str(path))
        handle.seek(0)
        saved = torch.load(handle, map_location="cpu", weights_only=False)
        handle.seek(0)
        require(hashlib.file_digest(handle, "sha256").hexdigest() == digest, "Checkpoint contents changed during same-file load")
        current_stat = path.stat()
        require((before_stat.st_dev, before_stat.st_ino) == (current_stat.st_dev, current_stat.st_ino),
                "Checkpoint was atomically replaced while verifying it")
    require(saved.get("format_version") == 1 and saved["manifest"] == recorded, "Checkpoint manifest differs")
    require(saved["frozen_buffers_sha256"] == recorded["frozen_buffers_sha256"], "Checkpoint frozen graph differs")
    c.graph_audit(saved["model_audit"], manifest)
    audit = trainer.parameter_audit(saved["parameters"], recorded["initial_parameter_audit"])
    require(saved["parameter_audit"] == audit and set(audit) == set(recorded["initial_parameter_audit"]),
            "Checkpoint parameter audit or inventory differs")
    require(sum(value.numel() for value in saved["parameters"].values()) == 2343125
            and all(value.device.type == "cpu" and value.dtype == torch.float32 for value in saved["parameters"].values())
            and list(saved["parameters"]["lm_head.0.weight"].shape) == [32, 49393]
            and list(saved["parameters"]["lm_head.1.weight"].shape) == [1024, 32], "Unexpected checkpoint parameter architecture")
    require(all(key in saved for key in ("optimizer", "scheduler", "rng", "cache"))
            and saved["optimizer"].get("state") and saved["optimizer"].get("param_groups")
            and saved["scheduler"] and all(key in saved["rng"] for key in ("python", "torch_cpu", "torch_mps")),
            "Checkpoint omitted resumable optimizer/scheduler/RNG/cache state")
    if entry is not None:
        require(saved["cursor"] == entry["cursor"] and saved["frozen_buffers_sha256"] == entry["frozen_buffers_sha256"],
                "Checkpoint cursor or graph differs from selector")
        best = saved["best" if selector == "minimum_validation_ce" else "best_accuracy"]
        require(all(best[key] == entry["cursor"][key] for key in ("epoch", "updates"))
                and all(best[key] == entry["validation"][key] for key in ("cross_entropy", "top1_accuracy")),
                "Checkpoint does not contain its declared winner")
    if latest_update is not None:
        require(saved["cursor"]["updates"] == latest_update, "Latest checkpoint is not the latest durable validation")
    return {"path": str(path), "sha256": digest, "cursor": saved["cursor"],
            "frozen_buffers_sha256": saved["frozen_buffers_sha256"], "frozen_buffers_preserved": True,
            "optimizer_rng_and_cache_retained": True, "checkpoint_contents_verified": True}


def checkpoint_set(arm, rows, manifest, recorded):
    selectors = c.verify_selected_checkpoints(arm, rows, manifest)
    require(set(selectors) == set(SELECTORS), "Both native selectors are required")
    verified = {}
    for name, entry in selectors.items():
        verified[name] = load_and_verify(arm / SELECTORS[name], recorded, manifest, entry["sha256"], entry, name)
    verified["latest"] = load_and_verify(arm / "latest.pt", recorded, manifest, latest_update=rows[-1]["updates"])
    return copy.deepcopy(selectors), verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send one SIGTERM to the verified dispatcher and preserve the stopped run")
    args = parser.parse_args()
    require_launch_bound()
    host = host_guard()
    run, arm = ROOT / RUN_RELATIVE, ROOT / RUN_RELATIVE / "arms" / ARM_NAME
    require(not (arm / "accepted-early-stop.json").exists() and not (arm / "stopped-checkpoints").exists(),
            "A stop/preservation already exists; inspect it rather than signaling again")
    require(c.file_hash(run / "manifest.json") == MANIFEST_SHA256, "Bound rank32 campaign manifest changed")
    manifest, launch, process = (c.read_json(path) for path in
        (run / "manifest.json", run / "launch.json", arm / "process-status.json"))
    require(manifest["launch_id"] == LAUNCH_ID and manifest["root"] == str(ROOT) and manifest["output"] == str(run),
            "Campaign manifest identity differs")
    require(launch["manifest_sha256"] == MANIFEST_SHA256, "Launch manifest hash differs")
    verify_identity(run, arm, launch, process)
    c.verify_bindings(manifest)
    recorded = c.read_json(arm / "manifest.json")
    expected = {"d_embed": 32, "readout_rank": 32, "history_length": 8, "plasticity": "fixed", "seed": 42,
                "device": "mps", "skip_final_test": True, "max_updates": None, "eval_limit": None}
    require(all(recorded["config"].get(key) == value for key, value in expected.items())
            and recorded["parameter_counts"]["total"] == 2343125
            and recorded["parameter_counts"]["readout"] == 1613344 and recorded.get("debug") is False,
            "H recorded configuration differs")
    import train_connectorch as trainer
    require(trainer.source_receipt() == recorded["trainer_sources_sha256"]
            and trainer.connectorch_receipt() == recorded["connectorch_source"], "Training sources or package changed")
    observed = c.read_json(arm / "status.json")
    if not safe_interval(observed, observed["updates"] // 100 * 100):
        not_ready("Outside a safe training interval; checkpoint files were not loaded", observed)
        return
    # The trainer publishes metrics, selectors and latest separately. Crossing
    # a validation boundary is a retryable snapshot race, not data corruption.
    initial_updates, initial_epoch = observed["updates"], observed["epoch"]
    try:
        rows = validation_rows(arm, manifest)
        if not safe_interval(observed, rows[-1]["updates"]):
            not_ready("Status and durable validation generation differ", observed)
            return
        evidence = plateau(rows, observed)
        native, verified = checkpoint_set(arm, rows, manifest, recorded)
    except Exception:
        current = c.read_json(arm / "status.json")
        if (current.get("activity") != "training" or current["updates"] // 100 != initial_updates // 100
                or current["epoch"] != initial_epoch):
            not_ready("Validation boundary crossed during checkpoint verification", current)
            return
        raise
    current_rows = validation_rows(arm, manifest)
    if c.read_json(arm / "selected-checkpoints.json")["selectors"] != native or current_rows != rows:
        not_ready("Durable checkpoint generation changed during verification", c.read_json(arm / "status.json"))
        return
    if not safe_interval(observed, rows[-1]["updates"]):
        print(json.dumps({"status": "not_ready_no_signal", "reason": "Outside a safe coherent training interval", "observed": observed,
                          "plateau_evidence": evidence}), flush=True)
        return
    verify_identity(run, arm, launch, c.read_json(arm / "process-status.json"))
    observed = c.read_json(arm / "status.json")
    if not safe_interval(observed, rows[-1]["updates"]):
        print(json.dumps({"status": "not_ready_no_signal", "reason": "Safe interval ended during verification", "observed": observed}), flush=True)
        return
    evidence = plateau(rows, observed)
    if not args.execute:
        print(json.dumps({"status": "ready_read_only", "signal_sent": False, "observed": observed,
                          "plateau_evidence": evidence, "verified_checkpoints": verified, "host": host}), flush=True)
        return
    # Register both exact process exits before the final live gate and signal.
    queue = select.kqueue()
    queue.control([select.kevent(pid, filter=select.KQ_FILTER_PROC, flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                                 fflags=select.KQ_NOTE_EXIT) for pid in (CONTROLLER, TRAINER)], 0, 0)
    verify_identity(run, arm, launch, c.read_json(arm / "process-status.json"))
    observed = c.read_json(arm / "status.json")
    if not safe_interval(observed, rows[-1]["updates"]):
        queue.close()
        print(json.dumps({"status": "not_ready_no_signal", "reason": "Final safe interval gate failed", "observed": observed}), flush=True)
        return
    evidence = plateau(rows, observed)
    decision = {"format_version": 1, "arm": ARM_NAME, "decision": "stop_plateau_for_rank32_assessment",
                "authorization_basis": {"standing_user_instructions": ["btw, you can stop any ongoing training",
                    "are we plateuing in the current run? if yes, we should move to the next experiment"],
                    "interpretation": "Apply the established operational plateau rule; no new approval or next experiment is inferred."},
                "plateau_evidence": evidence, "pre_signal_status": observed, "verified_checkpoints": verified,
                "launch_id": LAUNCH_ID, "campaign_manifest_sha256": MANIFEST_SHA256,
                "stop_helper_sha256": c.file_hash(Path(__file__))}
    c.atomic_json(arm / "early-stop-decision.json", decision)
    signaled_at = c.utc()
    os.kill(CONTROLLER, signal.SIGTERM)
    events = queue.control(None, 2, 30)
    if not (exited(CONTROLLER, CONTROLLER_BIRTH) and exited(TRAINER, TRAINER_BIRTH)):
        events += queue.control(None, 2, 30)
    queue.close()
    require(exited(CONTROLLER, CONTROLLER_BIRTH) and exited(TRAINER, TRAINER_BIRTH),
            "Exact processes did not both exit; no retry, restart, or inference is permitted")
    c.verify_bindings(manifest)
    rows = validation_rows(arm, manifest)
    selection, after = checkpoint_set(arm, rows, manifest, recorded)
    require(selection == native and after == verified, "Checkpoint generation changed across the stop; investigate before acceptance")
    preserved = arm / "stopped-checkpoints"
    preserved.mkdir()
    for name, entry in selection.items():
        original, target = Path(entry["path"]), preserved / SELECTORS[name]
        os.link(original, target)
        load_and_verify(target, recorded, manifest, entry["sha256"], entry, name)
        entry.update(original_path=str(original), path=str(target), checkpoint_contents_verified=True)
    os.link(arm / "latest.pt", preserved / "latest.pt")
    latest = load_and_verify(preserved / "latest.pt", recorded, manifest, after["latest"]["sha256"], latest_update=rows[-1]["updates"])
    cessation = {"confirmed": True, "confirmed_at_utc": c.utc(), "signal_target": "dispatcher",
                 "processes": [{"pid": pid, "birth": birth, "alive": False, "kernel_state": ps(pid, "state"), "exited": True}
                               for pid, birth in ((CONTROLLER, CONTROLLER_BIRTH), (TRAINER, TRAINER_BIRTH))],
                 "kernel_exit_events": [int(event.ident) for event in events]}
    final_status, native_process = c.read_json(arm / "status.json"), c.read_json(arm / "process-status.json")
    failure = c.read_json(run / "failure.json")
    require(native_process.get("status") == "failed" and native_process.get("exit_code") == -signal.SIGTERM
            and failure.get("status") == "failed" and "interrupted by signal 15" in failure.get("error", ""),
            "Native stop failure/exit records differ from intentional SIGTERM")
    selection_receipt = {"format_version": 1, "selectors": selection}
    stop = {"format_version": 1, "arm": ARM_NAME, "status": "intentionally_stopped", "signal": "SIGTERM",
            "signaled_at_utc": signaled_at, "observed_status_before_signal": observed, "observed_status_after_exit": final_status,
            "process_cessation": cessation, "native_process_status": native_process, "native_campaign_failure": failure,
            "sources_sha256": manifest["sources_sha256"], "selected_checkpoints": selection_receipt, "latest_checkpoint": latest,
            "no_training_sources_changed": True, "no_inference_performed": True, "launch_id": LAUNCH_ID,
            "stop_helper_sha256": decision["stop_helper_sha256"]}
    c.atomic_json(arm / "stop-receipt.json", stop)
    receipt = {"format_version": 1, "status": "accepted_early_stop", "arm": ARM_NAME, "training_host": "macm3",
               "accepted_at_utc": c.utc(), "accepted_by": "agent_under_standing_user_authorization",
               "acceptance_basis": "Operational early stop under the established plateau rule; incomplete schedule, not a new user approval.",
               "full_schedule_completed": False, "observed_updates": max(observed["updates"], final_status["updates"]),
               "durable_updates": latest["cursor"]["updates"], "completed_epochs": latest["cursor"]["epoch"],
               "planned_updates": sum(manifest["planned_phase_updates"]), "selected_checkpoints": selection_receipt,
               "latest_checkpoint": latest, "plateau_evidence": evidence, "process_cessation": cessation, "test_evaluated": False,
               "stop_receipt": {"path": str(arm / "stop-receipt.json"), "sha256": c.file_hash(arm / "stop-receipt.json")},
               "decision_receipt": {"path": str(arm / "early-stop-decision.json"), "sha256": c.file_hash(arm / "early-stop-decision.json")},
               "artifact_sha256": {str(arm / name): c.file_hash(arm / name) for name in
                   ("manifest.json", "launch.json", "metrics.jsonl", "selected-checkpoints.json", "status.json", "process-status.json")},
               "comparison_policy": "Compare G rank64 and H rank32 at common recorded budgets and both retained winners; total training budgets differ.",
               "launch_id": LAUNCH_ID, "campaign_manifest_sha256": MANIFEST_SHA256}
    c.atomic_json(arm / "accepted-early-stop.json", receipt)
    print(json.dumps({"status": receipt["status"], "observed_updates": receipt["observed_updates"],
                      "durable_updates": receipt["durable_updates"], "completed_epochs": receipt["completed_epochs"],
                      "receipt_sha256": c.file_hash(arm / "accepted-early-stop.json"), "process_cessation": cessation,
                      "selectors": {key: {"updates": entry["cursor"]["updates"], **entry["validation"]} for key, entry in selection.items()}}), flush=True)


if __name__ == "__main__":
    main()
