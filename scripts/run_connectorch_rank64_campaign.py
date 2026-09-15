#!/usr/bin/env python3
"""Run only G rank64 on macm3 after its own parity and eight-update smoke.

The accepted E rank128 snapshot is a comparison baseline. No F completion,
quality evaluation, or checkpoint is a prerequisite. All training starts from
scratch with the unchanged trainer; a failure or stop never retries itself.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

SELF = Path(__file__).resolve()
ROOT = SELF.parents[1]
spec = importlib.util.spec_from_file_location("rank64_parent_handoff", SELF.with_name("continue_connectorch_decoder_campaign.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
d, c = h.d, h.c
require, host_guard, validation_records = d.require, d.host_guard, d.validation_records
verify_cessation, verify_smoke_parameter_updates = d.verify_cessation, d.verify_smoke_parameter_updates
ARMS = (("G32rank64fixed", 32, "fixed"),)
SELECTORS, PLANNED_UPDATES = d.SELECTORS, d.PLANNED_UPDATES
SOURCE_FILES = (*d.SOURCE_FILES, "scripts/continue_connectorch_decoder_campaign.py",
                "scripts/run_connectorch_rank64_campaign.py", "scripts/audit_connectorch_rank64.py")


def arm_config(arm, smoke=False):
    require(arm in ARMS, "Unknown rank64 arm")
    return {"epochs": 30, "second_epochs": 14, "batch_size": 8, "chunk_size": 32, "seed": 42,
            "d_embed": 32, "plasticity": arm[2], "readout_rank": 64, "history_length": 8,
            "skip_final_test": True, "device": "mps", "eval_interval_updates": 100,
            "max_updates": 8 if smoke else None, "eval_limit": 8 if smoke else None}


def expected_counts(arm):
    result = {"embedding": 32768, "input_projection": 450048, "neurons": 148179,
              "edge_gains": 18322 if arm[2] == "bounded10" else 0, "layernorm": 98786,
              "readout": 64 * (49393 + 1024), "other": 0}
    result["total"] = sum(result.values())
    return result


def verify_preflight(parity, manifest):
    """Require the fresh, single-G graph and factor-gradient parity receipt."""
    require(parity.get("passed") is True and parity.get("sources_unchanged") is True
            and parity.get("inputs_unchanged") is True and parity.get("connectorch_unchanged") is True
            and parity.get("exact_ir", {}).get("passed") is True and parity.get("final_host_guard", {}).get("passed") is True
            and parity.get("data_file_sha256") == manifest["data_sha256"]
            and parity.get("groups_file_sha256") == manifest["groups_sha256"], "Rank64 parity failed or input hashes differ")
    require(set(parity.get("arms", {})) == {ARMS[0][0]}
            and set(parity.get("cases", {})) == {ARMS[0][0] + "_cpu", ARMS[0][0] + "_mps"}
            and parity.get("training_run") is False and parity.get("optimizer_steps") == 0
            and parity.get("reserved_test_evaluated") is False, "Expected single-G, inference/backward-only parity")
    require(parity.get("source_sha256", {}).get("scripts/audit_connectorch_rank64.py") is not None,
            "Rank64 preflight must bind its own audit source")
    for arm in ARMS:
        name = arm[0]
        case = parity.get("arms", {}).get(name, {})
        require(case.get("d_embed") == 32 and case.get("readout_rank") == 64
                and case.get("history_length") == 8 and case.get("seed") == 42 and case.get("plasticity") == arm[2]
                and case.get("parameter_counts") == expected_counts(arm), "Missing or incorrect rank64 preflight arm: " + name)
        comparison = parity.get("comparisons", {}).get(name + "_cpu_vs_mps", {})
        require(comparison.get("passed") is True and comparison.get("loss_abs_difference", math.inf) < 1e-4
                and comparison.get("max_gradient_relative_l2", math.inf) < 1e-3
                and comparison.get("cache_ids_and_length_equal") is True, "Rank64 CPU/MPS parity tolerance failed: " + name)
        for device in ("cpu", "mps"):
            audit = parity.get("cases", {}).get(name + "_" + device, {})
            require(audit.get("passed") is True and audit.get("device") == device and audit.get("targets") == 64
                    and audit.get("parameter_count") == expected_counts(arm)["total"]
                    and audit.get("frozen_buffers_unchanged") is True
                    and audit.get("frozen_buffers_sha256") == manifest["baseline_frozen_buffers_sha256"]
                    and audit.get("padding_gradient_zero") is True and audit.get("native_kernel_gate") is True,
                    "Rank64 preflight omitted graph/gradient/kernel evidence: " + name + "/" + device)
            finite = audit.get("parameters_and_gradients_finite", {})
            require(bool(finite) and all(finite.values()) and all(finite.get(key) is True for key in ("lm_head.0.weight", "lm_head.1.weight")), "Rank64 readout gradient missing")
            gradients = audit.get("required_gradients", {})
            require(audit.get("all_required_gradients_nonzero") is True
                    and all(gradients.get(key, {}).get("finite") is True and gradients[key].get("nonzero_values", 0) > 0
                            for key in ("lm_head.0.weight", "lm_head.1.weight")), "Rank64 readout factor gradients missing or zero")
    initialization = parity.get("initialization", {})
    require(initialization.get("passed") is True and initialization.get("cpu_mps_parameter_hashes_equal") is True
            and initialization.get("readout_shapes") == {"lm_head.0.weight": [64, 49393], "lm_head.1.weight": [1024, 64]}
            and bool(initialization.get("parameter_hashes")), "Rank64 CPU/MPS initialization or factor shapes differ")
    require(parity.get("connectorch") == manifest["baseline_connectorch_source"] and bool(parity.get("source_sha256")), "Parity installed package/source receipt differs")
    for relative, digest in parity["source_sha256"].items():
        require(manifest["sources_sha256"].get(str(Path(manifest["root"]) / relative)) == digest, "Preflight used unbound source: " + relative)


def prepare(args):
    host = host_guard()
    parent_path = args.parent / "manifest.json"
    parent = c.read_json(parent_path)
    require(parent.get("root") == str(args.root) and parent.get("output") == str(args.parent)
            and parent.get("planned_phase_updates") == PLANNED_UPDATES,
            "Parent must be the staged original decoder campaign")
    c.verify_bindings(parent)
    summary, baseline_files = h.accepted_e(parent, args.parent, args.accepted_e_receipt)
    # Only E acceptance is consumed. The stopped F campaign is neither read nor
    # required to finish; the native exclusive-process guard handles GPU overlap.
    d.bound(parent_path, c.file_hash(parent_path), baseline_files)
    sources = dict(parent["sources_sha256"])
    for name in SOURCE_FILES:
        path = args.root / name
        digest = c.file_hash(path)
        require(str(path) not in sources or sources[str(path)] == digest, "Parent source changed: " + name)
        sources[str(path)] = digest
    inputs = dict(parent["input_files_sha256"])
    branches_path = d.bound(args.experiment_branches, c.file_hash(args.experiment_branches), inputs)
    branches = c.read_json(branches_path)
    name = ARMS[0][0]
    require(branches.get("repository") == "https://github.com/fernando-neto-ai/fly-wordbrain"
            and set(branches.get("arms", {})) == {name}, "Branch mapping must name only G")
    branch = branches["arms"][name]
    commit = branch.get("commit", "")
    require(branch.get("branch") == "exp/encoder32-readout64-fixed" and len(commit) == 40
            and all(ch in "0123456789abcdef" for ch in commit), "Invalid G branch or commit provenance")
    config_path = args.root / "experiments/configs" / (name + ".json")
    d.bound(config_path, c.file_hash(config_path), inputs)
    require(branch.get("configuration") == str(config_path.relative_to(args.root))
            and branch.get("configuration_sha256") == inputs[str(config_path)],
            "G branch mapping configuration path or hash differs")
    declared = c.read_json(config_path)
    require(declared.get("experiment_id") == name and declared.get("branch") == branch["branch"]
            and declared.get("training_host") == "macm3" and declared.get("trainer") == "scripts/train_connectorch.py"
            and declared.get("trainable_parameters") == expected_counts(ARMS[0])["total"]
            and declared.get("dataset_sha256") == parent["data_sha256"] and declared.get("groups_sha256") == parent["groups_sha256"]
            and all(declared.get("training", {}).get(key) == value for key, value in arm_config(ARMS[0]).items()),
            "G branch configuration differs from the rank64 command")
    parity_path = d.bound(args.preflight_receipt, c.file_hash(args.preflight_receipt), inputs)
    parity = c.read_json(parity_path)
    for relative, digest in parity.get("source_sha256", {}).items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Preflight source must be repository relative")
        require(str(args.root / path) not in sources or sources[str(args.root / path)] == digest,
                "Rank64 parity conflicts with bound parent source: " + relative)
        d.bound(args.root / path, digest, sources)
    manifest = copy.deepcopy(parent)
    manifest.pop("baseline_quality", None)
    manifest.update(format_version=1, launch_id=str(uuid.uuid4()), created_at_utc=c.utc(), output=str(args.output),
        python=args.python, host_guard=host, authorized_launcher={"pid": os.getpid(), "birth": c.process_birth(os.getpid())},
        arms=[{"name": name, **arm_config(ARMS[0]), "parameter_counts": expected_counts(ARMS[0])}],
        parent_campaign={"path": str(parent_path), "sha256": baseline_files[str(parent_path)], "launch_id": parent["launch_id"]},
        baseline_run=summary["output"], baseline_summary=summary, baseline_selected_checkpoints=summary["selected_checkpoints"],
        baseline_acceptance={"path": str(args.accepted_e_receipt), "sha256": baseline_files[str(args.accepted_e_receipt)]},
        baseline_files_sha256=baseline_files, sources_sha256=sources, input_files_sha256=inputs,
        preflight_receipt={"path": str(parity_path), "sha256": inputs[str(parity_path)]},
        preflight_connectorch_source=parity.get("connectorch"), experiment_branches=branches,
        experiment_configs={name: {"path": str(config_path), "sha256": inputs[str(config_path)]}},
        initialization="from scratch with seed42; accepted E is a comparison baseline, not resumed weights",
        stop_policy="An authorized plateau stop preserves both selectors and latest, then halts. No automatic restart.",
        final_test_deferred=True, baseline_full_schedule_completed=False,
        qualification="Single rank64 fixed arm; accepted E was early-stopped; compare common budgets and both selectors. No F completion prerequisite.")
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


def verify_arm(manifest, arm, directory, smoke=False):
    directory = Path(directory)
    name, width, plasticity = arm
    require(arm in ARMS and not (directory / "failure.json").exists(), "Unknown or failed rank64 arm")
    recorded, status = c.read_json(directory / "manifest.json"), c.read_json(directory / "status.json")
    process = c.read_json(directory / "process-status.json")
    require(process.get("status") == "completed" and process.get("exit_code") == 0, "Rank64 worker did not exit successfully")
    require(all(recorded.get("config", {}).get(key) == value for key, value in arm_config(arm, smoke).items()), "Rank64 arm configuration differs")
    require(recorded.get("parameter_counts") == expected_counts(arm), "Rank64 parameter counts differ")
    require(recorded.get("data_sha256") == manifest["data_sha256"] and recorded.get("groups_sha256") == manifest["groups_sha256"], "Rank64 data/group identity differs")
    require(recorded.get("planned_phase_updates") == manifest["planned_phase_updates"], "Rank64 planned updates differ")
    require(set(recorded.get("trainer_sources_sha256", {})) == c.TRAINER_SOURCE_FILES, "Rank64 trainer source inventory differs")
    for relative, digest in recorded["trainer_sources_sha256"].items():
        require(manifest["sources_sha256"].get(str(Path(manifest["root"]) / relative)) == digest, "Rank64 used unbound source: " + relative)
    require(recorded.get("connectorch_source") == manifest["preflight_connectorch_source"], "Rank64 installed package differs")
    records = validation_records(directory)
    require(bool(records), "Rank64 has no validation records")
    for record in records:
        metric = record["validation"]
        require(metric.get("stories") == manifest["smoke_validation_stories" if smoke else "validation_stories"]
                and metric.get("tokens") == manifest["smoke_validation_tokens" if smoke else "validation_tokens"], "Rank64 validation coverage differs")
        require(math.isfinite(metric["cross_entropy"]) and metric["cross_entropy"] >= 0
                and 0 <= metric.get("correct", -1) <= metric["tokens"]
                and abs(metric["top1_accuracy"] - metric["correct"] / metric["tokens"]) < 1e-12, "Invalid rank64 validation metrics")
        c.graph_audit(record["model_audit"], manifest)
    retained = c.verify_selected_checkpoints(directory, records, manifest)
    if smoke:
        require(status.get("status") == "debug_stopped" and status.get("debug") is True
                and status.get("updates") == records[-1]["updates"] == 8, "Rank64 smoke did not stop at exactly eight updates")
        require(status.get("test_evaluated") is False and not (directory / "results.json").exists(), "Rank64 smoke accessed final testing")
        updates = verify_smoke_parameter_updates(directory, recorded, plasticity)
        adaptation = records[-1]["model_audit"]["adaptation"]
        if plasticity == "bounded10":
            require(adaptation.get("edge_displacement", {}).get("rms", 0) > 0
                    and all(adaptation.get(key, {}).get("rms", 0) > 0 for key in ("source_type_theta", "destination_type_theta")), "Bounded rank64 smoke did not alter effective edges")
        return {"name": name, "passed": True, "updates": 8, "validation": records[-1]["validation"], "parameter_updates": updates,
                "parameter_counts": recorded["parameter_counts"], "selected_checkpoints": retained}
    result = c.read_json(directory / "results.json")
    require(status.get("status") == result.get("status") == "completed" and result.get("debug") is False
            and result.get("epochs") == 44 and result.get("updates") == sum(PLANNED_UPDATES), "Rank64 full schedule is incomplete")
    require(result.get("test_evaluated") is False and result.get("test_deferred") is True and "test" not in result, "Rank64 accessed its reserved test")
    c.graph_audit(result["checks"], manifest)
    c.graph_audit(result["checks"]["selected_checkpoint_audit"], manifest)
    for selector, result_key in (("minimum_validation_ce", "selected"), ("maximum_validation_accuracy", "selected_accuracy")):
        entry = retained[selector]
        require(result[result_key]["updates"] == entry["cursor"]["updates"]
                and all(result[result_key][key] == entry["validation"][key] for key in ("cross_entropy", "top1_accuracy")), "Rank64 final selector differs")
    require(result.get("selected_checkpoints") == retained, "Rank64 final selector receipts differ")
    return {"name": name, "d_embed": width, "readout_rank": 64, "history_length": 8, "plasticity": plasticity,
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
                return {"status": "completed" if status.get("status") == "completed" else "blocked", "reason": "Existing rank64 campaign is never automatically restarted", "output": str(args.output)}
            manifest = prepare(args)
            c.atomic_json(args.output / "manifest.json", manifest)
            argv = [args.python, "-u", str(SELF), "--root", str(args.root), "--parent", str(args.parent),
                    "--accepted-e-receipt", str(args.accepted_e_receipt), "--preflight-receipt", str(args.preflight_receipt),
                    "--experiment-branches", str(args.experiment_branches), "--output", str(args.output),
                    "--python", args.python, "--worker", "--launch-id", manifest["launch_id"]]
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
                    and c.file_hash(args.output / "manifest.json") == launch_record["manifest_sha256"], "Worker does not match the pending rank64 launch")
            c.verify_bindings(manifest)
            host_guard(manifest)
            verify_cessation(c.read_json(manifest["baseline_acceptance"]["path"]))
            previous = {}
            def stop(signum, frame):
                raise c.CampaignError("Rank64 campaign interrupted by signal " + str(signum))
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
                          "finished_at_utc": c.utc(), "qualification": "G rank64 uses the unchanged full recipe; accepted E was early-stopped. Compare common budgets; one seed, no randomized-connectome control."}
                c.atomic_json(args.output / "results.json", result)
                c.campaign_status(args.output, status="completed", phase="finished", completed_arms=1, test_evaluated=False)
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
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--accepted-e-receipt", type=Path, required=True)
    p.add_argument("--preflight-receipt", type=Path, required=True)
    p.add_argument("--experiment-branches", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/connectorch-rank64-v1"))
    p.add_argument("--python", default=sys.executable)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--worker", action="store_true")
    p.add_argument("--launch-id", help=argparse.SUPPRESS)
    return p


def normalized_args(args):
    args.root = args.root.expanduser().resolve()
    for name in ("parent", "accepted_e_receipt", "preflight_receipt", "experiment_branches", "output"):
        path = getattr(args, name).expanduser()
        setattr(args, name, (args.root / path).resolve() if not path.is_absolute() else path.resolve())
    args.python = str(Path(args.python).expanduser().absolute())
    require(args.root == ROOT and args.output != args.parent and args.output not in args.parent.parents
            and args.parent not in args.output.parents, "Use this staged root and a fresh separate rank64 output")
    return args


def main():
    args = normalized_args(parser().parse_args())
    result = launch(args) if args.launch else worker(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    if result["status"] in ("failed", "blocked"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
