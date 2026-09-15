"""Launch one serial ConnecTorch campaign after the reference run has finished.

The launcher is idempotent and never polls or waits for training completion.
An independent worker validates parity, runs four smoke checks, then trains the
four predeclared arms serially. Every campaign arm reserves the final test set.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
REVISION = "65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
ARMS = (("A128fixed", 128, "fixed"), ("B32fixed", 32, "fixed"),
        ("C128bounded", 128, "bounded10"), ("D32bounded", 32, "bounded10"))
ENVIRONMENT = {"PYTORCH_ENABLE_MPS_FALLBACK": "0", "OMP_NUM_THREADS": "4",
               "VECLIB_MAXIMUM_THREADS": "4"}
SOURCE_FILES = ("scripts/run_connectorch_campaign.py", "scripts/train_connectorch.py", "scripts/train_ngxson.py",
    "scripts/audit_connectorch.py", "scripts/prepare_connectorch_groups.py", "scripts/prepare_ngxson.py",
    "fly_wordbrain/connectorch_model.py", "fly_wordbrain/connectorch_backend.py",
    "fly_wordbrain/metal_sparse_trainable.py", "fly_wordbrain/metal_sparse.py",
    "fly_wordbrain/plastic_brain.py", "connectorch-source.json", "requirements-connectorch.txt", "requirements-ngxson.txt")
TRAINER_SOURCE_FILES = set(SOURCE_FILES) - {"scripts/run_connectorch_campaign.py", "scripts/train_ngxson.py",
                                          "scripts/audit_connectorch.py", "requirements-connectorch.txt", "requirements-ngxson.txt"}
BASELINE_FILES = ("manifest.json", "launch.json", "process-status.json", "status.json", "results.json")


class CampaignError(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def lock_file(path):
    with Path(path).open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_birth(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def process_alive(pid, birth=None):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return birth is None or process_birth(pid) == birth


def accepted_baseline_readiness(directory, receipt_path):
    """An explicit user disposition is separate from successful recipe completion."""
    directory, receipt_path = Path(directory), Path(receipt_path)
    receipt = read_json(receipt_path)
    if (receipt.get("format_version") != 1 or receipt.get("status") != "accepted_early_stop"
            or receipt.get("accepted_by") != "user" or not receipt.get("instruction", "").strip()):
        raise CampaignError("The accepted baseline receipt lacks explicit user acceptance")
    bound = {str(receipt_path): file_hash(receipt_path)}

    def referenced(entry):
        path = Path(entry["path"])
        if not path.is_absolute() or file_hash(path) != entry.get("sha256"):
            raise CampaignError("Accepted baseline artifact is missing or changed: " + str(path))
        bound[str(path)] = entry["sha256"]
        return path

    manifest_path = referenced(receipt["manifest"])
    if manifest_path.resolve() != (directory / "manifest.json").resolve():
        raise CampaignError("Acceptance receipt belongs to a different baseline")
    stop = read_json(referenced(receipt["stop_receipt"]))
    cessation = receipt.get("process_cessation", {})
    processes = cessation.get("processes", [])
    if (cessation.get("confirmed") is not True or not processes
            or stop.get("process_cessation") != cessation):
        raise CampaignError("Accepted baseline lacks matching process cessation evidence")
    for process in processes:
        if (not isinstance(process.get("pid"), int) or process["pid"] < 1
                or process.get("alive") is not False
                or process_alive(process["pid"], process.get("birth"))):
            raise CampaignError("An accepted baseline process is still alive or has invalid evidence")
    required = ("manifest.json", "launch.json", "status.json", "process-status.json")
    available = {name: read_json(directory / name) for name in required}
    if available["status.json"].get("status") != "stopped_by_user":
        raise CampaignError("Accepted baseline must retain an explicit stopped_by_user status")
    recorded_pids = {artifact[key] for artifact in available.values()
                     for key in ("pid", "worker_pid", "runner_pid", "caffeinate_pid")
                     if isinstance(artifact.get(key), int) and artifact[key] > 0}
    if not recorded_pids.issubset({process["pid"] for process in processes}):
        raise CampaignError("Process cessation evidence omits a recorded baseline process")
    manifest = available["manifest.json"]
    config, planned = manifest.get("config", {}), manifest.get("planned_phase_updates", [])
    conditions = {
        "original_reference": manifest.get("reference_repository") == "ngxson/fly-llm-hf" and manifest.get("reference_revision") == REVISION,
        "not_debug": manifest.get("debug") is False,
        "recipe": all(config.get(key) == value for key, value in {"epochs": 30, "second_epochs": 14,
                    "batch_size": 8, "chunk_size": 32, "seed": 42}.items()),
        "planned_schedule": len(planned) == 2 and all(isinstance(n, int) and n > 0 for n in planned),
        "user_accepted_early_stop": True,
        "process_cessation": True,
    }
    checkpoints = receipt.get("preserved_checkpoints", {})
    if set(checkpoints) != {"minimum_validation_ce", "maximum_validation_accuracy"}:
        raise CampaignError("Acceptance must preserve both validation checkpoint selectors")
    for selector, checkpoint in checkpoints.items():
        referenced(checkpoint)
        metric, cursor = checkpoint.get("validation", {}), checkpoint.get("cursor", {})
        if (checkpoint.get("frozen_buffers_preserved") is not True
                or not manifest.get("frozen_buffers_sha256")
                or checkpoint.get("frozen_buffers_sha256") != manifest["frozen_buffers_sha256"]
                or not isinstance(cursor.get("updates"), int) or cursor["updates"] < 1
                or not math.isfinite(metric.get("cross_entropy", float("nan")))
                or metric["cross_entropy"] < 0 or not 0 <= metric.get("top1_accuracy", -1) <= 1):
            raise CampaignError("Invalid preserved checkpoint evidence: " + selector)
    failures = [name for name, passed in conditions.items() if not passed]
    if failures:
        raise CampaignError("Accepted baseline identity gates failed: " + ", ".join(failures))
    for name in required:
        bound[str(directory / name)] = file_hash(directory / name)
    return {"status": "ready", "artifacts": available, "gates": conditions,
            "acceptance_mode": "user_accepted_early_stop", "acceptance": receipt,
            "baseline_files_sha256": bound}


def baseline_readiness(directory, accepted_baseline_receipt=None):
    """Pure file checks; running, failed and incomplete references never launch."""
    directory = Path(directory)
    if accepted_baseline_receipt is not None:
        return accepted_baseline_readiness(directory, accepted_baseline_receipt)
    available = {name: read_json(directory / name) for name in BASELINE_FILES if (directory / name).exists()}
    status = available.get("status.json", {})
    process = available.get("process-status.json", {})
    if (directory / "failure.json").exists() or status.get("status") in ("failed", "stopped", "debug_stopped", "debug_completed") or process.get("status") == "failed":
        return {"status": "blocked", "reason": "The reference failed, stopped, or was a debug run"}
    if status.get("status") in ("running", "starting") or process.get("status") in ("running", "starting"):
        return {"status": "waiting_for_baseline", "reason": "The reference is still running"}
    if len(available) != len(BASELINE_FILES):
        return {"status": "blocked", "reason": "Reference completion artifacts are missing"}
    manifest, result = available["manifest.json"], available["results.json"]
    config = manifest.get("config", {})
    planned = manifest.get("planned_phase_updates", [])
    checks = result.get("checks", {})
    conditions = {
        "original_reference": manifest.get("reference_repository") == "ngxson/fly-llm-hf" and manifest.get("reference_revision") == REVISION,
        "completed": status.get("status") == result.get("status") == "completed",
        "not_debug": manifest.get("debug") is False and result.get("debug") is False,
        "recipe": all(config.get(key) == value for key, value in {"epochs": 30, "second_epochs": 14,
                    "batch_size": 8, "chunk_size": 32, "seed": 42}.items()),
        "44_epochs": result.get("epochs") == 44,
        "planned_updates": len(planned) == 2 and all(isinstance(n, int) and n > 0 for n in planned)
                           and result.get("updates") == sum(planned),
        "frozen_graph": checks.get("frozen_buffers_preserved") is True
                        and checks.get("frozen_buffers_sha256") == manifest.get("frozen_buffers_sha256")
                        and bool(manifest.get("frozen_buffers_sha256")),
        "final_test_finished": result.get("test_evaluated") is True,
        "successful_process": process.get("status") == "completed" and process.get("exit_code") == 0,
    }
    failures = [key for key, passed in conditions.items() if not passed]
    if failures:
        return {"status": "blocked", "reason": "Reference completion gates failed", "failed_gates": failures}
    return {"status": "ready", "artifacts": available, "gates": conditions,
            "acceptance_mode": "completed_reference_recipe"}


def bind_inputs(args, ready):
    baseline = ready["artifacts"]["manifest.json"]
    baseline_root = Path(ready["artifacts"]["launch.json"]["cwd"])
    model, data, groups = Path(baseline["model_path"]), Path(baseline["data_path"]), args.groups
    sources = {str(args.root / name): file_hash(args.root / name) for name in SOURCE_FILES}
    for name, expected in baseline["trainer_sources_sha256"].items():
        path = baseline_root / name
        if file_hash(path) != expected:
            raise CampaignError("Original reference source changed: " + str(path))
        staged = str(args.root / name)
        if staged in sources and sources[staged] != expected:
            raise CampaignError("Staged reference helper differs from the completed baseline: " + staged)
        sources[str(path)] = expected
    inputs = {str(data): file_hash(data), str(groups): file_hash(groups)}
    if inputs[str(data)] != baseline["data_sha256"]:
        raise CampaignError("The reference dataset changed")
    for name, receipt in baseline["reference_files"].items():
        path = model / name
        if file_hash(path) != receipt["sha256"]:
            raise CampaignError("The pinned reference file changed: " + str(path))
        inputs[str(path)] = receipt["sha256"]
    baseline_files = ready.get("baseline_files_sha256")
    if baseline_files is None:
        baseline_files = {str(args.baseline_run / name): file_hash(args.baseline_run / name) for name in BASELINE_FILES}
    experiment_branches = None
    if args.experiment_branches is not None:
        experiment_branches = read_json(args.experiment_branches)
        entries = experiment_branches.get("arms", {})
        if not experiment_branches.get("repository") or set(entries) != {arm[0] for arm in ARMS}:
            raise CampaignError("Experiment branch mapping must identify the repository and all four arms")
        for name, entry in entries.items():
            commit = entry.get("commit", "")
            if (not entry.get("branch") or len(commit) != 40
                    or any(character not in "0123456789abcdef" for character in commit)):
                raise CampaignError("Invalid experiment branch or commit: " + name)
        inputs[str(args.experiment_branches)] = file_hash(args.experiment_branches)
    dataset = read_json(data)
    validation = dataset["validation"]
    if not validation:
        raise CampaignError("The validation split is empty")
    return {"sources_sha256": sources, "input_files_sha256": inputs, "experiment_branches": experiment_branches,
            "baseline_files_sha256": baseline_files, "model": str(model), "data": str(data),
            "groups": str(groups), "data_sha256": inputs[str(data)], "groups_sha256": inputs[str(groups)],
            "baseline_frozen_buffers_sha256": baseline["frozen_buffers_sha256"],
            "planned_phase_updates": baseline["planned_phase_updates"],
            "validation_stories": len(validation),
            "validation_tokens": sum(len(row["ids"]) - 1 for row in validation),
            "smoke_validation_stories": len(validation[:8]),
            "smoke_validation_tokens": sum(len(row["ids"]) - 1 for row in validation[:8])}


def verify_bindings(manifest):
    for category in ("sources_sha256", "input_files_sha256", "baseline_files_sha256"):
        for path, expected in manifest[category].items():
            if file_hash(path) != expected:
                raise CampaignError("A bound campaign file changed: " + path)


def normalized_args(args):
    args.root = args.root.expanduser().resolve()
    args.baseline_run = args.baseline_run.expanduser().resolve()
    for name in ("accepted_baseline_receipt", "experiment_branches"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.output = (args.root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    args.groups = (args.root / "data/connectorch-groups-v1/groups.npz") if args.groups is None else args.groups.expanduser().absolute()
    # Do not resolve the final symlink: invoking a venv's python symlink by its
    # system target would silently discard the selected virtual environment.
    args.python = str(Path(args.python).expanduser().absolute())
    return args


def campaign_status(output, **values):
    value = {"updated_at_utc": utc(), **values}
    atomic_json(Path(output) / "campaign-status.json", value)
    return value


def launch(args):
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with lock_file(args.output / "launch.lock"):
            if (args.output / "manifest.json").exists():
                current = read_json(args.output / "campaign-status.json") if (args.output / "campaign-status.json").exists() else {}
                if current.get("status") == "completed":
                    return {"status": "completed", "output": str(args.output), "already_exists": True}
                launched = read_json(args.output / "launch.json") if (args.output / "launch.json").exists() else {}
                if current.get("status") in ("starting", "running") and process_alive(launched.get("worker_pid"), launched.get("process_birth")):
                    return {"status": "already_running", "worker_pid": launched["worker_pid"], "output": str(args.output)}
                return {"status": "blocked", "reason": "An existing campaign needs review; automatic restart is disabled", "output": str(args.output)}
            ready = baseline_readiness(args.baseline_run, args.accepted_baseline_receipt)
            if ready["status"] != "ready":
                report = {**ready, "baseline_run": str(args.baseline_run), "checked_at_utc": utc()}
                atomic_json(args.output / "readiness.json", report)
                return report
            bindings = bind_inputs(args, ready)
            launch_id = str(uuid.uuid4())
            manifest = {"format_version": 1, "launch_id": launch_id, "created_at_utc": utc(),
                        "root": str(args.root), "output": str(args.output), "python": args.python,
                        "baseline_run": str(args.baseline_run), "baseline_gates": ready["gates"],
                        "baseline_acceptance_mode": ready["acceptance_mode"],
                        "baseline_acceptance": ready.get("acceptance"),
                        "environment": ENVIRONMENT, "arms": [{"name": name, "d_embed": width, "plasticity": plasticity}
                                                                 for name, width, plasticity in ARMS], **bindings}
            atomic_json(args.output / "manifest.json", manifest)
            command = [args.python, "-u", str(args.root / "scripts/run_connectorch_campaign.py"),
                       "--root", str(args.root), "--baseline-run", str(args.baseline_run),
                       "--output", str(args.output), "--python", args.python, "--groups", str(args.groups),
                       "--worker", "--launch-id", launch_id]
            campaign_status(args.output, status="starting", phase="dispatch", launch_id=launch_id)
            record = {"launch_id": launch_id, "command": command, "cwd": str(args.root),
                      "environment": ENVIRONMENT, "created_at_utc": utc(), "sources_sha256": bindings["sources_sha256"]}
            atomic_json(args.output / "launch.json", record)
            environment = {**os.environ, **ENVIRONMENT}
            with (args.output / "campaign.log").open("a") as log:
                child = subprocess.Popen(command, cwd=args.root, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(worker_pid=child.pid, process_birth=process_birth(child.pid))
            atomic_json(args.output / "launch.json", record)
            return {"status": "launched", "worker_pid": child.pid, "output": str(args.output)}
    except BlockingIOError:
        return {"status": "already_dispatching", "output": str(args.output)}
    except Exception as error:
        report = {"status": "blocked", "reason": str(error), "type": type(error).__name__}
        atomic_json(args.output / "readiness.json", report)
        return report


def trainer_command(manifest, arm, output, smoke=False):
    name, width, plasticity = arm
    command = [manifest["python"], "-u", str(Path(manifest["root"]) / "scripts/train_connectorch.py"),
        "--model", manifest["model"], "--data", manifest["data"], "--groups", manifest["groups"],
        "--output", str(output), "--device", "mps", "--epochs", "30", "--second-epochs", "14",
        "--batch-size", "8", "--chunk-size", "32", "--seed", "42", "--eval-interval-updates", "100",
        "--d-embed", str(width), "--plasticity", plasticity, "--readout-rank", "0",
        "--history-length", "8", "--skip-final-test"]
    if smoke:
        command += ["--max-updates", "8", "--eval-limit", "8"]
    return command


def execute_job(manifest, phase, arm, directory, command):
    """Wait for exactly one child; no scheduler loops or overlapping GPU work."""
    verify_bindings(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("launch.json", "manifest.json", "process-status.json")):
        raise CampaignError("Job output already exists; refusing automatic rerun: " + str(directory))
    launch_record = {"command": command, "cwd": manifest["root"], "environment": ENVIRONMENT,
                     "created_at_utc": utc(), "campaign_launch_id": manifest["launch_id"],
                     "sources_sha256": manifest["sources_sha256"]}
    if manifest.get("experiment_branches") and arm in manifest["experiment_branches"]["arms"]:
        launch_record["experiment_git"] = {"repository": manifest["experiment_branches"]["repository"],
                                            **manifest["experiment_branches"]["arms"][arm]}
    atomic_json(directory / "launch.json", launch_record)
    process = {"status": "starting", "started_at_utc": utc(), "command": command, "runner_pid": os.getpid()}
    atomic_json(directory / "process-status.json", process)
    child = None
    try:
        with (directory / "training.log").open("a") as log:
            child = subprocess.Popen(command, cwd=manifest["root"], env={**os.environ, **ENVIRONMENT},
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            process.update(status="running", worker_pid=child.pid)
            atomic_json(directory / "process-status.json", process)
            campaign_status(manifest["output"], status="running", phase=phase, arm=arm,
                            worker_pid=os.getpid(), child_pid=child.pid, job_output=str(directory))
            code = child.wait()
        process.update(status="completed" if code == 0 else "failed", exit_code=code)
        if code:
            raise CampaignError(f"{phase}/{arm} exited with code {code}; inspect {directory / 'training.log'}")
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            process["exit_code"] = child.wait()
        process.update(status="failed", error=str(error))
        raise
    finally:
        process["finished_at_utc"] = utc()
        atomic_json(directory / "process-status.json", process)
    verify_bindings(manifest)


def validation_records(directory):
    records = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return [record for record in records if record.get("event") == "validation"]


def verify_parity(parity, manifest):
    expected = {"upstream_cpu_vs_connectorch_d128_fixed_cpu", "upstream_cpu_vs_connectorch_d128_fixed_mps",
                "connectorch_d32_bounded10_cpu_vs_mps"}
    comparisons = parity.get("comparisons", {})
    if (parity.get("passed") is not True or parity.get("sources_unchanged") is not True
            or not expected.issubset(comparisons) or not all(comparisons[name].get("passed") is True for name in expected)):
        raise CampaignError("Full-graph parity preflight did not pass all required comparisons")
    if parity.get("data_file_sha256") != manifest["data_sha256"] or parity.get("groups_file_sha256") != manifest["groups_sha256"]:
        raise CampaignError("Parity used different data or node-type groups")
    sources = parity.get("source_sha256", {})
    if not sources or not parity.get("connectorch"):
        raise CampaignError("Parity omitted required source or installed-package receipts")
    for relative, digest in sources.items():
        if manifest["sources_sha256"].get(str(Path(manifest["root"]) / relative)) != digest:
            raise CampaignError("Parity used unbound source: " + relative)


def graph_audit(audit, manifest):
    if audit.get("frozen_buffers_preserved") is not True:
        raise CampaignError("An arm did not preserve frozen graph buffers")
    for name, digest in manifest["baseline_frozen_buffers_sha256"].items():
        if audit.get("frozen_buffers_sha256", {}).get(name) != digest:
            raise CampaignError("An arm differs from the reference graph/interface: " + name)
    adaptation = audit.get("adaptation", {})
    if adaptation.get("zero_edges_preserved") is not True or adaptation.get("nonzero_signs_preserved") is not True:
        raise CampaignError("An arm changed stored zero edges or edge signs")


def verify_selected_checkpoints(directory, records, manifest):
    receipt = read_json(Path(directory) / "selected-checkpoints.json")
    selectors = receipt.get("selectors", {})
    expected = {
        "minimum_validation_ce": min(records, key=lambda record: record["validation"]["cross_entropy"]),
        "maximum_validation_accuracy": max(records, key=lambda record: record["validation"]["top1_accuracy"]),
    }
    if receipt.get("format_version") != 1 or set(selectors) != set(expected):
        raise CampaignError("An arm did not retain both validation selectors")
    for name, record in expected.items():
        entry = selectors[name]
        filename = "best.pt" if name == "minimum_validation_ce" else "best-accuracy.pt"
        path = Path(directory) / filename
        if (Path(entry.get("path", "")).resolve() != path.resolve()
                or file_hash(path) != entry.get("sha256")
                or entry.get("cursor", {}).get("updates") != record["updates"]
                or entry.get("validation") != record["validation"]
                or entry.get("frozen_buffers_preserved") is not True
                or any(entry.get("frozen_buffers_sha256", {}).get(key) != digest
                       for key, digest in manifest["baseline_frozen_buffers_sha256"].items())):
            raise CampaignError("Retained validation checkpoint identity or selector mismatch: " + name)
    return selectors


def verify_arm(manifest, arm, directory, smoke=False):
    directory = Path(directory)
    name, width, plasticity = arm
    if (directory / "failure.json").exists():
        raise CampaignError("Arm failure artifact exists: " + str(directory))
    status, recorded = read_json(directory / "status.json"), read_json(directory / "manifest.json")
    process = read_json(directory / "process-status.json")
    if process.get("status") != "completed" or process.get("exit_code") != 0:
        raise CampaignError("Arm worker did not exit successfully: " + name)
    expected = {"epochs": 30, "second_epochs": 14, "batch_size": 8, "chunk_size": 32, "seed": 42,
                "d_embed": width, "plasticity": plasticity, "readout_rank": 0, "history_length": 8,
                "skip_final_test": True, "device": "mps", "eval_interval_updates": 100,
                "max_updates": 8 if smoke else None, "eval_limit": 8 if smoke else None}
    if any(recorded.get("config", {}).get(key) != value for key, value in expected.items()):
        raise CampaignError("Arm configuration differs from the declared campaign: " + name)
    if recorded.get("data_sha256") != manifest["data_sha256"] or recorded.get("groups_sha256") != manifest["groups_sha256"]:
        raise CampaignError("Arm data or group identity changed: " + name)
    if set(recorded.get("trainer_sources_sha256", {})) != TRAINER_SOURCE_FILES:
        raise CampaignError("Arm omitted or changed its trainer source receipt: " + name)
    for relative, digest in recorded["trainer_sources_sha256"].items():
        if manifest["sources_sha256"].get(str(Path(manifest["root"]) / relative)) != digest:
            raise CampaignError("Arm used unbound training source: " + relative)
    if recorded.get("connectorch_source") != manifest.get("preflight_connectorch_source"):
        raise CampaignError("Arm used a different installed Connectorch package: " + name)
    records = validation_records(directory)
    if not records:
        raise CampaignError("Arm has no validation records: " + name)
    latest = records[-1]
    graph_audit(latest["model_audit"], manifest)
    stories = manifest["smoke_validation_stories" if smoke else "validation_stories"]
    tokens = manifest["smoke_validation_tokens" if smoke else "validation_tokens"]
    for record in records:
        metric = record["validation"]
        if metric.get("stories") != stories or metric.get("tokens") != tokens:
            raise CampaignError("Arm validation coverage changed: " + name)
        if not math.isfinite(metric["cross_entropy"]) or metric["cross_entropy"] < 0 or not 0 <= metric["top1_accuracy"] <= 1:
            raise CampaignError("Arm validation metric is invalid: " + name)
    retained = verify_selected_checkpoints(directory, records, manifest)
    if smoke:
        if status.get("status") != "debug_stopped" or status.get("updates") != 8 or not status.get("debug") or latest.get("updates") != 8:
            raise CampaignError("Smoke did not complete exactly eight updates: " + name)
        if status.get("test_evaluated") is not False or (directory / "results.json").exists():
            raise CampaignError("Smoke unexpectedly reached final testing: " + name)
        if plasticity == "bounded10":
            gradients = read_json(directory / "initial-gradients.json")
            for parameter, stats in (("brain.edge_theta_source", "source_type_theta"),
                                     ("brain.edge_theta_destination", "destination_type_theta")):
                grad = gradients.get(parameter, {})
                if grad.get("present") is not True or grad.get("finite") is not True or grad.get("nonzero_values", 0) <= 0:
                    raise CampaignError("Plastic gain gradient was missing, zero or nonfinite: " + parameter)
                if latest["model_audit"]["adaptation"].get(stats, {}).get("rms", 0) <= 0:
                    raise CampaignError("Plastic gain parameters did not move from zero: " + parameter)
            if latest["model_audit"]["adaptation"]["edge_displacement"]["rms"] <= 0:
                raise CampaignError("Plastic parameters did not change effective edge weights")
        return {"name": name, "passed": True, "updates": 8, "validation": latest["validation"],
                "selected_checkpoints": retained}
    result = read_json(directory / "results.json")
    if (status.get("status") != "completed" or result.get("status") != "completed" or result.get("debug") is not False
            or result.get("epochs") != 44 or result.get("updates") != sum(manifest["planned_phase_updates"])
            or recorded.get("planned_phase_updates") != manifest["planned_phase_updates"]):
        raise CampaignError("Full arm did not complete the entire declared schedule: " + name)
    if result.get("test_evaluated") is not False or result.get("test_deferred") is not True or "test" in result:
        raise CampaignError("Full arm accessed its reserved final test: " + name)
    graph_audit(result["checks"], manifest)
    graph_audit(result["checks"]["selected_checkpoint_audit"], manifest)
    selected = min(records, key=lambda record: record["validation"]["cross_entropy"])
    if (result["selected"]["cross_entropy"] != selected["validation"]["cross_entropy"]
            or result["selected"]["updates"] != selected["updates"]):
        raise CampaignError("Arm did not select its best validation checkpoint: " + name)
    accuracy_selected = max(records, key=lambda record: record["validation"]["top1_accuracy"])
    if (result.get("selected_accuracy", {}).get("top1_accuracy") != accuracy_selected["validation"]["top1_accuracy"]
            or result["selected_accuracy"].get("updates") != accuracy_selected["updates"]
            or result.get("selected_checkpoints") != retained):
        raise CampaignError("Arm did not retain its maximum validation accuracy checkpoint: " + name)
    return {"name": name, "d_embed": width, "plasticity": plasticity, "updates": result["updates"],
            "selected_updates": selected["updates"], "validation": selected["validation"],
            "accuracy_selected_updates": accuracy_selected["updates"], "accuracy_validation": accuracy_selected["validation"],
            "selected_checkpoints": retained,
            "experiment_git": ({"repository": manifest["experiment_branches"]["repository"],
                                 **manifest["experiment_branches"]["arms"][name]} if manifest.get("experiment_branches") else None),
            "parameter_counts": recorded["parameter_counts"], "test_evaluated": False,
            "output": str(directory), "elapsed_seconds": result.get("elapsed_seconds_this_invocation")}


def compare_arms(arms):
    by_name = {arm["name"]: arm for arm in arms}
    if set(by_name) != {name for name, _, _ in ARMS} or len(arms) != 4:
        raise CampaignError("Require exactly the four declared arms")
    ce = {name: arm["validation"]["cross_entropy"] for name, arm in by_name.items()}
    accuracy = {name: arm["accuracy_validation"]["top1_accuracy"] for name, arm in by_name.items()}
    fixed = ce["B32fixed"] - ce["A128fixed"]
    plastic = ce["D32bounded"] - ce["C128bounded"]
    return {"compression_ce_penalty_fixed": fixed, "compression_ce_penalty_plastic": plastic,
            "compression_plasticity_interaction_ce": plastic - fixed,
            "interaction_interpretation": "Negative values mean bounded plasticity reduced the compression CE penalty.",
            "plasticity_ce_improvement_width128": ce["A128fixed"] - ce["C128bounded"],
            "plasticity_ce_improvement_width32": ce["B32fixed"] - ce["D32bounded"],
            "validation_best_arm": min(ce, key=ce.get), "validation_best_accuracy_arm": max(accuracy, key=accuracy.get),
            "accuracy_compression_penalty_fixed": accuracy["A128fixed"] - accuracy["B32fixed"],
            "accuracy_compression_penalty_plastic": accuracy["C128bounded"] - accuracy["D32bounded"],
            "needs_replication_before_geometry_claim": True,
            "matched_random_graph_control_trained": False, "final_test_deferred": True}


def start_awake():
    if Path("/usr/bin/caffeinate").exists():
        return subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None


def worker(args):
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with lock_file(args.output / "worker.lock"):
            manifest = read_json(args.output / "manifest.json")
            status = read_json(args.output / "campaign-status.json")
            if args.launch_id != manifest["launch_id"] or status.get("status") != "starting":
                return {"status": "blocked", "reason": "Worker is not the pending authorized campaign launch"}
            verify_bindings(manifest)
            campaign_status(args.output, status="running", phase="preflight", worker_pid=os.getpid())
            awake = None
            previous = {}
            def stop(signum, frame):
                raise CampaignError("Campaign interrupted by signal " + str(signum))
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, stop)
            try:
                awake = start_awake()
                preflight = args.output / "preflight"
                command = [manifest["python"], "-u", str(args.root / "scripts/audit_connectorch.py"),
                           "--model", manifest["model"], "--data", manifest["data"], "--groups", manifest["groups"],
                           "--output", str(preflight / "parity.json")]
                execute_job(manifest, "preflight", "parity", preflight, command)
                parity = read_json(preflight / "parity.json")
                verify_parity(parity, manifest)
                manifest["preflight_connectorch_source"] = parity["connectorch"]
                manifest["preflight_receipt_sha256"] = file_hash(preflight / "parity.json")
                atomic_json(args.output / "manifest.json", manifest)
                smokes = []
                for arm in ARMS:
                    directory = args.output / "smokes" / arm[0]
                    execute_job(manifest, "smoke", arm[0], directory, trainer_command(manifest, arm, directory, smoke=True))
                    smokes.append(verify_arm(manifest, arm, directory, smoke=True))
                atomic_json(args.output / "smoke-results.json", {"passed": True, "arms": smokes})
                arms = []
                for arm in ARMS:
                    directory = args.output / "arms" / arm[0]
                    execute_job(manifest, "training", arm[0], directory, trainer_command(manifest, arm, directory))
                    arms.append(verify_arm(manifest, arm, directory))
                    atomic_json(args.output / "partial-results.json", {"completed_arms": arms, "final_test_deferred": True})
                verify_bindings(manifest)
                result = {"status": "completed", "finished_at_utc": utc(), "arms": arms,
                          "comparison": compare_arms(arms), "test_evaluated": False,
                          "preflight_passed": True, "smoke_checks_passed": True}
                atomic_json(args.output / "results.json", result)
                campaign_status(args.output, status="completed", phase="finished", worker_pid=os.getpid(),
                                completed_arms=4, comparison=result["comparison"], test_evaluated=False)
                return result
            finally:
                if awake is not None:
                    awake.terminate()
                    awake.wait()
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
    except BlockingIOError:
        return {"status": "already_running", "output": str(args.output)}
    except BaseException as error:
        previous = read_json(args.output / "campaign-status.json") if (args.output / "campaign-status.json").exists() else {}
        failure = {"status": "failed", "error": str(error), "type": type(error).__name__,
                   "phase": previous.get("phase"), "arm": previous.get("arm"), "finished_at_utc": utc()}
        atomic_json(args.output / "failure.json", failure)
        campaign_status(args.output, **failure)
        return failure


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--accepted-baseline-receipt", type=Path,
                   help="Explicit user acceptance of a preserved early-stopped baseline; never implies full recipe completion")
    p.add_argument("--experiment-branches", type=Path,
                   help="Immutable repository/branch/commit provenance mapping for all four arms")
    p.add_argument("--output", type=Path, default=Path("results/connectorch-encoder-v1"))
    p.add_argument("--groups", type=Path)
    p.add_argument("--python", default=sys.executable)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--worker", action="store_true")
    p.add_argument("--launch-id", help=argparse.SUPPRESS)
    return p


def main():
    args = normalized_args(parser().parse_args())
    result = launch(args) if args.launch else worker(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    if result["status"] in ("failed", "blocked"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
