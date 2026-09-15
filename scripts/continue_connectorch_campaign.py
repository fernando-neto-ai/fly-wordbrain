"""Continue B/C/D after a separately accepted, preserved early stop of arm A.

The original campaign and all bound training sources remain immutable. This
companion owns a new output directory and never relabels an interrupted arm as
having completed its planned schedule.
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
spec = importlib.util.spec_from_file_location("original_connectorch_campaign", SELF.with_name("run_connectorch_campaign.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def bound_artifact(entry, bindings):
    path = Path(entry["path"])
    if not path.is_absolute() or c.file_hash(path) != entry.get("sha256"):
        raise c.CampaignError("Acceptance artifact changed: " + str(path))
    bindings[str(path)] = entry["sha256"]
    return path


def accepted_arm(parent, parent_dir, receipt_path):
    """Validate disposition and the original dual-selector evidence separately."""
    receipt = c.read_json(receipt_path)
    bindings = {str(receipt_path): c.file_hash(receipt_path)}
    if (receipt.get("format_version") != 1 or receipt.get("status") != "accepted_early_stop"
            or receipt.get("accepted_by") != "user" or receipt.get("arm") != "A128fixed"
            or not receipt.get("instruction", "").strip()):
        raise c.CampaignError("Explicit user acceptance of A128fixed is required")
    directory = parent_dir / "arms/A128fixed"
    for key, expected in (("campaign_manifest", parent_dir / "manifest.json"),
                          ("arm_manifest", directory / "manifest.json"),
                          ("metrics", directory / "metrics.jsonl"),
                          ("selected_checkpoint_receipt", directory / "selected-checkpoints.json")):
        if bound_artifact(receipt[key], bindings).resolve() != expected.resolve():
            raise c.CampaignError("Acceptance belongs to another campaign or arm: " + key)
    stop = c.read_json(bound_artifact(receipt["stop_receipt"], bindings))
    cessation = receipt.get("process_cessation", {})
    processes = cessation.get("processes", [])
    if cessation.get("confirmed") is not True or not processes or stop.get("process_cessation") != cessation:
        raise c.CampaignError("Matching process cessation evidence is required")
    for p in processes:
        if (not isinstance(p.get("pid"), int) or p["pid"] < 1 or p.get("alive") is not False
                or c.process_alive(p["pid"], p.get("birth"))):
            raise c.CampaignError("A stopped campaign process is still alive")
    known = set()
    for path in (parent_dir / "launch.json", directory / "process-status.json", directory / "status.json"):
        artifact = c.read_json(path)
        known.update(artifact[k] for k in ("pid", "worker_pid", "runner_pid", "caffeinate_pid")
                     if isinstance(artifact.get(k), int) and artifact[k] > 0)
        bindings[str(path)] = c.file_hash(path)
    if not known.issubset({p["pid"] for p in processes}):
        raise c.CampaignError("Process cessation omitted a recorded worker")
    records = c.validation_records(directory)
    if not records:
        raise c.CampaignError("Stopped A has no validation observations")
    recorded = c.read_json(directory / "manifest.json")
    expected = {"d_embed": 128, "plasticity": "fixed", "readout_rank": 0, "history_length": 8,
                "device": "mps", "skip_final_test": True, "max_updates": None, "eval_limit": None,
                "epochs": 30, "second_epochs": 14, "batch_size": 8, "chunk_size": 32, "seed": 42}
    if any(recorded.get("config", {}).get(k) != v for k, v in expected.items()):
        raise c.CampaignError("Stopped A configuration differs from the declared arm")
    if (recorded.get("data_sha256") != parent["data_sha256"]
            or recorded.get("groups_sha256") != parent["groups_sha256"]
            or recorded.get("connectorch_source") != parent["preflight_connectorch_source"]
            or recorded.get("planned_phase_updates") != parent["planned_phase_updates"]
            or set(recorded.get("trainer_sources_sha256", {})) != c.TRAINER_SOURCE_FILES):
        raise c.CampaignError("Stopped A input, source or schedule identity differs")
    for relative, digest in recorded["trainer_sources_sha256"].items():
        if parent["sources_sha256"].get(str(Path(parent["root"]) / relative)) != digest:
            raise c.CampaignError("Stopped A used unbound source: " + relative)
    for row in records:
        if (row["validation"].get("stories") != parent["validation_stories"]
                or row["validation"].get("tokens") != parent["validation_tokens"]):
            raise c.CampaignError("Stopped A validation coverage differs")
        metric = row["validation"]
        if (not math.isfinite(metric.get("cross_entropy", float("nan"))) or metric["cross_entropy"] < 0
                or not 0 <= metric.get("top1_accuracy", -1) <= 1):
            raise c.CampaignError("Stopped A validation metric is invalid")
        c.graph_audit(row["model_audit"], parent)
    selectors = c.verify_selected_checkpoints(directory, records, parent)
    preserved = receipt.get("preserved_checkpoints", {})
    if set(preserved) != {*selectors, "latest"}:
        raise c.CampaignError("Both selectors and latest must be preserved")
    for name, entry in preserved.items():
        bound_artifact(entry, bindings)
        original = directory / ("latest.pt" if name == "latest" else
                                "best.pt" if name == "minimum_validation_ce" else "best-accuracy.pt")
        if c.file_hash(original) != entry["sha256"]:
            raise c.CampaignError("Preservation differs from original checkpoint: " + name)
        bindings[str(original)] = entry["sha256"]
        if name in selectors and any(entry.get(k) != selectors[name][k] for k in ("sha256", "cursor", "validation")):
            raise c.CampaignError("Preserved selector evidence differs: " + name)
    durable = preserved["latest"].get("cursor", {}).get("updates")
    epochs = receipt.get("completed_epochs")
    if (not isinstance(durable, int) or durable != max(r["updates"] for r in records)
            or not 0 < durable < sum(parent["planned_phase_updates"])
            or not isinstance(receipt.get("observed_updates"), int) or receipt["observed_updates"] < durable
            or not isinstance(epochs, int) or not 0 <= epochs < 44
            or epochs != preserved["latest"]["cursor"].get("epoch")):
        raise c.CampaignError("Stopped A has invalid actual training counts")
    if (receipt.get("test_evaluated") is not False or (directory / "results.json").exists()
            or c.read_json(directory / "status.json").get("test_evaluated") is True):
        raise c.CampaignError("A is not an early-stopped arm with reserved test")
    ce = min(records, key=lambda r: r["validation"]["cross_entropy"])
    accuracy = max(records, key=lambda r: r["validation"]["top1_accuracy"])
    result = {"name": "A128fixed", "d_embed": 128, "plasticity": "fixed", "status": "accepted_early_stop",
              "full_schedule_completed": False, "updates": durable, "observed_updates": receipt["observed_updates"],
              "epochs": epochs, "selected_updates": ce["updates"], "validation": ce["validation"],
              "accuracy_selected_updates": accuracy["updates"], "accuracy_validation": accuracy["validation"],
              "selected_checkpoints": selectors, "parameter_counts": recorded["parameter_counts"],
              "test_evaluated": False, "output": str(directory),
              "elapsed_seconds": c.read_json(directory / "status.json").get("elapsed_seconds_this_invocation")}
    return result, bindings


def prepare(args):
    parent_path = args.parent / "manifest.json"
    parent = c.read_json(parent_path)
    if Path(parent["root"]).resolve() != SELF.parents[1]:
        raise c.CampaignError("Companion must run beside the unchanged parent training sources")
    c.verify_bindings(parent)
    accepted, extra = accepted_arm(parent, args.parent, args.accepted_arm_receipt)
    parity_path = args.parent / "preflight/parity.json"
    parity = c.read_json(parity_path)
    c.verify_parity(parity, parent)
    if (c.file_hash(parity_path) != parent.get("preflight_receipt_sha256")
            or parity.get("connectorch") != parent.get("preflight_connectorch_source")):
        raise c.CampaignError("Parent preflight receipt changed")
    extra[str(parity_path)] = c.file_hash(parity_path)
    smokes = []
    for arm in c.ARMS:
        directory = args.parent / "smokes" / arm[0]
        smokes.append(c.verify_arm(parent, arm, directory, smoke=True))
        for filename in ("manifest.json", "status.json", "process-status.json", "metrics.jsonl", "selected-checkpoints.json"):
            extra[str(directory / filename)] = c.file_hash(directory / filename)
    if c.read_json(args.parent / "smoke-results.json") != {"passed": True, "arms": smokes}:
        raise c.CampaignError("Parent smoke summary differs from verified artifacts")
    extra[str(args.parent / "smoke-results.json")] = c.file_hash(args.parent / "smoke-results.json")
    manifest = copy.deepcopy(parent)
    manifest.update(launch_id=str(uuid.uuid4()), created_at_utc=c.utc(), output=str(args.output),
                    parent_campaign={"path": str(parent_path), "sha256": c.file_hash(parent_path), "launch_id": parent["launch_id"]},
                    acceptance_receipt=str(args.accepted_arm_receipt), accepted_early_stopped_arm=accepted,
                    continuation_arms=[a[0] for a in c.ARMS[1:]], unequal_training_budgets=True)
    manifest["sources_sha256"][str(SELF)] = c.file_hash(SELF)
    manifest["input_files_sha256"].update(extra)
    manifest["input_files_sha256"][str(parent_path)] = c.file_hash(parent_path)
    return manifest


def matched_update_comparison(arms):
    histories = {arm["name"]: {r["updates"]: r for r in c.validation_records(arm["output"])} for arm in arms}
    common = sorted(set.intersection(*(set(rows) for rows in histories.values())))
    if not common:
        raise c.CampaignError("No common validation update across arms")
    update = common[-1]
    return {"updates": update, "selection": "latest update observed by every arm",
            "arms": {name: rows[update]["validation"] for name, rows in histories.items()},
            "qualification": "Matched optimizer updates, one seed; not a topology advantage test."}


def launch(args):
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with c.lock_file(args.output / "launch.lock"):
            if (args.output / "manifest.json").exists():
                status = c.read_json(args.output / "campaign-status.json")
                launched = c.read_json(args.output / "launch.json") if (args.output / "launch.json").exists() else {}
                if status.get("status") in ("starting", "running") and c.process_alive(launched.get("worker_pid"), launched.get("process_birth")):
                    return {"status": "already_running", "output": str(args.output)}
                return {"status": "completed" if status.get("status") == "completed" else "blocked",
                        "reason": "Existing continuation is never automatically restarted", "output": str(args.output)}
            manifest = prepare(args)
            c.atomic_json(args.output / "manifest.json", manifest)
            command = [manifest["python"], "-u", str(SELF), "--parent", str(args.parent),
                       "--accepted-arm-receipt", str(args.accepted_arm_receipt), "--output", str(args.output),
                       "--worker", "--launch-id", manifest["launch_id"]]
            record = {"command": command, "cwd": manifest["root"], "environment": c.ENVIRONMENT,
                      "launch_id": manifest["launch_id"], "created_at_utc": c.utc()}
            c.campaign_status(args.output, status="starting", phase="dispatch", launch_id=manifest["launch_id"])
            c.atomic_json(args.output / "launch.json", record)
            with (args.output / "campaign.log").open("a") as log:
                child = subprocess.Popen(command, cwd=manifest["root"], env={**os.environ, **c.ENVIRONMENT},
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(worker_pid=child.pid, process_birth=c.process_birth(child.pid))
            c.atomic_json(args.output / "launch.json", record)
            return {"status": "launched", "worker_pid": child.pid, "output": str(args.output)}
    except BlockingIOError:
        return {"status": "already_dispatching"}
    except Exception as error:
        report = {"status": "blocked", "reason": str(error), "type": type(error).__name__}
        c.atomic_json(args.output / "readiness.json", report)
        return report


def worker(args):
    try:
        with c.lock_file(args.output / "worker.lock"):
            manifest = c.read_json(args.output / "manifest.json")
            if manifest["launch_id"] != args.launch_id or c.read_json(args.output / "campaign-status.json").get("status") != "starting":
                raise c.CampaignError("Worker is not the pending authorized continuation")
            c.verify_bindings(manifest)
            # Recheck process cessation immediately before scheduling any GPU work.
            accepted_arm(c.read_json(args.parent / "manifest.json"), args.parent, args.accepted_arm_receipt)
            previous = {}
            def stop(signum, frame):
                raise c.CampaignError("Continuation interrupted by signal " + str(signum))
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, stop)
            awake = c.start_awake()
            try:
                arms = [manifest["accepted_early_stopped_arm"]]
                c.atomic_json(args.output / "partial-results.json", {"completed_arms": [], "accepted_early_stopped_arms": arms,
                              "final_test_deferred": True, "unequal_training_budgets": True})
                for arm in c.ARMS[1:]:
                    directory = args.output / "arms" / arm[0]
                    c.execute_job(manifest, "training", arm[0], directory, c.trainer_command(manifest, arm, directory))
                    arms.append({**c.verify_arm(manifest, arm, directory), "status": "completed", "full_schedule_completed": True})
                    c.atomic_json(args.output / "partial-results.json", {"completed_arms": arms[1:],
                                  "accepted_early_stopped_arms": arms[:1], "final_test_deferred": True,
                                  "unequal_training_budgets": True})
                c.verify_bindings(manifest)
                comparison = c.compare_arms(arms)
                comparison.update(unequal_training_budgets=True,
                                  qualification="A was accepted early-stopped. Retained-best contrasts use unequal training budgets.")
                result = {"status": "completed", "finished_at_utc": c.utc(), "arms": arms,
                          "full_schedule_completed_arms": [a[0] for a in c.ARMS[1:]], "accepted_early_stopped_arms": ["A128fixed"],
                          "comparison": comparison, "matched_update_comparison": matched_update_comparison(arms),
                          "test_evaluated": False, "preflight_reused_and_verified": True, "smoke_checks_reused_and_verified": True}
                c.atomic_json(args.output / "results.json", result)
                c.campaign_status(args.output, status="completed", phase="finished", completed_arms=3,
                                  accepted_early_stopped_arms=["A128fixed"], test_evaluated=False)
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
        previous = c.read_json(args.output / "campaign-status.json")
        failure = {"status": "failed", "error": str(error), "type": type(error).__name__,
                   "phase": previous.get("phase"), "arm": previous.get("arm"), "finished_at_utc": c.utc()}
        c.atomic_json(args.output / "failure.json", failure)
        c.campaign_status(args.output, **failure)
        return failure


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--accepted-arm-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--worker", action="store_true")
    p.add_argument("--launch-id", help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    for name in ("parent", "accepted_arm_receipt", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output == args.parent or args.output in args.parent.parents:
        raise SystemExit("Continuation requires a distinct new output directory")
    result = launch(args) if args.launch else worker(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    if result["status"] in ("failed", "blocked"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
