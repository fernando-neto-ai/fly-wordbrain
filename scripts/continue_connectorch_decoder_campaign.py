#!/usr/bin/env python3
"""Deliberately hand off to F after a preserved, accepted early stop of E.

The stopped parent and all existing training sources stay unchanged. Existing
rank128 preflight and both smoke runs are reverified, not rerun. Only the full F
arm starts in the fresh continuation directory; an interrupted F never retries.
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
import uuid

SELF = Path(__file__).resolve()
ROOT = SELF.parents[1]
spec = importlib.util.spec_from_file_location("original_decoder_campaign", SELF.with_name("run_connectorch_decoder_campaign.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
c, require = d.c, d.require
E, F = d.ARMS


def verified_checkpoint(path, entry, recorded, selector=None):
    """Cross-check contents after cessation: atomic files are not a transaction."""
    import torch
    trainer_spec = importlib.util.spec_from_file_location("decoder_handoff_checkpoint_helpers", ROOT / "scripts/train_connectorch.py")
    trainer = importlib.util.module_from_spec(trainer_spec)
    trainer_spec.loader.exec_module(trainer)
    require(c.file_hash(path) == entry["sha256"], "Accepted checkpoint changed before loading")
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    require(saved.get("format_version") == 1 and saved["manifest"] == recorded, "Accepted checkpoint manifest differs")
    require(saved["cursor"] == entry["cursor"], "Accepted checkpoint cursor differs from its receipt")
    require(saved["frozen_buffers_sha256"] == recorded["frozen_buffers_sha256"] == entry["frozen_buffers_sha256"]
            and saved["model_audit"]["frozen_buffers_preserved"] is True
            and saved["model_audit"]["frozen_buffers_sha256"] == recorded["frozen_buffers_sha256"],
            "Accepted checkpoint graph audit differs")
    require(trainer.parameter_audit(saved["parameters"], recorded["initial_parameter_audit"]) == saved["parameter_audit"],
            "Accepted checkpoint parameter hashes differ from actual values")
    if selector is not None:
        best = saved["best" if selector == "minimum_validation_ce" else "best_accuracy"]
        require(all(best[key] == entry["cursor"][key] for key in ("updates", "epoch"))
                and all(best[key] == entry["validation"][key] for key in ("cross_entropy", "top1_accuracy")),
                "Accepted checkpoint is not its declared validation winner")
    require(c.file_hash(path) == entry["sha256"], "Accepted checkpoint changed while loading")
    return {"sha256": entry["sha256"], "cursor": saved["cursor"], "parameters_verified": True,
            "frozen_buffers_verified": True}


def accepted_e(parent, parent_directory, receipt_path):
    directory = parent_directory / "arms" / E[0]
    receipt = c.read_json(receipt_path)
    require(receipt_path == directory / "accepted-early-stop.json", "E acceptance belongs to another directory")
    require(receipt.get("format_version") == 1 and receipt.get("status") == "accepted_early_stop"
            and receipt.get("arm") == E[0] and receipt.get("accepted_by") in ("user", "agent_under_standing_user_authorization")
            and receipt.get("full_schedule_completed") is False and receipt.get("test_evaluated") is False,
            "A separate accepted E early-stop receipt is required")
    bindings = {}
    d.bound(receipt_path, c.file_hash(receipt_path), bindings)
    for path, digest in receipt.get("artifact_sha256", {}).items():
        d.bound(path, digest, bindings)
    for name in ("manifest.json", "launch.json", "metrics.jsonl", "selected-checkpoints.json"):
        require(str(directory / name) in bindings, "E acceptance omitted an immutable arm artifact: " + name)
    stop = c.read_json(d.bound_entry(receipt["stop_receipt"], bindings))
    require(Path(receipt["stop_receipt"]["path"]).resolve() == directory / "stop-receipt.json"
            and stop.get("process_cessation") == receipt.get("process_cessation"), "E stop and acceptance cessation differ")
    d.verify_cessation(receipt)
    if receipt.get("decision_receipt"):
        d.bound_entry(receipt["decision_receipt"], bindings)
    stopped = {row["pid"] for row in receipt["process_cessation"]["processes"]}
    known_workers = set()
    for name, key in (("status.json", "pid"), ("process-status.json", "worker_pid")):
        path = directory / name
        record = c.read_json(path)
        if isinstance(record.get(key), int):
            known_workers.add(record[key])
        d.bound(path, c.file_hash(path), bindings)
    require(bool(known_workers) and known_workers <= stopped, "E cessation omits the recorded training worker")
    require(not (directory / "results.json").exists(), "E already has a full-run result; this handoff requires an early stop")
    require(c.read_json(directory / "status.json").get("test_evaluated") is not True, "E accessed its reserved test")
    recorded = c.read_json(directory / "manifest.json")
    require(all(recorded.get("config", {}).get(key) == value for key, value in d.arm_config(E).items())
            and recorded.get("debug") is False and recorded.get("parameter_counts") == d.expected_counts(E)
            and recorded.get("planned_phase_updates") == parent["planned_phase_updates"] == d.PLANNED_UPDATES,
            "Accepted E architecture or schedule differs from the declared parent arm")
    require(recorded.get("reference_repository") == "ngxson/fly-llm-hf" and recorded.get("reference_revision") == c.REVISION
            and recorded.get("data_sha256") == parent["data_sha256"] and recorded.get("groups_sha256") == parent["groups_sha256"]
            and recorded.get("frozen_buffers_sha256") == parent["baseline_frozen_buffers_sha256"]
            and recorded.get("connectorch_source") == parent["preflight_connectorch_source"], "E model/data/group/package identity differs")
    require(set(recorded.get("trainer_sources_sha256", {})) == c.TRAINER_SOURCE_FILES, "E trainer source inventory differs")
    for relative, digest in recorded["trainer_sources_sha256"].items():
        require(parent["sources_sha256"].get(str(Path(parent["root"]) / relative)) == digest, "E used unbound source: " + relative)
    launched = c.read_json(directory / "launch.json")
    expected_git = {"repository": parent["experiment_branches"]["repository"], **parent["experiment_branches"]["arms"][E[0]]}
    require(launched.get("campaign_launch_id") == parent["launch_id"] and launched.get("experiment_git") == expected_git,
            "E launch differs from parent campaign or branch provenance")
    records = d.validation_records(directory)
    require(bool(records), "E has no complete validation records")
    for row in records:
        metric = row["validation"]
        require(metric.get("stories") == parent["validation_stories"] and metric.get("tokens") == parent["validation_tokens"]
                and math.isfinite(metric["cross_entropy"]) and metric["cross_entropy"] >= 0
                and 0 <= metric.get("correct", -1) <= metric["tokens"]
                and abs(metric["top1_accuracy"] - metric["correct"] / metric["tokens"]) < 1e-12,
                "E validation metrics or population differ")
        c.graph_audit(row["model_audit"], parent)
    native = c.verify_selected_checkpoints(directory, records, parent)
    preserved = receipt.get("selected_checkpoints", {})
    require(preserved.get("format_version") == 1 and set(preserved.get("selectors", {})) == set(d.SELECTORS),
            "E acceptance must preserve both validation winners")
    audits = {}
    for selector, entry in preserved["selectors"].items():
        path = d.bound_entry(entry, bindings)
        require(path == directory / "stopped-checkpoints" / d.SELECTORS[selector], "Unexpected preserved E winner path")
        d.bound_entry(native[selector], bindings)
        require(entry.get("frozen_buffers_preserved") is True
                and all(entry[key] == native[selector][key] for key in ("sha256", "cursor", "validation", "frozen_buffers_sha256")),
                "Preserved E winner and native selector disagree")
        choose, key = (min, "cross_entropy") if selector == "minimum_validation_ce" else (max, "top1_accuracy")
        selected_row = choose(records, key=lambda row: row["validation"][key])
        require(entry["cursor"]["epoch"] == selected_row["epoch"], "E selected epoch differs from its validation record")
        audits[selector] = verified_checkpoint(path, entry, recorded, selector)
    latest = receipt["latest_checkpoint"]
    latest_path = d.bound_entry(latest, bindings)
    require(latest_path == directory / "stopped-checkpoints/latest.pt" and latest.get("frozen_buffers_preserved") is True,
            "Expected an immutable preserved E latest checkpoint")
    d.bound(directory / "latest.pt", latest["sha256"], bindings)
    audits["latest"] = verified_checkpoint(latest_path, latest, recorded)
    durable, observed, epochs = receipt.get("durable_updates"), receipt.get("observed_updates"), receipt.get("completed_epochs")
    require(isinstance(durable, int) and isinstance(observed, int) and isinstance(epochs, int)
            and 0 < durable <= observed < sum(d.PLANNED_UPDATES) and 0 <= epochs < 44
            and latest["cursor"]["updates"] == durable == max(row["updates"] for row in records)
            and latest["cursor"]["epoch"] == epochs, "E stop cursor is incoherent with its completed validation history")
    for path, digest in bindings.items():
        d.bound(path, digest, {})
    summary = {"name": E[0], "status": "accepted_early_stop", "output": str(directory), "d_embed": 32,
               "readout_rank": 128, "history_length": 8, "plasticity": "fixed", "full_schedule_completed": False,
               "updates": durable, "observed_updates": observed, "epochs": epochs, "test_evaluated": False,
               "selected_checkpoints": preserved["selectors"], "parameter_counts": recorded["parameter_counts"],
               "validation": native["minimum_validation_ce"]["validation"], "accuracy_validation": native["maximum_validation_accuracy"]["validation"],
               "checkpoint_content_audits": audits, "experiment_git": expected_git}
    return summary, bindings


def prepare(args):
    d.host_guard()
    parent_path = args.parent / "manifest.json"
    parent = c.read_json(parent_path)
    require(Path(parent["root"]).resolve() == ROOT, "Handoff must run beside the parent's unchanged training sources")
    require(parent.get("planned_phase_updates") == d.PLANNED_UPDATES
            and parent.get("arms") == [{"name": arm[0], **d.arm_config(arm), "parameter_counts": d.expected_counts(arm)} for arm in d.ARMS],
            "Parent is not the original two-arm decoder campaign")
    c.verify_bindings(parent)
    parent_launch = c.read_json(args.parent / "launch.json")
    require(parent_launch.get("manifest_sha256") == c.file_hash(parent_path)
            and parent_launch.get("launch_id") == parent["launch_id"], "Parent launch/manifest binding differs")
    for filename in ("manifest.json", "launch.json", "process-status.json", "status.json", "metrics.jsonl", "latest.pt"):
        require(not (args.parent / "arms" / F[0] / filename).exists(), "Parent F already started; refusing a duplicate full run")
    summary, extra = accepted_e(parent, args.parent, args.accepted_arm_receipt)
    for path in (parent_path, args.parent / "launch.json"):
        d.bound(path, c.file_hash(path), extra)
    parity_path = args.parent / "preflight/parity.json"
    parity = c.read_json(d.bound(parity_path, c.file_hash(parity_path), extra))
    require(parity == c.read_json(parent["preflight_receipt"]["path"]), "Parent preflight copy differs from its bound original")
    d.verify_preflight(parity, parent)
    smokes = []
    for arm in d.ARMS:
        directory = args.parent / "smokes" / arm[0]
        smokes.append(d.verify_arm(parent, arm, directory, smoke=True))
        for filename in ("manifest.json", "launch.json", "status.json", "process-status.json", "metrics.jsonl", "selected-checkpoints.json", "initial-gradients.json", "best.pt", "best-accuracy.pt", "latest.pt"):
            path = directory / filename
            d.bound(path, c.file_hash(path), extra)
    smoke_path = args.parent / "smoke-results.json"
    require(c.read_json(smoke_path) == {"passed": True, "arms": smokes}, "Parent smoke summary differs from verified smoke artifacts")
    d.bound(smoke_path, c.file_hash(smoke_path), extra)
    manifest = copy.deepcopy(parent)
    manifest.update(launch_id=str(uuid.uuid4()), created_at_utc=c.utc(), output=str(args.output),
                    authorized_launcher={"pid": os.getpid(), "birth": c.process_birth(os.getpid())},
                    parent_campaign={"path": str(parent_path), "sha256": c.file_hash(parent_path), "launch_id": parent["launch_id"]},
                    accepted_arm_receipt={"path": str(args.accepted_arm_receipt), "sha256": extra[str(args.accepted_arm_receipt)]},
                    accepted_early_stopped_arm=summary, continuation_arms=[F[0]], unequal_training_budgets=True,
                    preflight_reused_and_verified=True, smoke_checks_reused_and_verified=True,
                    initialization="F starts from scratch with unchanged seed42/rank128 recipe; E weights are retained for comparison, not resumed",
                    stop_policy="An interrupted or plateau-stopped F halts this handoff; no automatic retries or new experiments")
    manifest["sources_sha256"][str(SELF)] = c.file_hash(SELF)
    manifest["input_files_sha256"].update(extra)
    c.verify_bindings(manifest)
    return manifest


def launch(args):
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with c.lock_file(args.output / "launch.lock"):
            if (args.output / "manifest.json").exists():
                state = c.read_json(args.output / "campaign-status.json") if (args.output / "campaign-status.json").exists() else {}
                launched = c.read_json(args.output / "launch.json") if (args.output / "launch.json").exists() else {}
                if state.get("status") in ("starting", "running") and c.process_alive(launched.get("worker_pid"), launched.get("process_birth")):
                    return {"status": "already_running", "output": str(args.output)}
                return {"status": "completed" if state.get("status") == "completed" else "blocked", "reason": "Existing F handoff is never automatically restarted", "output": str(args.output)}
            require(not (args.output / "arms").exists(), "Continuation output already contains arms")
            manifest = prepare(args)
            c.atomic_json(args.output / "manifest.json", manifest)
            argv = [manifest["python"], "-u", str(SELF), "--parent", str(args.parent), "--accepted-arm-receipt", str(args.accepted_arm_receipt),
                    "--output", str(args.output), "--worker", "--launch-id", manifest["launch_id"]]
            record = {"command": argv, "cwd": manifest["root"], "environment": c.ENVIRONMENT, "created_at_utc": c.utc(),
                      "launch_id": manifest["launch_id"], "manifest_sha256": c.file_hash(args.output / "manifest.json")}
            c.campaign_status(args.output, status="starting", phase="dispatch", launch_id=manifest["launch_id"])
            c.atomic_json(args.output / "launch.json", record)
            with (args.output / "campaign.log").open("a") as log:
                child = subprocess.Popen(argv, cwd=manifest["root"], env={**os.environ, **c.ENVIRONMENT}, stdin=subprocess.DEVNULL,
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
            launched = c.read_json(args.output / "launch.json")
            require(args.launch_id == manifest["launch_id"] and manifest.get("continuation_arms") == [F[0]]
                    and launched.get("manifest_sha256") == c.file_hash(args.output / "manifest.json")
                    and c.read_json(args.output / "campaign-status.json").get("status") == "starting", "Worker does not match the pending F handoff")
            c.verify_bindings(manifest)
            d.host_guard(manifest)
            parent = c.read_json(args.parent / "manifest.json")
            summary, _ = accepted_e(parent, args.parent, args.accepted_arm_receipt)
            require(summary == manifest["accepted_early_stopped_arm"], "E acceptance changed after F dispatch")
            previous = {}
            def stop(signum, frame):
                raise c.CampaignError("F handoff interrupted by signal " + str(signum))
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, stop)
            awake = c.start_awake()
            try:
                partial = {"completed_arms": [], "accepted_early_stopped_arms": [summary], "baseline": manifest["baseline_summary"],
                           "final_test_deferred": True, "unequal_training_budgets": True}
                c.atomic_json(args.output / "partial-results.json", partial)
                d.host_guard(manifest)
                directory = args.output / "arms" / F[0]
                c.execute_job(manifest, "training", F[0], directory, d.trainer_command(manifest, F, directory))
                finished = d.verify_arm(manifest, F, directory)
                c.verify_bindings(manifest)
                partial["completed_arms"] = [finished]
                c.atomic_json(args.output / "partial-results.json", partial)
                result = {"status": "completed", "arms": [summary, finished], "baseline": manifest["baseline_summary"],
                          "accepted_early_stopped_arms": [E[0]], "full_schedule_completed_arms": [F[0]],
                          "preflight_reused_and_verified": True, "smoke_checks_reused_and_verified": True,
                          "test_evaluated": False, "unequal_training_budgets": True, "finished_at_utc": c.utc(),
                          "qualification": "E was accepted early-stopped; F completed the full schedule. Retained-best comparisons use unequal budgets; compare common logged updates too."}
                c.atomic_json(args.output / "results.json", result)
                c.campaign_status(args.output, status="completed", phase="finished", completed_arms=1,
                                  accepted_early_stopped_arms=[E[0]], test_evaluated=False)
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
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--accepted-arm-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--worker", action="store_true")
    p.add_argument("--launch-id", help=argparse.SUPPRESS)
    return p


def normalized_args(args):
    for name in ("parent", "accepted_arm_receipt", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    require(args.output != args.parent and args.output not in args.parent.parents and args.parent not in args.output.parents,
            "Use a distinct sibling continuation directory, not the parent's tree")
    return args


def main():
    args = normalized_args(parser().parse_args())
    result = launch(args) if args.launch else worker(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    if result["status"] in ("failed", "blocked"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
