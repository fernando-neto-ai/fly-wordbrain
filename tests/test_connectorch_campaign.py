"""Campaign lifecycle and artifact gates without launching any GPU workload."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("connectorch_campaign", Path(__file__).resolve().parents[1] / "scripts/run_connectorch_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed_baseline(tmp_path):
    root, original = tmp_path / "new root", tmp_path / "original root"
    for name in campaign.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source: " + name)
    (original / "scripts").mkdir(parents=True)
    old_trainer = original / "scripts/train_ngxson.py"
    old_trainer.write_text((root / "scripts/train_ngxson.py").read_text())
    model, data = original / "model", original / "data.json"
    model.mkdir()
    reference_files = {}
    for name in ("config.json", "model.safetensors"):
        (model / name).write_text("pinned: " + name)
        reference_files[name] = {"sha256": campaign.file_hash(model / name)}
    write(data, {"train": [{"id": "t", "ids": [1, 3, 2]}],
                 "validation": [{"id": "v", "ids": [1, 4, 2]}],
                 "test": [{"id": "e", "ids": [1, 5, 2]}]})
    groups = root / "data/connectorch-groups-v1/groups.npz"
    groups.parent.mkdir(parents=True)
    groups.write_bytes(b"immutable node-type archive; parsed by separate audited trainer")
    baseline = original / "results/reference"
    graph = {"brain.w_values": "unchanged-graph", "brain.in_index": "unchanged-inputs"}
    manifest = {"reference_repository": "ngxson/fly-llm-hf", "reference_revision": campaign.REVISION,
        "debug": False, "config": {"epochs": 30, "second_epochs": 14, "batch_size": 8, "chunk_size": 32, "seed": 42},
        "planned_phase_updates": [30, 14], "frozen_buffers_sha256": graph,
        "model_path": str(model), "data_path": str(data), "data_sha256": campaign.file_hash(data),
        "reference_files": reference_files,
        "trainer_sources_sha256": {"scripts/train_ngxson.py": campaign.file_hash(old_trainer)}}
    result = {"status": "completed", "debug": False, "epochs": 44, "updates": 44, "test_evaluated": True,
              "checks": {"frozen_buffers_preserved": True, "frozen_buffers_sha256": graph}}
    write(baseline / "manifest.json", manifest)
    write(baseline / "status.json", result)
    write(baseline / "results.json", result)
    write(baseline / "launch.json", {"cwd": str(original), "command": [sys.executable, str(old_trainer)]})
    write(baseline / "process-status.json", {"status": "completed", "exit_code": 0})
    args = campaign.normalized_args(campaign.parser().parse_args([
        "--root", str(root), "--baseline-run", str(baseline), "--launch"]))
    return args


def intercept_dispatch(monkeypatch):
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=987654)
    monkeypatch.setattr(campaign.subprocess, "Popen", popen)
    monkeypatch.setattr(campaign, "process_birth", lambda pid: "known-start-time")
    monkeypatch.setattr(campaign, "process_alive", lambda pid, birth=None: pid == 987654 and birth == "known-start-time")
    return calls


def dispatched_worker(tmp_path, monkeypatch):
    args = completed_baseline(tmp_path)
    calls = intercept_dispatch(monkeypatch)
    assert campaign.launch(args)["status"] == "launched"
    assert len(calls) == 1
    manifest = campaign.read_json(args.output / "manifest.json")
    args.launch_id = manifest["launch_id"]
    args.launch, args.worker = False, True
    monkeypatch.setattr(campaign, "start_awake", lambda: None)
    return args, manifest


def test_running_baseline_only_returns_waiting_without_spawning(tmp_path, monkeypatch):
    args = completed_baseline(tmp_path)
    write(args.baseline_run / "status.json", {"status": "running", "updates": 2100})
    write(args.baseline_run / "process-status.json", {"status": "running"})
    (args.baseline_run / "results.json").unlink()
    calls = intercept_dispatch(monkeypatch)
    result = campaign.launch(args)
    assert result["status"] == "waiting_for_baseline" and not calls
    assert not (args.output / "manifest.json").exists()
    assert campaign.read_json(args.output / "readiness.json")["status"] == "waiting_for_baseline"


@pytest.mark.parametrize("field,value", [("debug", True), ("epochs", 43), ("updates", 43), ("test_evaluated", False)])
def test_incomplete_reference_cannot_launch(tmp_path, monkeypatch, field, value):
    args = completed_baseline(tmp_path)
    result = campaign.read_json(args.baseline_run / "results.json")
    result[field] = value
    write(args.baseline_run / "results.json", result)
    calls = intercept_dispatch(monkeypatch)
    assert campaign.launch(args)["status"] == "blocked" and not calls


def test_failed_baseline_is_blocked_without_killing_or_restarting(tmp_path, monkeypatch):
    args = completed_baseline(tmp_path)
    write(args.baseline_run / "process-status.json", {"status": "failed", "exit_code": 1})
    calls = intercept_dispatch(monkeypatch)
    assert campaign.launch(args)["status"] == "blocked"
    assert not calls


def test_dispatch_is_idempotent_and_preserves_structured_paths(tmp_path, monkeypatch):
    args = completed_baseline(tmp_path)
    calls = intercept_dispatch(monkeypatch)
    first = campaign.launch(args)
    assert first["status"] == "launched"
    assert campaign.launch(args)["status"] == "already_running"
    assert len(calls) == 1
    command, options = calls[0]
    assert command[command.index("--root")+1] == str(args.root)
    assert " " in command[command.index("--root")+1]
    assert options["start_new_session"] is True
    assert options["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert "shell" not in options


def test_existing_dead_campaign_is_never_automatically_restarted(tmp_path, monkeypatch):
    args = completed_baseline(tmp_path)
    calls = intercept_dispatch(monkeypatch)
    assert campaign.launch(args)["status"] == "launched"
    monkeypatch.setattr(campaign, "process_alive", lambda *args: False)
    assert campaign.launch(args)["status"] == "blocked" and len(calls) == 1


def test_reference_helpers_are_checked_in_both_roots(tmp_path, monkeypatch):
    args = completed_baseline(tmp_path)
    (args.root / "scripts/train_ngxson.py").write_text("different initialization helper")
    calls = intercept_dispatch(monkeypatch)
    result = campaign.launch(args)
    assert result["status"] == "blocked" and "Staged reference helper" in result["reason"]
    assert not calls


def test_environment_python_symlink_is_not_resolved_to_system_interpreter(tmp_path):
    target = tmp_path / "system-python"
    target.write_text("placeholder")
    venv = tmp_path / "venv/bin/python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(target)
    args = campaign.normalized_args(campaign.parser().parse_args([
        "--baseline-run", str(tmp_path / "baseline"), "--python", str(venv), "--launch"]))
    assert args.python == str(venv) and args.python != str(venv.resolve())


def parity_receipt(manifest):
    return {"passed": True, "sources_unchanged": True,
        "data_file_sha256": manifest["data_sha256"], "groups_file_sha256": manifest["groups_sha256"],
        "source_sha256": {"scripts/audit_connectorch.py": manifest["sources_sha256"][str(Path(manifest["root"]) / "scripts/audit_connectorch.py")]},
        "connectorch": {"revision": "pinned-library", "python_files_sha256": {"nn/model.py": "fixed-source"}},
        "comparisons": {key: {"passed": True} for key in (
            "upstream_cpu_vs_connectorch_d128_fixed_cpu", "upstream_cpu_vs_connectorch_d128_fixed_mps",
            "connectorch_d32_bounded10_cpu_vs_mps")}}


def arm_artifacts(manifest, arm, directory, smoke):
    name, width, plasticity = arm
    directory.mkdir(parents=True, exist_ok=True)
    audit = {"frozen_buffers_preserved": True, "frozen_buffers_sha256": manifest["baseline_frozen_buffers_sha256"],
             "adaptation": {"zero_edges_preserved": True, "nonzero_signs_preserved": True,
                            "source_type_theta": {"rms": .01}, "destination_type_theta": {"rms": .02},
                            "edge_displacement": {"rms": .001}}}
    expected_config = {"epochs": 30, "second_epochs": 14, "batch_size": 8, "chunk_size": 32, "seed": 42,
        "d_embed": width, "plasticity": plasticity, "readout_rank": 0, "history_length": 8,
        "skip_final_test": True, "device": "mps", "eval_interval_updates": 100,
        "max_updates": 8 if smoke else None, "eval_limit": 8 if smoke else None}
    recorded = {"config": expected_config, "data_sha256": manifest["data_sha256"], "groups_sha256": manifest["groups_sha256"],
        "trainer_sources_sha256": {relative: manifest["sources_sha256"][str(Path(manifest["root"]) / relative)]
                                    for relative in campaign.TRAINER_SOURCE_FILES},
        "connectorch_source": manifest["preflight_connectorch_source"],
        "parameter_counts": {"total": 100}, "planned_phase_updates": manifest["planned_phase_updates"]}
    updates = 8 if smoke else sum(manifest["planned_phase_updates"])
    metric = {"stories": manifest["smoke_validation_stories" if smoke else "validation_stories"],
              "tokens": manifest["smoke_validation_tokens" if smoke else "validation_tokens"],
              "cross_entropy": {"A128fixed": 5., "B32fixed": 5.3, "C128bounded": 4.9, "D32bounded": 5.}[name],
              "top1_accuracy": .3}
    record = {"event": "validation", "updates": updates, "validation": metric, "model_audit": audit}
    selectors = {}
    for key, filename in (("minimum_validation_ce", "best.pt"), ("maximum_validation_accuracy", "best-accuracy.pt")):
        checkpoint = directory / filename
        checkpoint.write_text("fixture retained checkpoint: " + key)
        selectors[key] = {"path": str(checkpoint.resolve()), "sha256": campaign.file_hash(checkpoint),
            "cursor": {"updates": updates}, "validation": metric,
            "frozen_buffers_preserved": True, "frozen_buffers_sha256": manifest["baseline_frozen_buffers_sha256"]}
    write(directory / "selected-checkpoints.json", {"format_version": 1, "selectors": selectors})
    write(directory / "manifest.json", recorded)
    (directory / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    write(directory / "process-status.json", {"status": "completed", "exit_code": 0})
    if smoke:
        write(directory / "status.json", {"status": "debug_stopped", "debug": True, "updates": 8, "test_evaluated": False})
        write(directory / "initial-gradients.json", {key: {"present": True, "finite": True, "nonzero_values": 5}
            for key in ("brain.edge_theta_source", "brain.edge_theta_destination")})
    else:
        result = {"status": "completed", "debug": False, "updates": updates, "epochs": 44,
                  "test_evaluated": False, "test_deferred": True,
                  "checks": {**audit, "selected_checkpoint_audit": audit},
                  "selected": {"cross_entropy": metric["cross_entropy"], "updates": updates},
                  "selected_accuracy": {"top1_accuracy": metric["top1_accuracy"], "updates": updates},
                  "selected_checkpoints": selectors}
        write(directory / "status.json", result)
        write(directory / "results.json", result)


def simulated_jobs(monkeypatch, fail_at=None):
    calls = []
    def execute(manifest, phase, name, directory, command):
        campaign.verify_bindings(manifest)
        calls.append((phase, name, command))
        campaign.campaign_status(manifest["output"], status="running", phase=phase, arm=name)
        if (phase, name) == fail_at:
            raise campaign.CampaignError("injected failure")
        if phase == "preflight":
            directory.mkdir(parents=True, exist_ok=True)
            write(directory / "parity.json", parity_receipt(manifest))
        else:
            arm = next(arm for arm in campaign.ARMS if arm[0] == name)
            arm_artifacts(manifest, arm, directory, smoke=phase == "smoke")
    monkeypatch.setattr(campaign, "execute_job", execute)
    return calls


def test_campaign_is_serial_smokes_precede_training_and_every_arm_defers_test(tmp_path, monkeypatch):
    args, manifest = dispatched_worker(tmp_path, monkeypatch)
    calls = simulated_jobs(monkeypatch)
    result = campaign.worker(args)
    assert result["status"] == "completed"
    assert [(phase, arm) for phase, arm, _ in calls] == [("preflight", "parity")] + [
        (phase, arm[0]) for phase in ("smoke", "training") for arm in campaign.ARMS]
    for phase, _, command in calls[1:]:
        assert "--skip-final-test" in command and "--resume" not in command
        assert command[command.index("--history-length")+1] == "8"
        assert command[command.index("--readout-rank")+1] == "0"
        assert ("--max-updates" in command) == (phase == "smoke")
    comparison = result["comparison"]
    assert comparison["compression_ce_penalty_fixed"] == pytest.approx(.3)
    assert comparison["compression_ce_penalty_plastic"] == pytest.approx(.1)
    assert comparison["compression_plasticity_interaction_ce"] == pytest.approx(-.2)
    assert comparison["needs_replication_before_geometry_claim"]
    assert result["test_evaluated"] is False
    assert campaign.read_json(args.output / "campaign-status.json")["completed_arms"] == 4


def test_failure_halts_campaign_before_remaining_arms(tmp_path, monkeypatch):
    args, _ = dispatched_worker(tmp_path, monkeypatch)
    calls = simulated_jobs(monkeypatch, fail_at=("smoke", "B32fixed"))
    result = campaign.worker(args)
    assert result["status"] == "failed" and result["arm"] == "B32fixed"
    assert len(calls) == 3 and not any(phase == "training" for phase, _, _ in calls)
    assert not (args.output / "results.json").exists()
    assert (args.output / "failure.json").exists()
    assert campaign.launch(args)["status"] == "blocked"


def test_changed_bound_source_blocks_worker_before_any_job(tmp_path, monkeypatch):
    args, _ = dispatched_worker(tmp_path, monkeypatch)
    calls = simulated_jobs(monkeypatch)
    (args.root / "scripts/train_connectorch.py").write_text("different training recipe")
    result = campaign.worker(args)
    assert result["status"] == "failed" and "bound campaign file changed" in result["error"]
    assert not calls


def test_missing_parity_comparisons_cannot_pass_vacuously(tmp_path, monkeypatch):
    _, manifest = dispatched_worker(tmp_path, monkeypatch)
    parity = parity_receipt(manifest)
    parity["comparisons"] = {}
    with pytest.raises(campaign.CampaignError, match="required comparisons"):
        campaign.verify_parity(parity, manifest)


def test_plastic_smoke_rejects_unwired_gain_and_full_arm_rejects_test_access(tmp_path, monkeypatch):
    args, manifest = dispatched_worker(tmp_path, monkeypatch)
    manifest["preflight_connectorch_source"] = parity_receipt(manifest)["connectorch"]
    arm, directory = campaign.ARMS[3], args.output / "test-smoke"
    arm_artifacts(manifest, arm, directory, smoke=True)
    gradients = campaign.read_json(directory / "initial-gradients.json")
    gradients["brain.edge_theta_destination"]["nonzero_values"] = 0
    write(directory / "initial-gradients.json", gradients)
    with pytest.raises(campaign.CampaignError, match="gradient was missing, zero or nonfinite"):
        campaign.verify_arm(manifest, arm, directory, smoke=True)
    full = args.output / "test-full"
    arm_artifacts(manifest, arm, full, smoke=False)
    result = campaign.read_json(full / "results.json")
    result["test_evaluated"] = True
    write(full / "results.json", result)
    with pytest.raises(campaign.CampaignError, match="reserved final test"):
        campaign.verify_arm(manifest, arm, full)


def test_parallel_launch_lock_returns_without_spawning(tmp_path, monkeypatch):
    args = completed_baseline(tmp_path)
    args.output.mkdir(parents=True)
    calls = intercept_dispatch(monkeypatch)
    with campaign.lock_file(args.output / "launch.lock"):
        result = campaign.launch(args)
    assert result["status"] == "already_dispatching" and not calls


def accepted_baseline(tmp_path):
    args = completed_baseline(tmp_path)
    (args.baseline_run / "results.json").unlink()
    write(args.baseline_run / "status.json", {"status": "stopped_by_user", "updates": 20})
    write(args.baseline_run / "process-status.json", {"status": "failed", "exit_code": -15})
    # SIGTERM can leave an honest failure artifact; acceptance is a separate disposition.
    write(args.baseline_run / "failure.json", {"status": "failed", "reason": "User requested SIGTERM"})
    cessation = {"confirmed": True, "processes": [{"pid": 654321, "birth": "original birth", "alive": False}]}
    stop_path = args.baseline_run / "stop-receipt.json"
    write(stop_path, {"status": "stopped_by_user", "process_cessation": cessation})
    manifest_path = args.baseline_run / "manifest.json"
    manifest = campaign.read_json(manifest_path)
    checkpoints = {}
    for name in ("minimum_validation_ce", "maximum_validation_accuracy"):
        path = args.baseline_run / (name + ".pt")
        path.write_text("retained " + name)
        checkpoints[name] = {"path": str(path), "sha256": campaign.file_hash(path),
            "cursor": {"updates": 17}, "validation": {"cross_entropy": 4.5, "top1_accuracy": .32},
            "frozen_buffers_preserved": True, "frozen_buffers_sha256": manifest["frozen_buffers_sha256"]}
    receipt = {"format_version": 1, "status": "accepted_early_stop", "accepted_by": "user",
        "instruction": "Stop the training here and move forward into the next experiments",
        "manifest": {"path": str(manifest_path), "sha256": campaign.file_hash(manifest_path)},
        "stop_receipt": {"path": str(stop_path), "sha256": campaign.file_hash(stop_path)},
        "process_cessation": cessation, "preserved_checkpoints": checkpoints}
    args.accepted_baseline_receipt = args.baseline_run / "accepted-baseline.json"
    write(args.accepted_baseline_receipt, receipt)
    return args


def test_explicit_user_accepted_stop_launches_without_fabricating_completed_baseline(tmp_path, monkeypatch):
    args = accepted_baseline(tmp_path)
    calls = intercept_dispatch(monkeypatch)
    assert campaign.baseline_readiness(args.baseline_run)["status"] == "blocked"
    assert campaign.launch(args)["status"] == "launched" and len(calls) == 1
    manifest = campaign.read_json(args.output / "manifest.json")
    assert manifest["baseline_acceptance_mode"] == "user_accepted_early_stop"
    assert "44_epochs" not in manifest["baseline_gates"]
    assert not (args.baseline_run / "results.json").exists()
    assert campaign.read_json(args.baseline_run / "process-status.json")["exit_code"] == -15
    assert str(args.accepted_baseline_receipt) in manifest["baseline_files_sha256"]
    campaign.verify_bindings(manifest)


@pytest.mark.parametrize("damage", ["checkpoint", "manifest", "stop_receipt", "missing_selector", "wrong_graph", "no_acceptance", "alive", "omitted_process", "running_status"])
def test_accepted_stop_cannot_bypass_artifact_or_process_evidence(tmp_path, monkeypatch, damage):
    args = accepted_baseline(tmp_path)
    receipt = campaign.read_json(args.accepted_baseline_receipt)
    if damage in ("checkpoint", "manifest", "stop_receipt"):
        entry = receipt["preserved_checkpoints"]["minimum_validation_ce"] if damage == "checkpoint" else receipt[damage]
        Path(entry["path"]).write_text("tampered")
    elif damage == "missing_selector":
        del receipt["preserved_checkpoints"]["maximum_validation_accuracy"]
    elif damage == "wrong_graph":
        receipt["preserved_checkpoints"]["maximum_validation_accuracy"]["frozen_buffers_sha256"] = {"wrong": "graph"}
    elif damage == "no_acceptance":
        receipt["accepted_by"] = "automatic_quality_threshold"
    elif damage == "omitted_process":
        write(args.baseline_run / "process-status.json", {"status": "failed", "worker_pid": 765432, "exit_code": -15})
    elif damage == "running_status":
        write(args.baseline_run / "status.json", {"status": "running", "updates": 20})
    write(args.accepted_baseline_receipt, receipt)
    calls = intercept_dispatch(monkeypatch)
    if damage == "alive":
        monkeypatch.setattr(campaign, "process_alive", lambda *args: True)
    assert campaign.launch(args)["status"] == "blocked" and not calls


def test_accuracy_winner_must_match_validation_history_and_retained_file(tmp_path, monkeypatch):
    args, manifest = dispatched_worker(tmp_path, monkeypatch)
    manifest["preflight_connectorch_source"] = parity_receipt(manifest)["connectorch"]
    directory, arm = args.output / "test-dual", campaign.ARMS[0]
    arm_artifacts(manifest, arm, directory, smoke=False)
    result = campaign.verify_arm(manifest, arm, directory)
    assert result["accuracy_validation"]["top1_accuracy"] == .3
    (directory / "best-accuracy.pt").write_text("wrong retained model")
    with pytest.raises(campaign.CampaignError, match="selector mismatch: maximum_validation_accuracy"):
        campaign.verify_arm(manifest, arm, directory)


def test_branch_mapping_is_bound_and_attached_to_each_arm(tmp_path, monkeypatch):
    args = accepted_baseline(tmp_path)
    mapping = {"repository": "https://github.com/fernando-neto-ai/fly-wordbrain",
               "arms": {name: {"branch": "experiments/" + name, "commit": "a" * 40} for name, _, _ in campaign.ARMS}}
    args.experiment_branches = args.root / "branches.json"
    write(args.experiment_branches, mapping)
    intercept_dispatch(monkeypatch)
    assert campaign.launch(args)["status"] == "launched"
    manifest = campaign.read_json(args.output / "manifest.json")
    assert manifest["experiment_branches"] == mapping
    args.experiment_branches.write_text("changed mapping")
    with pytest.raises(campaign.CampaignError, match="bound campaign file changed"):
        campaign.verify_bindings(manifest)
