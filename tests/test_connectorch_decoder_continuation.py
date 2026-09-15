"""Future F handoff preparation with synthetic CPU checkpoints; no training."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decoder_handoff", ROOT / "scripts/continue_connectorch_decoder_campaign.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
trainer_spec = importlib.util.spec_from_file_location("handoff_fixture_audits", ROOT / "scripts/train_connectorch.py")
trainer = importlib.util.module_from_spec(trainer_spec)
trainer_spec.loader.exec_module(trainer)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def entry(path):
    return {"path": str(path), "sha256": h.c.file_hash(path)}


def accepted_fixture(tmp_path, monkeypatch):
    parent_dir = tmp_path / "parent"
    directory = parent_dir / "arms" / h.E[0]
    directory.mkdir(parents=True)
    graph = {"brain.w_values": "fixture-graph"}
    source = {str(ROOT / name): h.c.file_hash(ROOT / name) for name in h.d.SOURCE_FILES}
    mapping = {"repository": "https://github.com/fernando-neto-ai/fly-wordbrain",
               "arms": {arm[0]: {"branch": "exp/" + arm[0], "commit": "a" * 40} for arm in h.d.ARMS}}
    parent = {"root": str(ROOT), "output": str(parent_dir), "python": "fixture-python", "launch_id": "parent-id",
              "model": "/fixture-model", "data": "/fixture-data", "groups": "/fixture-groups",
              "data_sha256": "data", "groups_sha256": "groups", "planned_phase_updates": h.d.PLANNED_UPDATES,
              "sources_sha256": source, "input_files_sha256": {}, "baseline_files_sha256": {},
              "baseline_frozen_buffers_sha256": graph, "preflight_connectorch_source": {"revision": "pinned"},
              "experiment_branches": mapping, "baseline_summary": {"name": "B32fixed", "status": "accepted_early_stop"},
              "validation_stories": 100, "validation_tokens": 21874,
              "arms": [{"name": arm[0], **h.d.arm_config(arm), "parameter_counts": h.d.expected_counts(arm)} for arm in h.d.ARMS]}
    initial = {"fixture.weight": torch.zeros(2)}
    recorded = {"config": h.d.arm_config(h.E), "debug": False, "parameter_counts": h.d.expected_counts(h.E),
                "planned_phase_updates": h.d.PLANNED_UPDATES, "reference_repository": "ngxson/fly-llm-hf", "reference_revision": h.c.REVISION,
                "data_sha256": "data", "groups_sha256": "groups", "frozen_buffers_sha256": graph,
                "connectorch_source": {"revision": "pinned"}, "initial_parameter_audit": trainer.parameter_audit(initial),
                "trainer_sources_sha256": {name: source[str(ROOT / name)] for name in h.c.TRAINER_SOURCE_FILES}}
    write(directory / "manifest.json", recorded)
    write(directory / "launch.json", {"campaign_launch_id": "parent-id", "experiment_git": {"repository": mapping["repository"], **mapping["arms"][h.E[0]]}})
    write(directory / "status.json", {"status": "running", "pid": 123456, "updates": 307})
    write(directory / "process-status.json", {"status": "failed", "worker_pid": 123456, "exit_code": -15})
    audit = {"frozen_buffers_preserved": True, "frozen_buffers_sha256": graph,
             "adaptation": {"zero_edges_preserved": True, "nonzero_signs_preserved": True}}
    metrics = [
        {"event": "validation", "updates": step, "epoch": epoch,
         "validation": {"cross_entropy": ce, "correct": correct, "tokens": 21874, "stories": 100, "top1_accuracy": correct / 21874},
         "model_audit": audit}
        for step, epoch, ce, correct in ((100, 0, 4., 6000), (200, 1, 5., 7000), (300, 1, 5.1, 6500))]
    (directory / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    preserved, native = {}, {}
    stopped = directory / "stopped-checkpoints"; stopped.mkdir()
    for name, row in (("minimum_validation_ce", metrics[0]), ("maximum_validation_accuracy", metrics[1]), ("latest", metrics[2])):
        values = {"fixture.weight": torch.full((2,), float(row["updates"]))}
        cursor = {"updates": row["updates"], "epoch": row["epoch"]}
        ce_best = {"updates": 100, "epoch": 0, "cross_entropy": 4., "top1_accuracy": 6000 / 21874}
        acc_row = metrics[0] if row["updates"] == 100 else metrics[1]
        acc_best = {"updates": acc_row["updates"], "epoch": acc_row["epoch"], "cross_entropy": acc_row["validation"]["cross_entropy"], "top1_accuracy": acc_row["validation"]["top1_accuracy"]}
        saved = {"format_version": 1, "manifest": recorded, "cursor": cursor, "parameters": values,
                 "parameter_audit": trainer.parameter_audit(values, recorded["initial_parameter_audit"]),
                 "frozen_buffers_sha256": graph, "model_audit": audit, "best": ce_best, "best_accuracy": acc_best}
        filename = h.d.SELECTORS.get(name, "latest.pt")
        path = directory / filename
        torch.save(saved, path)
        (stopped / filename).write_bytes(path.read_bytes())
        preserved[name] = {**entry(stopped / filename), "cursor": cursor, "frozen_buffers_sha256": graph, "frozen_buffers_preserved": True}
        if name != "latest":
            preserved[name]["validation"] = row["validation"]
            native[name] = {**preserved[name], "path": str(path)}
    write(directory / "selected-checkpoints.json", {"format_version": 1, "selectors": native})
    cessation = {"confirmed": True, "processes": [{"pid": 123456, "alive": False, "birth": "prior birth"}]}
    write(directory / "stop-receipt.json", {"process_cessation": cessation})
    receipt = {"format_version": 1, "status": "accepted_early_stop", "arm": h.E[0],
               "accepted_by": "agent_under_standing_user_authorization", "full_schedule_completed": False,
               "test_evaluated": False, "durable_updates": 300, "observed_updates": 307, "completed_epochs": 1,
               "process_cessation": cessation, "stop_receipt": entry(directory / "stop-receipt.json"),
               "artifact_sha256": {str(directory / name): h.c.file_hash(directory / name) for name in ("manifest.json", "launch.json", "metrics.jsonl", "selected-checkpoints.json")},
               "selected_checkpoints": {"format_version": 1, "selectors": {name: preserved[name] for name in h.d.SELECTORS}},
               "latest_checkpoint": preserved["latest"]}
    receipt_path = directory / "accepted-early-stop.json"; write(receipt_path, receipt)
    monkeypatch.setattr(h.d, "verify_cessation", lambda receipt: None)
    return parent_dir, parent, receipt_path, receipt


def test_accepted_stop_binds_both_winners_latest_and_actual_checkpoint_contents(tmp_path, monkeypatch):
    directory, parent, receipt_path, _ = accepted_fixture(tmp_path, monkeypatch)
    summary, bindings = h.accepted_e(parent, directory, receipt_path)
    assert summary["status"] == "accepted_early_stop" and summary["full_schedule_completed"] is False
    assert summary["updates"] == 300 and summary["observed_updates"] == 307
    assert set(summary["selected_checkpoints"]) == set(h.d.SELECTORS)
    assert all(value["parameters_verified"] for value in summary["checkpoint_content_audits"].values())
    assert str(receipt_path) in bindings
    assert h.c.read_json(directory / "arms" / h.E[0] / "status.json")["status"] == "running"


@pytest.mark.parametrize("damage", ["missing_acceptance", "missing_selector", "latest_cursor", "winner_epoch", "artifact", "graph", "omitted_worker", "test", "completed"])
def test_incoherent_or_unaccepted_stop_is_rejected(tmp_path, monkeypatch, damage):
    directory, parent, receipt_path, receipt = accepted_fixture(tmp_path, monkeypatch)
    arm = directory / "arms" / h.E[0]
    if damage == "missing_acceptance": receipt["accepted_by"] = "raw_failure_only"
    if damage == "missing_selector": receipt["selected_checkpoints"]["selectors"].pop("maximum_validation_accuracy")
    if damage == "latest_cursor": receipt["latest_checkpoint"]["cursor"]["updates"] = 299
    if damage == "winner_epoch": receipt["selected_checkpoints"]["selectors"]["minimum_validation_ce"]["cursor"]["epoch"] = 1
    if damage == "artifact": (arm / "metrics.jsonl").write_text("changed")
    if damage == "graph": receipt["latest_checkpoint"]["frozen_buffers_sha256"] = {"other": "wrong"}
    if damage == "omitted_worker":
        receipt["process_cessation"]["processes"][0]["pid"] = 777777
        write(arm / "stop-receipt.json", {"process_cessation": receipt["process_cessation"]})
        receipt["stop_receipt"] = entry(arm / "stop-receipt.json")
    if damage == "test": receipt["test_evaluated"] = True
    if damage == "completed": write(arm / "results.json", {"status": "completed"})
    write(receipt_path, receipt)
    with pytest.raises(h.c.CampaignError):
        h.accepted_e(parent, directory, receipt_path)


def test_checkpoint_parameter_corruption_rejected_even_with_updated_file_receipt(tmp_path, monkeypatch):
    directory, parent, receipt_path, receipt = accepted_fixture(tmp_path, monkeypatch)
    selected = receipt["latest_checkpoint"]
    path = Path(selected["path"])
    saved = torch.load(path, weights_only=False)
    saved["parameters"]["fixture.weight"][0] = -1
    torch.save(saved, path)
    selected["sha256"] = h.c.file_hash(path)
    (directory / "arms" / h.E[0] / "latest.pt").write_bytes(path.read_bytes())
    write(receipt_path, receipt)
    with pytest.raises(h.c.CampaignError, match="parameter hashes"):
        h.accepted_e(parent, directory, receipt_path)


def prepare_fixture(tmp_path, monkeypatch):
    parent_dir, parent, receipt_path, _ = accepted_fixture(tmp_path, monkeypatch)
    parity = {"passed": True, "marker": "already checked parent parity"}
    parity_path = tmp_path / "original-parity.json"; write(parity_path, parity)
    parent["preflight_receipt"] = entry(parity_path)
    parent["input_files_sha256"][str(parity_path)] = h.c.file_hash(parity_path)
    write(parent_dir / "preflight/parity.json", parity)
    write(parent_dir / "manifest.json", parent)
    write(parent_dir / "launch.json", {"launch_id": parent["launch_id"], "manifest_sha256": h.c.file_hash(parent_dir / "manifest.json")})
    smokes = []
    for arm in h.d.ARMS:
        smoke_dir = parent_dir / "smokes" / arm[0]
        smoke_dir.mkdir(parents=True)
        for name in ("manifest.json", "launch.json", "status.json", "process-status.json", "metrics.jsonl", "selected-checkpoints.json", "initial-gradients.json", "best.pt", "best-accuracy.pt", "latest.pt"):
            (smoke_dir / name).write_text("existing smoke artifact")
        smokes.append({"name": arm[0], "passed": True, "updates": 8})
    write(parent_dir / "smoke-results.json", {"passed": True, "arms": smokes})
    monkeypatch.setattr(h.d, "host_guard", lambda manifest=None: {})
    monkeypatch.setattr(h.d, "verify_preflight", lambda receipt, manifest: None)
    monkeypatch.setattr(h.d, "verify_arm", lambda manifest, arm, directory, smoke=False: {"name": arm[0], "passed": True, "updates": 8} if smoke else {"name": arm[0], "status": "completed", "full_schedule_completed": True, "test_evaluated": False})
    return SimpleNamespace(parent=parent_dir, accepted_arm_receipt=receipt_path, output=tmp_path / "continuation"), parent


def test_prepare_reuses_both_smokes_preserves_recipe_and_blocks_parent_F_duplicate(tmp_path, monkeypatch):
    args, parent = prepare_fixture(tmp_path, monkeypatch)
    manifest = h.prepare(args)
    assert manifest["continuation_arms"] == [h.F[0]]
    assert manifest["experiment_branches"] == parent["experiment_branches"]
    assert manifest["planned_phase_updates"] == h.d.PLANNED_UPDATES
    assert manifest["parent_campaign"]["path"] == str(args.parent / "manifest.json")
    assert str(h.SELF) in manifest["sources_sha256"]
    assert all(str(args.parent / "smokes" / arm[0] / "latest.pt") in manifest["input_files_sha256"] for arm in h.d.ARMS)
    write(args.parent / "arms" / h.F[0] / "launch.json", {"already": "started"})
    with pytest.raises(h.c.CampaignError, match="Parent F already started"):
        h.prepare(args)


@pytest.mark.parametrize("fail", [False, True])
def test_worker_dispatches_only_F_without_repeating_parity_or_smokes(tmp_path, monkeypatch, fail):
    args, _ = prepare_fixture(tmp_path, monkeypatch)
    manifest = h.prepare(args)
    args.launch_id = manifest["launch_id"]
    write(args.output / "manifest.json", manifest)
    write(args.output / "launch.json", {"manifest_sha256": h.c.file_hash(args.output / "manifest.json")})
    write(args.output / "campaign-status.json", {"status": "starting"})
    monkeypatch.setattr(h.c, "start_awake", lambda: None)
    calls = []
    def execute(manifest, phase, name, directory, command):
        calls.append((phase, name, command))
        h.c.campaign_status(args.output, status="running", phase=phase, arm=name)
        if fail: raise h.c.CampaignError("injected failure")
    monkeypatch.setattr(h.c, "execute_job", execute)
    result = h.worker(args)
    assert [(phase, name) for phase, name, _ in calls] == [("training", h.F[0])]
    command = calls[0][2]
    assert "--max-updates" not in command and "--resume" not in command and "--skip-final-test" in command
    assert command[command.index("--readout-rank") + 1] == "128"
    if fail:
        assert result["status"] == "failed" and h.launch(args)["status"] == "blocked"
        assert not (args.output / "results.json").exists()
    else:
        assert result["status"] == "completed" and result["full_schedule_completed_arms"] == [h.F[0]]
        assert result["accepted_early_stopped_arms"] == [h.E[0]] and result["unequal_training_budgets"] is True


def test_changed_bound_parent_source_or_input_blocks_handoff(tmp_path, monkeypatch):
    args, parent = prepare_fixture(tmp_path, monkeypatch)
    Path(parent["preflight_receipt"]["path"]).write_text("changed")
    with pytest.raises(h.c.CampaignError, match="bound campaign file changed"):
        h.prepare(args)
