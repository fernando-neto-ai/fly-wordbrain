"""Early-stop continuation lifecycle checks, using tiny files and fake workers."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load("continuation", ROOT / "scripts/continue_connectorch_campaign.py")
fixtures = load("campaign_fixtures", ROOT / "tests/test_connectorch_campaign.py")
c = m.c


def fixture(tmp_path, monkeypatch):
    args = fixtures.completed_baseline(tmp_path)
    parent = args.output
    parent.mkdir(parents=True)
    manifest = {"root": str(args.root), "output": str(parent), "python": args.python,
                "launch_id": "parent-id", **c.bind_inputs(args, c.baseline_readiness(args.baseline_run))}
    parity = fixtures.parity_receipt(manifest)
    manifest["preflight_connectorch_source"] = parity["connectorch"]
    fixtures.write(parent / "preflight/parity.json", parity)
    manifest["preflight_receipt_sha256"] = c.file_hash(parent / "preflight/parity.json")
    fixtures.write(parent / "manifest.json", manifest)
    fixtures.write(parent / "launch.json", {"worker_pid": 34567})
    smokes = []
    for arm in c.ARMS:
        directory = parent / "smokes" / arm[0]
        fixtures.arm_artifacts(manifest, arm, directory, True)
        smokes.append(c.verify_arm(manifest, arm, directory, True))
    fixtures.write(parent / "smoke-results.json", {"passed": True, "arms": smokes})
    directory = parent / "arms/A128fixed"
    fixtures.arm_artifacts(manifest, c.ARMS[0], directory, False)
    (directory / "results.json").unlink()
    records = c.validation_records(directory)
    records[0]["updates"] = 20
    records[0]["epoch"] = 10
    (directory / "metrics.jsonl").write_text(json.dumps(records[0]) + "\n")
    selected = c.read_json(directory / "selected-checkpoints.json")
    for entry in selected["selectors"].values():
        entry["cursor"] = {"updates": 20, "epoch": 10}
    fixtures.write(directory / "selected-checkpoints.json", selected)
    fixtures.write(directory / "status.json", {"status": "running", "updates": 21, "pid": 34568})
    fixtures.write(directory / "process-status.json", {"status": "failed", "exit_code": -15,
                   "worker_pid": 34568, "runner_pid": 34567})
    (directory / "latest.pt").write_bytes(b"preserved latest fixture")
    def reference(path):
        return {"path": str(path), "sha256": c.file_hash(path)}
    cessation = {"confirmed": True, "processes": [{"pid": p, "birth": "old-birth", "alive": False} for p in (34567, 34568)]}
    fixtures.write(directory / "stop-receipt.json", {"process_cessation": cessation})
    receipt = {"format_version": 1, "status": "accepted_early_stop", "accepted_by": "user",
               "instruction": "If plateaued, move to the next experiment", "arm": "A128fixed",
               "campaign_manifest": reference(parent / "manifest.json"),
               "arm_manifest": reference(directory / "manifest.json"), "metrics": reference(directory / "metrics.jsonl"),
               "selected_checkpoint_receipt": reference(directory / "selected-checkpoints.json"),
               "stop_receipt": reference(directory / "stop-receipt.json"), "process_cessation": cessation,
               "preserved_checkpoints": {**selected["selectors"], "latest": {**reference(directory / "latest.pt"),
                                         "cursor": {"updates": 20, "epoch": 10}}},
               "observed_updates": 21, "completed_epochs": 10, "test_evaluated": False}
    receipt_path = directory / "accepted-early-stop.json"
    fixtures.write(receipt_path, receipt)
    companion = args.root / "scripts/continue_connectorch_campaign.py"
    companion.write_text(m.SELF.read_text())
    monkeypatch.setattr(m, "SELF", companion)
    monkeypatch.setattr(c, "process_alive", lambda *args: False)
    args = SimpleNamespace(parent=parent, output=parent.with_name("continuation"), accepted_arm_receipt=receipt_path, launch_id=None)
    return args, manifest, receipt


def test_prepares_continuation_without_changing_parent_or_training_sources(tmp_path, monkeypatch):
    args, parent, _ = fixture(tmp_path, monkeypatch)
    original = (args.parent / "manifest.json").read_bytes()
    manifest = m.prepare(args)
    assert manifest["continuation_arms"] == ["B32fixed", "C128bounded", "D32bounded"]
    assert manifest["accepted_early_stopped_arm"]["updates"] == 20
    assert manifest["accepted_early_stopped_arm"]["full_schedule_completed"] is False
    assert manifest["sources_sha256"].items() >= parent["sources_sha256"].items()
    assert (args.parent / "manifest.json").read_bytes() == original
    c.verify_bindings(manifest)


@pytest.mark.parametrize("mutation", ["live", "missing_pid", "checkpoint_changed", "no_acceptance", "fake_counts",
                                      "epoch_mismatch", "test_access", "invalid_metric"])
def test_rejects_unsafe_or_unbound_early_stop(tmp_path, monkeypatch, mutation):
    args, parent, receipt = fixture(tmp_path, monkeypatch)
    if mutation == "live":
        monkeypatch.setattr(c, "process_alive", lambda *args: True)
    elif mutation == "missing_pid":
        receipt["process_cessation"]["processes"].pop()
        fixtures.write(Path(receipt["stop_receipt"]["path"]), {"process_cessation": receipt["process_cessation"]})
        receipt["stop_receipt"]["sha256"] = c.file_hash(receipt["stop_receipt"]["path"])
    elif mutation == "checkpoint_changed":
        Path(receipt["preserved_checkpoints"]["latest"]["path"]).write_text("changed")
    elif mutation == "no_acceptance":
        receipt["accepted_by"] = "assistant"
    elif mutation == "epoch_mismatch":
        receipt["completed_epochs"] = 9
    elif mutation == "test_access":
        receipt["test_evaluated"] = True
    elif mutation == "invalid_metric":
        path = Path(receipt["metrics"]["path"])
        rows = c.validation_records(path.parent)
        rows[0]["validation"]["cross_entropy"] = -1
        path.write_text(json.dumps(rows[0]) + "\n")
        receipt["metrics"]["sha256"] = c.file_hash(path)
    else:
        receipt["completed_epochs"] = 44
    fixtures.write(args.accepted_arm_receipt, receipt)
    with pytest.raises(c.CampaignError):
        m.prepare(args)


def test_dispatch_idempotent_and_dead_worker_not_restarted(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: calls.append((a, k)) or SimpleNamespace(pid=76543))
    monkeypatch.setattr(c, "process_birth", lambda p: "new-birth")
    assert m.launch(args)["status"] == "launched"
    monkeypatch.setattr(c, "process_alive", lambda p, birth=None: p == 76543 and birth == "new-birth")
    assert m.launch(args)["status"] == "already_running"
    monkeypatch.setattr(c, "process_alive", lambda *args: False)
    assert m.launch(args)["status"] == "blocked"
    assert len(calls) == 1
    assert calls[0][1]["start_new_session"] is True


def test_worker_runs_only_b_c_d_and_reports_unequal_budgets(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    args.output.mkdir()
    manifest = m.prepare(args)
    fixtures.write(args.output / "manifest.json", manifest)
    fixtures.write(args.output / "campaign-status.json", {"status": "starting"})
    args.launch_id = manifest["launch_id"]
    monkeypatch.setattr(c, "start_awake", lambda: None)
    calls = []
    def execute(manifest, phase, name, directory, command):
        calls.append(name)
        arm = next(a for a in c.ARMS if a[0] == name)
        fixtures.arm_artifacts(manifest, arm, directory, False)
        records = c.validation_records(directory)
        common = {**records[0], "updates": 20, "validation": {**records[0]["validation"], "cross_entropy": 10, "top1_accuracy": .1}}
        (directory / "metrics.jsonl").write_text(json.dumps(common) + "\n" + json.dumps(records[0]) + "\n")
    monkeypatch.setattr(c, "execute_job", execute)
    result = m.worker(args)
    assert result["status"] == "completed", result
    assert calls == ["B32fixed", "C128bounded", "D32bounded"]
    assert result["full_schedule_completed_arms"] == calls
    assert result["accepted_early_stopped_arms"] == ["A128fixed"]
    assert result["comparison"]["unequal_training_budgets"] is True
    assert result["matched_update_comparison"]["updates"] == 20
    assert not (args.parent / "arms/A128fixed/results.json").exists()


def test_worker_stops_at_first_failure_without_launching_later_arms(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    args.output.mkdir()
    manifest = m.prepare(args)
    fixtures.write(args.output / "manifest.json", manifest)
    fixtures.write(args.output / "campaign-status.json", {"status": "starting"})
    args.launch_id = manifest["launch_id"]
    monkeypatch.setattr(c, "start_awake", lambda: None)
    calls = []
    def fail(manifest, phase, name, directory, command):
        calls.append(name)
        raise RuntimeError("injected failure")
    monkeypatch.setattr(c, "execute_job", fail)
    assert m.worker(args)["status"] == "failed"
    assert calls == ["B32fixed"]
    assert not (args.output / "results.json").exists()
