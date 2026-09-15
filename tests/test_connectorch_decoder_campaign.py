"""Decoder campaign gates and queue order without GPU work or training."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decoder_campaign", ROOT / "scripts/run_connectorch_decoder_campaign.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def manifest_fixture(tmp_path):
    source = tmp_path / "immutable-source.py"
    source.write_text("bound fixture")
    return {"root": str(tmp_path), "output": str(tmp_path / "campaign"), "python": "python-fixture",
            "launch_id": "test-launch", "model": "/pinned-model", "data": "/data.json", "groups": "/groups.npz",
            "data_sha256": "data", "groups_sha256": "groups", "planned_phase_updates": d.PLANNED_UPDATES,
            "baseline_frozen_buffers_sha256": {"brain.w_values": "graph"},
            "baseline_connectorch_source": {"revision": "pinned"}, "preflight_connectorch_source": {"revision": "pinned"},
            "sources_sha256": {str(tmp_path / name): "test-source" for name in d.c.TRAINER_SOURCE_FILES},
            "input_files_sha256": {str(source): d.c.file_hash(source)}, "baseline_files_sha256": {},
            "validation_stories": 100, "validation_tokens": 21874,
            "smoke_validation_stories": 8, "smoke_validation_tokens": 1600,
            "baseline_summary": {"name": "B32fixed", "status": "accepted_early_stop"},
            "baseline_acceptance": {"path": str(tmp_path / "accepted.json")}}


def parity_fixture(manifest):
    cases, comparisons, arms = {}, {}, {}
    for arm in d.ARMS:
        name = arm[0]
        arms[name] = {**d.arm_config(arm), "parameter_counts": d.expected_counts(arm)}
        for device in ("cpu", "mps"):
            cases[name + "_" + device] = {"passed": True, "device": device, "targets": 64,
                "parameter_count": d.expected_counts(arm)["total"], "frozen_buffers_unchanged": True,
                "frozen_buffers_sha256": manifest["baseline_frozen_buffers_sha256"], "padding_gradient_zero": True,
                "native_kernel_gate": True, "parameters_and_gradients_finite": {"lm_head.0.weight": True, "lm_head.1.weight": True},
                "factorized_gain_gradients": {key: {"nonzero": 5} for key in ("brain.edge_theta_source", "brain.edge_theta_destination")}}
        comparisons[name + "_cpu_vs_mps"] = {"passed": True, "loss_abs_difference": 1e-7, "max_gradient_relative_l2": 1e-6, "cache_ids_and_length_equal": True}
    return {"passed": True, "sources_unchanged": True, "data_file_sha256": "data", "groups_file_sha256": "groups",
            "inputs_unchanged": True, "connectorch_unchanged": True, "exact_ir": {"passed": True}, "final_host_guard": {"passed": True},
            "arms": arms, "cases": cases, "comparisons": comparisons,
            "source_sha256": {next(iter(d.c.TRAINER_SOURCE_FILES)): "test-source"}, "connectorch": {"revision": "pinned"},
            "initialization_equivalence": {"passed": True, "shared_parameter_hashes_equal": True, "zero_adaptive_parameters": True, "logits_equal": True}}


def arm_fixture(manifest, arm, directory, smoke):
    counts = d.expected_counts(arm)
    recorded = {"config": d.arm_config(arm, smoke), "parameter_counts": counts,
                "data_sha256": "data", "groups_sha256": "groups", "planned_phase_updates": d.PLANNED_UPDATES,
                "trainer_sources_sha256": {name: "test-source" for name in d.c.TRAINER_SOURCE_FILES},
                "connectorch_source": {"revision": "pinned"}}
    graph = {"frozen_buffers_preserved": True, "frozen_buffers_sha256": manifest["baseline_frozen_buffers_sha256"],
             "adaptation": {"zero_edges_preserved": True, "nonzero_signs_preserved": True,
                            "source_type_theta": {"rms": .01}, "destination_type_theta": {"rms": .02}, "edge_displacement": {"rms": .001}}}
    tokens = manifest["smoke_validation_tokens" if smoke else "validation_tokens"]
    metric = {"cross_entropy": 4., "tokens": tokens, "correct": 400, "top1_accuracy": 400 / tokens,
              "stories": manifest["smoke_validation_stories" if smoke else "validation_stories"]}
    steps = 8 if smoke else sum(d.PLANNED_UPDATES)
    record = {"event": "validation", "updates": steps, "epoch": 0 if smoke else 44, "validation": metric, "model_audit": graph}
    directory.mkdir(parents=True, exist_ok=True)
    selectors = {}
    for selector, filename in d.SELECTORS.items():
        path = directory / filename
        path.write_text("retained fixture " + selector)
        selectors[selector] = {"path": str(path), "sha256": d.c.file_hash(path), "cursor": {"updates": steps},
                               "validation": metric, "frozen_buffers_preserved": True, "frozen_buffers_sha256": manifest["baseline_frozen_buffers_sha256"]}
    write(directory / "selected-checkpoints.json", {"format_version": 1, "selectors": selectors})
    write(directory / "manifest.json", recorded)
    write(directory / "process-status.json", {"status": "completed", "exit_code": 0})
    (directory / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    if smoke:
        write(directory / "status.json", {"status": "debug_stopped", "debug": True, "updates": 8, "test_evaluated": False})
    else:
        selected = {"updates": steps, "cross_entropy": metric["cross_entropy"], "top1_accuracy": metric["top1_accuracy"]}
        result = {"status": "completed", "debug": False, "epochs": 44, "updates": steps,
                  "test_evaluated": False, "test_deferred": True, "checks": {**graph, "selected_checkpoint_audit": graph},
                  "selected": selected, "selected_accuracy": selected, "selected_checkpoints": selectors}
        write(directory / "status.json", result)
        write(directory / "results.json", result)
    return recorded


def test_commands_and_counts_are_explicitly_rank128_and_full_recipe(tmp_path):
    manifest = manifest_fixture(tmp_path)
    for arm in d.ARMS:
        for smoke in (False, True):
            command = d.trainer_command(manifest, arm, tmp_path / arm[0], smoke)
            assert command[command.index("--readout-rank") + 1] == "128"
            assert command[command.index("--d-embed") + 1] == "32"
            assert command[command.index("--history-length") + 1] == "8"
            assert "--skip-final-test" in command and "--resume" not in command
            assert ("--max-updates" in command) == smoke
        assert d.expected_counts(arm)["readout"] == 6453376
    assert [d.expected_counts(arm)["total"] for arm in d.ARMS] == [7183157, 7201479]
    assert sum(d.PLANNED_UPDATES) == 52609
    with pytest.raises(d.c.CampaignError, match="Unknown decoder"):
        d.arm_config(("B32fixed", 32, "fixed"))


@pytest.mark.parametrize("damage", ["rank0", "missing_arm", "head_gradient", "gain_gradient", "wrong_graph", "init", "source"])
def test_preflight_rejects_wrong_architecture_and_missing_wiring(tmp_path, damage):
    manifest = manifest_fixture(tmp_path)
    parity = parity_fixture(manifest)
    d.verify_preflight(parity, manifest)
    if damage == "rank0": parity["arms"]["E32rank128fixed"]["readout_rank"] = 0
    if damage == "missing_arm": parity["arms"].pop("F32rank128bounded")
    if damage == "head_gradient": parity["cases"]["E32rank128fixed_mps"]["parameters_and_gradients_finite"].pop("lm_head.1.weight")
    if damage == "gain_gradient": parity["cases"]["F32rank128bounded_mps"]["factorized_gain_gradients"]["brain.edge_theta_destination"]["nonzero"] = 0
    if damage == "wrong_graph": parity["cases"]["E32rank128fixed_mps"]["frozen_buffers_sha256"] = {}
    if damage == "init": parity["initialization_equivalence"]["shared_parameter_hashes_equal"] = False
    if damage == "source": parity["source_sha256"] = {"unknown.py": "unbound"}
    with pytest.raises(d.c.CampaignError): d.verify_preflight(parity, manifest)


def test_process_guard_rejects_live_or_suspended_trainers_and_dispatchers():
    for state in ("S", "Ts"):
        for script in ("train_connectorch.py", "continue_connectorch_campaign.py", "run_connectorch_decoder_campaign.py", "audit_connectorch_decoder.py"):
            records = d.process_inventory(f"42 1 {state} /Framework/Python scripts/{script}")
            with pytest.raises(d.c.CampaignError, match="Competing"):
                d.reject_competing_jobs(records)
    assert d.reject_competing_jobs(d.process_inventory("42 1 Z <defunct>"))["no_competing_jobs"]
    assert d.reject_competing_jobs(d.process_inventory("42 1 S Python scripts/run_connectorch_decoder_campaign.py"), {42})["no_competing_jobs"]


def test_smoke_verifies_actual_head_and_gain_changes_without_training(tmp_path):
    spec = importlib.util.spec_from_file_location("fixture_training_audit", ROOT / "scripts/train_connectorch.py")
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    names = ("lm_head.0.weight", "lm_head.1.weight", "brain.edge_theta_source", "brain.edge_theta_destination")
    initial = {name: torch.zeros(2) for name in names}
    values = {name: torch.ones(2) for name in names}
    recorded = {"initial_parameter_audit": trainer.parameter_audit(initial)}
    gradients = {name: {"present": True, "finite": True, "nonzero_values": 2} for name in names}
    write(tmp_path / "initial-gradients.json", gradients)
    saved = {"manifest": recorded, "cursor": {"updates": 8}, "parameters": values,
             "parameter_audit": trainer.parameter_audit(values, recorded["initial_parameter_audit"])}
    torch.save(saved, tmp_path / "latest.pt")
    assert set(d.verify_smoke_parameter_updates(tmp_path, recorded, "bounded10")) == set(names)
    values["lm_head.1.weight"].zero_()
    saved["parameter_audit"] = trainer.parameter_audit(values, recorded["initial_parameter_audit"])
    torch.save(saved, tmp_path / "latest.pt")
    with pytest.raises(d.c.CampaignError, match="did not actually update"):
        d.verify_smoke_parameter_updates(tmp_path, recorded, "bounded10")


def worker_fixture(tmp_path, monkeypatch):
    manifest = manifest_fixture(tmp_path)
    # Worker fixtures bind only their synthetic files; source-content rejection
    # is tested independently from the trainer receipt's relative-name contract.
    for path in manifest["sources_sha256"]:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text("test-source")
        manifest["sources_sha256"][path] = d.c.file_hash(target)
    parity = parity_fixture(manifest)
    parity["source_sha256"] = {name: manifest["sources_sha256"][str(tmp_path / name)] for name in d.c.TRAINER_SOURCE_FILES}
    parity_path = tmp_path / "parity.json"; write(parity_path, parity)
    manifest["preflight_receipt"] = {"path": str(parity_path), "sha256": d.c.file_hash(parity_path)}
    output = Path(manifest["output"])
    write(output / "manifest.json", manifest)
    write(output / "launch.json", {"manifest_sha256": d.c.file_hash(output / "manifest.json")})
    write(output / "campaign-status.json", {"status": "starting"})
    write(Path(manifest["baseline_acceptance"]["path"]), {})
    monkeypatch.setattr(d, "host_guard", lambda manifest=None: {})
    monkeypatch.setattr(d, "verify_cessation", lambda receipt: None)
    monkeypatch.setattr(d.c, "start_awake", lambda: None)
    monkeypatch.setattr(d, "verify_smoke_parameter_updates", lambda *args: {"fixture": True})
    return SimpleNamespace(output=output, launch_id=manifest["launch_id"]), manifest


@pytest.mark.parametrize("fail_arm", [None, "F32rank128bounded"])
def test_serial_queue_requires_both_smokes_before_any_full_training(tmp_path, monkeypatch, fail_arm):
    args, manifest = worker_fixture(tmp_path, monkeypatch)
    calls = []
    def execute(current, phase, name, directory, command):
        calls.append((phase, name))
        d.c.campaign_status(args.output, status="running", phase=phase, arm=name)
        if phase == "smoke" and name == fail_arm:
            raise d.c.CampaignError("injected smoke failure")
        arm = next(arm for arm in d.ARMS if arm[0] == name)
        recorded = arm_fixture(current, arm, directory, phase == "smoke")
        recorded["trainer_sources_sha256"] = {relative: current["sources_sha256"][str(Path(current["root"]) / relative)] for relative in d.c.TRAINER_SOURCE_FILES}
        write(directory / "manifest.json", recorded)
    monkeypatch.setattr(d.c, "execute_job", execute)
    result = d.worker(args)
    expected = [(phase, arm[0]) for phase in ("smoke", "training") for arm in d.ARMS]
    if fail_arm:
        assert result["status"] == "failed" and calls == expected[:2]
        assert not (args.output / "results.json").exists()
    else:
        assert result["status"] == "completed" and calls == expected
        assert result["test_evaluated"] is False


def test_rank0_configuration_and_reserved_test_access_are_rejected(tmp_path):
    manifest = manifest_fixture(tmp_path)
    directory = tmp_path / "arm"
    recorded = arm_fixture(manifest, d.ARMS[0], directory, False)
    d.verify_arm(manifest, d.ARMS[0], directory)
    recorded["config"]["readout_rank"] = 0
    write(directory / "manifest.json", recorded)
    with pytest.raises(d.c.CampaignError, match="configuration"):
        d.verify_arm(manifest, d.ARMS[0], directory)
    arm_fixture(manifest, d.ARMS[0], directory, False)
    result = d.c.read_json(directory / "results.json"); result["test_evaluated"] = True
    write(directory / "results.json", result)
    with pytest.raises(d.c.CampaignError, match="reserved test"):
        d.verify_arm(manifest, d.ARMS[0], directory)


def test_existing_failed_campaign_never_restarts(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path)
    write(tmp_path / "manifest.json", {"launch_id": "old"})
    write(tmp_path / "campaign-status.json", {"status": "failed"})
    write(tmp_path / "launch.json", {})
    monkeypatch.setattr(d, "prepare", lambda args: pytest.fail("Must not prepare an automatic retry"))
    assert d.launch(args)["status"] == "blocked"


def test_jsonl_uses_physical_lines_not_unicode_separators(tmp_path):
    row = {"event": "validation", "text": "text\u2028still in one JSON value", "updates": 8}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert d.validation_records(tmp_path) == [row]
