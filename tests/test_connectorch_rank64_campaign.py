"""Decoder campaign gates and queue order without GPU work or training."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rank64_campaign", ROOT / "scripts/run_connectorch_rank64_campaign.py")
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
    name = d.ARMS[0][0]
    source = "scripts/audit_connectorch_rank64.py"
    manifest["sources_sha256"][str(Path(manifest["root"]) / source)] = "test-source"
    case = {"passed": True, "targets": 64, "parameter_count": 3956469,
        "frozen_buffers_unchanged": True, "frozen_buffers_sha256": manifest["baseline_frozen_buffers_sha256"],
        "padding_gradient_zero": True, "native_kernel_gate": True,
        "parameters_and_gradients_finite": {"lm_head.0.weight": True, "lm_head.1.weight": True},
        "all_required_gradients_nonzero": True,
        "required_gradients": {key: {"finite": True, "nonzero_values": 5} for key in ("lm_head.0.weight", "lm_head.1.weight")}}
    return {"passed": True, "sources_unchanged": True, "inputs_unchanged": True, "connectorch_unchanged": True,
        "training_run": False, "optimizer_steps": 0, "reserved_test_evaluated": False,
        "data_file_sha256": "data", "groups_file_sha256": "groups", "exact_ir": {"passed": True}, "final_host_guard": {"passed": True},
        "arms": {name: {**d.arm_config(d.ARMS[0]), "parameter_counts": d.expected_counts(d.ARMS[0])}},
        "cases": {name + "_" + device: {**copy.deepcopy(case), "device": device} for device in ("cpu", "mps")},
        "comparisons": {name + "_cpu_vs_mps": {"passed": True, "loss_abs_difference": 1e-7, "max_gradient_relative_l2": 1e-6, "cache_ids_and_length_equal": True}},
        "source_sha256": {source: "test-source"}, "connectorch": {"revision": "pinned"},
        "initialization": {"passed": True, "cpu_mps_parameter_hashes_equal": True, "parameter_hashes": {"fixture": "hash"},
            "readout_shapes": {"lm_head.0.weight": [64, 49393], "lm_head.1.weight": [1024, 64]}}}


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


def test_single_g_exact_rank64_counts_and_recipe(tmp_path):
    assert d.ARMS == (("G32rank64fixed", 32, "fixed"),)
    manifest = manifest_fixture(tmp_path)
    counts = d.expected_counts(d.ARMS[0])
    assert counts["total"] == 3956469 and counts["readout"] == 3226688 and counts["edge_gains"] == 0
    for smoke in (False, True):
        command = d.trainer_command(manifest, d.ARMS[0], tmp_path / "G", smoke)
        assert command[command.index("--readout-rank") + 1] == "64"
        assert command[command.index("--d-embed") + 1] == "32"
        assert command[command.index("--history-length") + 1] == "8"
        assert command[command.index("--plasticity") + 1] == "fixed"
        assert "--skip-final-test" in command and "--resume" not in command
        assert ("--max-updates" in command) == smoke
    assert sum(d.PLANNED_UPDATES) == 52609
    with pytest.raises(d.c.CampaignError):
        d.arm_config(("F32rank128bounded", 32, "bounded10"))


@pytest.mark.parametrize("damage", ["rank128", "missing_case", "gradient", "zero_gradient", "graph", "init", "source", "test"])
def test_parity_rejects_wrong_architecture_and_missing_evidence(tmp_path, damage):
    manifest = manifest_fixture(tmp_path)
    parity = parity_fixture(manifest)
    d.verify_preflight(parity, manifest)
    name = d.ARMS[0][0]
    if damage == "rank128": parity["arms"][name]["readout_rank"] = 128
    if damage == "missing_case": parity["cases"].pop(name + "_mps")
    if damage == "gradient": parity["cases"][name + "_mps"]["parameters_and_gradients_finite"].pop("lm_head.1.weight")
    if damage == "zero_gradient": parity["cases"][name + "_mps"]["required_gradients"]["lm_head.0.weight"]["nonzero_values"] = 0
    if damage == "graph": parity["cases"][name + "_mps"]["frozen_buffers_sha256"] = {}
    if damage == "init": parity["initialization"]["readout_shapes"]["lm_head.0.weight"][0] = 128
    if damage == "source": parity["source_sha256"] = {"unknown.py": "unbound"}
    if damage == "test": parity["reserved_test_evaluated"] = True
    with pytest.raises(d.c.CampaignError):
        d.verify_preflight(parity, manifest)


def worker_fixture(tmp_path, monkeypatch):
    manifest = manifest_fixture(tmp_path)
    parity = parity_fixture(manifest)
    for path in manifest["sources_sha256"]:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text("test-source")
        manifest["sources_sha256"][path] = d.c.file_hash(target)
    parity["source_sha256"] = {"scripts/audit_connectorch_rank64.py": manifest["sources_sha256"][str(tmp_path / "scripts/audit_connectorch_rank64.py")]}
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


@pytest.mark.parametrize("fail_smoke", [False, True])
def test_worker_runs_only_g_smoke_then_g_full(tmp_path, monkeypatch, fail_smoke):
    args, manifest = worker_fixture(tmp_path, monkeypatch)
    calls = []
    def execute(current, phase, name, directory, command):
        calls.append((phase, name))
        d.c.campaign_status(args.output, status="running", phase=phase, arm=name)
        if phase == "smoke" and fail_smoke:
            raise d.c.CampaignError("injected smoke failure")
        recorded = arm_fixture(current, d.ARMS[0], directory, phase == "smoke")
        recorded["trainer_sources_sha256"] = {relative: current["sources_sha256"][str(Path(current["root"]) / relative)] for relative in d.c.TRAINER_SOURCE_FILES}
        write(directory / "manifest.json", recorded)
    monkeypatch.setattr(d.c, "execute_job", execute)
    result = d.worker(args)
    assert calls == ([("smoke", "G32rank64fixed")] if fail_smoke else [("smoke", "G32rank64fixed"), ("training", "G32rank64fixed")])
    assert result["status"] == ("failed" if fail_smoke else "completed")
    if not fail_smoke:
        assert result["test_evaluated"] is False and len(result["arms"]) == 1


def test_full_arm_checks_dual_selectors_rank_and_reserved_test(tmp_path):
    manifest = manifest_fixture(tmp_path)
    directory = tmp_path / "arm"
    arm_fixture(manifest, d.ARMS[0], directory, False)
    result = d.verify_arm(manifest, d.ARMS[0], directory)
    assert set(result["selected_checkpoints"]) == set(d.SELECTORS)
    assert result["readout_rank"] == 64
    recorded = d.c.read_json(directory / "manifest.json"); recorded["config"]["readout_rank"] = 128
    write(directory / "manifest.json", recorded)
    with pytest.raises(d.c.CampaignError, match="configuration"):
        d.verify_arm(manifest, d.ARMS[0], directory)
    arm_fixture(manifest, d.ARMS[0], directory, False)
    result = d.c.read_json(directory / "results.json"); result["test_evaluated"] = True
    write(directory / "results.json", result)
    with pytest.raises(d.c.CampaignError, match="reserved test"):
        d.verify_arm(manifest, d.ARMS[0], directory)


def test_failed_campaign_never_automatically_restarts(tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", {})
    write(tmp_path / "campaign-status.json", {"status": "failed"})
    write(tmp_path / "launch.json", {})
    monkeypatch.setattr(d, "prepare", lambda args: pytest.fail("must not prepare an existing campaign"))
    assert d.launch(SimpleNamespace(output=tmp_path))["status"] == "blocked"
