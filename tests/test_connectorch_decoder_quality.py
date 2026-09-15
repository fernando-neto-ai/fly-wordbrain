"""Rank-aware evaluator contracts; fake fixtures and small CPU tensors only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("decoder_quality", ROOT / "scripts/evaluate_connectorch_decoder_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_physical_jsonl_preserves_unicode_separators(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"text": "first\u2028second\u2029third"}, ensure_ascii=False) + "\n" + json.dumps({"end": True}) + "\n")
    assert quality.jsonl(path) == [{"text": "first\u2028second\u2029third"}, {"end": True}]


@pytest.mark.parametrize("script", sorted(quality.DISPATCHERS))
def test_all_dispatcher_generations_must_be_held(script):
    row = {"pid": 41, "ppid": 1, "state": "S", "command": f"python -u scripts/{script} --worker"}
    with pytest.raises(ValueError, match="Idle GPU"):
        quality.process_guard([row])
    assert quality.process_guard([{**row, "state": "Ts"}])["held_dispatchers"][0]["pid"] == 41


@pytest.mark.parametrize("script", ["train_connectorch.py", "evaluate_connectorch_quality.py", "evaluate_connectorch_decoder_quality.py", "audit_connectorch_rank64.py"])
@pytest.mark.parametrize("state", ["S", "Ts"])
def test_paused_or_running_gpu_workers_are_rejected(script, state):
    with pytest.raises(ValueError, match="Idle GPU"):
        quality.process_guard([{"pid": 42, "ppid": 1, "state": state, "command": f"python scripts/{script}"}])


def test_zombie_exited_own_pid_excluded_shell_parent_not_gpu_worker():
    rows = quality.parse_processes("42 1 Z <defunct>\n43 1 S python scripts/evaluate_connectorch_decoder_quality.py\n44 1 S sh -c python scripts/train_connectorch.py\n")
    result = quality.process_guard(rows, {42}, own_pid=43)
    assert [r["pid"] for r in result["exited_zombies"]] == [42]


def test_declared_configs_accepted_rank0_and_recipe_drift_rejected():
    arms = []
    for label, rank in quality.ARMS.items():
        manifest = {"config": {**quality.COMMON_CONFIG, "readout_rank": rank},
                    "debug": False, "parameter_counts": quality.expected_counts(rank),
                    "reference_repository": "ngxson/fly-llm-hf", "reference_revision": quality.trainer.REVISION}
        for key in ("data_sha256", "groups_sha256", "reference_files", "frozen_buffers_sha256", "trainer_sources_sha256",
                    "connectorch_source", "assumptions", "planned_phase_updates", "epoch_updates", "split_sizes", "data_provenance"):
            manifest[key] = {"same": "bound"}
        quality.verify_config(manifest, label)
        arms.append({"manifest": manifest})
        bad = copy.deepcopy(manifest); bad["config"]["readout_rank"] = 0
        with pytest.raises(ValueError, match="architecture"):
            quality.verify_config(bad, label)
    quality.verify_pair(*arms)
    bad = copy.deepcopy(arms[1]); bad["manifest"]["config"]["seed"] = 43
    with pytest.raises(ValueError, match="Recipe differs"):
        quality.verify_pair(arms[0], bad)
    assert quality.expected_counts(128)["total"] == 7183157
    assert quality.expected_counts(64)["total"] == 3956469


def test_cessation_covers_workers_and_is_not_a_paused_worker():
    receipt = {"process_cessation": {"confirmed": True, "processes": [{"pid": 7, "alive": False, "exited": True}]}}
    assert quality.verify_cessation(receipt, {7})[0]["pid"] == 7
    with pytest.raises(ValueError, match="omits"):
        quality.verify_cessation(receipt, {7, 8})
    receipt["process_cessation"]["processes"][0]["alive"] = True
    with pytest.raises(ValueError, match="exited"):
        quality.verify_cessation(receipt, {7})


def test_exact_chunk_exposure_matches_log_and_detects_missing_or_corrupt_updates():
    rows = [{"ids": [1, 4, 5, 6, 2]}, {"ids": [1, 8, 2]}]
    cfg = {"epochs": 2, "second_epochs": 0, "batch_size": 2, "chunk_size": 3, "seed": 42}
    table = quality.budget_table(rows, cfg, 4)
    assert [table[i]["tokens"] for i in range(5)] == [0, 5, 6, 11, 12]
    training = [{"updates": i, "tokens": table[i]["chunk_tokens"], **{k: table[i][k] for k in ("epoch", "batch", "offset")}} for i in range(1, 5)]
    arm = {"accepted": {"durable_updates": 4, "observed_updates": 4}, "training_records": training}
    assert quality.verify_training_budget(arm, table)["durable_training_tokens"] == 12
    training[2]["tokens"] += 1
    with pytest.raises(ValueError, match="data exposure"):
        quality.verify_training_budget(arm, table)
    assert quality.budget_table(rows, cfg, 0) == {0: {"tokens": 0, "epoch": 0}}


def test_common_budget_retains_earlier_ties_and_excludes_later_winners():
    def row(update, ce, acc):
        return {"updates": update, "validation": {"cross_entropy": ce, "top1_accuracy": acc}}
    e = {"records": [row(0, 7, .0), row(100, 4, .3), row(200, 4, .3), row(300, 3, .4)]}
    g = {"records": [row(0, 7, .0), row(100, 5, .2), row(200, 4.1, .29)]}
    table = {step: {"tokens": step * 10} for step in (0, 100, 200, 300)}
    result = quality.compare_history(e, g, table)
    assert result["common_budget_updates"] == 200
    assert result["common_budget_training_tokens"] == 2000
    assert result["best_through_common_budget"]["E"]["minimum_validation_ce"]["updates"] == 100
    assert result["matched_updates"][-1]["comparison"]["delta_accuracy"] == pytest.approx(-.01)


def test_per_neuron_differences_direction_and_zero_initial_bias():
    original = {"brain.gain": torch.tensor([1., 2.]), "brain.rec_gain": torch.tensor([2., 3.]), "brain.bias": torch.zeros(2)}
    changed = {"brain.gain": torch.tensor([2., 4.]), "brain.rec_gain": torch.tensor([2., 3.]), "brain.bias": torch.tensor([0., 1.])}
    result = quality.neuron_differences(original, changed)
    assert result["brain.gain"]["mean_delta"] == 1.5
    assert result["brain.gain"]["rms_delta"] == pytest.approx((2.5) ** .5)
    assert result["brain.rec_gain"]["different_values"] == 0
    assert result["brain.bias"]["relative_l2_delta"] is None
    assert result["gain_times_rec_gain"]["maximum_absolute_delta"] == 6


def test_wrong_host_fails_before_mps_is_queried(monkeypatch):
    monkeypatch.setattr(quality.socket, "gethostname", lambda: "other-host.local")
    monkeypatch.setattr(quality.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(quality.torch.backends.mps, "is_available", lambda: pytest.fail("No local MPS call allowed"))
    with pytest.raises(ValueError, match="registered macm3"):
        quality.idle_host_guard()


def test_checkpoint_contents_are_checked_against_parameter_and_selector_hashes():
    initial = {"brain.gain": torch.tensor([1.]), "brain.rec_gain": torch.tensor([2.]), "brain.bias": torch.tensor([0.])}
    manifest = {"initial_parameter_audit": quality.trainer.parameter_audit(initial), "frozen_buffers_sha256": {"edge": "hash"}}
    parameters = {key: value.clone() for key, value in initial.items()}; parameters["brain.gain"] += .2
    cursor = {"updates": 100, "epoch": 0}
    metric = {"cross_entropy": 3., "top1_accuracy": .4}
    entry = {"cursor": cursor, "validation": metric, "frozen_buffers_sha256": {"edge": "hash"}}
    arm = {"manifest": manifest, "selectors": {"minimum_validation_ce": entry}, "latest": entry}
    saved = {"format_version": 1, "manifest": manifest, "cursor": cursor, "frozen_buffers_sha256": {"edge": "hash"},
             "model_audit": {"frozen_buffers_preserved": True, "frozen_buffers_sha256": {"edge": "hash"}},
             "parameters": parameters, "parameter_audit": quality.trainer.parameter_audit(parameters, manifest["initial_parameter_audit"]),
             "best": {**cursor, **metric}}
    quality.verify_saved(saved, arm, "minimum_validation_ce")
    quality.verify_saved(saved, arm)
    saved["parameters"]["brain.gain"] += 1
    with pytest.raises(ValueError, match="hashes"):
        quality.verify_saved(saved, arm, "minimum_validation_ce")
