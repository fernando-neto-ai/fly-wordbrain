"""Decoder preflight gates on CPU tensor fixtures; no model training or optimizer."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("decoder_audit", ROOT / "scripts/audit_connectorch_decoder.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def initial_values():
    fixed = {name: torch.tensor([1., 2.]) for name in audit.COMMON_PARAMETERS}
    bounded = {name: value.clone() for name, value in fixed.items()}
    bounded.update({name: torch.zeros(9161) for name in audit.EXTRA_PARAMETERS})
    logits = torch.tensor([[[1., 2.], [3., 4.]]])
    return fixed, bounded, logits, logits.clone()


def test_initialization_requires_exact_common_values_and_two_zero_gains():
    inputs = initial_values()
    receipt = audit.verify_initialization(*inputs)
    assert receipt["passed"] and receipt["shared_parameter_hashes_equal"] and receipt["logits_equal"]
    assert receipt["extra_parameter_count"] == 18322
    inputs[1]["lm_head.0.weight"][0] += 1
    with pytest.raises(ValueError, match="Shared E/F"):
        audit.verify_initialization(*inputs)


@pytest.mark.parametrize("damage", ["nonzero_gain", "wrong_gain_size", "extra_parameter", "different_logits"])
def test_initialization_rejects_architecture_or_functional_mismatch(damage):
    fixed, bounded, first, second = initial_values()
    name = "brain.edge_theta_source"
    if damage == "nonzero_gain":
        bounded[name][0] = .1
    elif damage == "wrong_gain_size":
        bounded[name] = torch.zeros(9160)
    elif damage == "extra_parameter":
        bounded["extra"] = torch.ones(1)
    else:
        second[0, 0, 0] += .01
    with pytest.raises(ValueError):
        audit.verify_initialization(fixed, bounded, first, second)


@pytest.mark.parametrize("plasticity", ["fixed", "bounded10"])
def test_gradients_require_both_head_factors_and_every_declared_path(plasticity):
    names = audit.COMMON_PARAMETERS | (audit.EXTRA_PARAMETERS if plasticity == "bounded10" else set())
    gradients = {name: torch.tensor([.1, .2]) for name in names}
    record = {"passed": True, "parameter_count": audit.TOTALS[plasticity]}
    checked = audit.verify_gradients(record, {"gradients": gradients}, plasticity)
    assert checked["all_required_gradients_nonzero"]
    assert all(checked["required_gradients"][name]["nonzero_values"] == 2 for name in names)
    gradients["lm_head.1.weight"].zero_()
    with pytest.raises(ValueError, match="zero or nonfinite: lm_head.1.weight"):
        audit.verify_gradients(record, {"gradients": gradients}, plasticity)


def test_gradients_reject_missing_names_nonfinite_values_and_wrong_counts():
    gradients = {name: torch.ones(1) for name in audit.COMMON_PARAMETERS}
    record = {"passed": True, "parameter_count": audit.TOTALS["fixed"]}
    del gradients["brain.in_proj"]
    with pytest.raises(ValueError, match="gradient names"):
        audit.verify_gradients(record, {"gradients": gradients}, "fixed")
    gradients["brain.in_proj"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="nonfinite"):
        audit.verify_gradients(record, {"gradients": gradients}, "fixed")
    with pytest.raises(ValueError, match="parameter count"):
        audit.verify_gradients({**record, "parameter_count": 51308213}, {"gradients": gradients}, "fixed")


def test_process_guard_allows_own_controller_and_held_prior_queue_only():
    inventory = "10 1 S python scripts/run_connectorch_decoder_campaign.py --worker\n20 10 R python scripts/audit_connectorch_decoder.py\n30 1 Ts python scripts/continue_connectorch_campaign.py --worker"
    receipt = audit.process_guard(inventory, 20)
    assert receipt["ancestor_dispatchers"][0]["pid"] == 10
    assert receipt["held_dispatchers"][0]["pid"] == 30
    with pytest.raises(ValueError, match="active dispatcher"):
        audit.process_guard(inventory + "\n40 1 S python scripts/run_connectorch_decoder_campaign.py --worker", 20)
    for state in ("S", "Ts"):
        with pytest.raises(ValueError, match="Another training"):
            audit.process_guard(inventory + f"\n40 1 {state} /venv/bin/Python scripts/train_connectorch.py", 20)
    with pytest.raises(ValueError, match="Another training"):
        audit.process_guard(inventory + "\n40 1 R python scripts/evaluate_connectorch_quality.py", 20)
    assert audit.process_guard(inventory + "\n40 1 Z <defunct>", 20)["passed"]


def test_guard_rejects_wrong_host_before_native_execution(monkeypatch):
    monkeypatch.setattr(audit.socket, "gethostname", lambda: "not-macm3.local")
    monkeypatch.setattr(audit.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="registered macm3"):
        audit.host_guard()
    monkeypatch.setattr(audit.socket, "gethostname", lambda: "Fernandos-MacBook-Pro-2.local")
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *args, **kwargs: "Apple M4 Max")
    with pytest.raises(ValueError, match="Apple M3"):
        audit.host_guard()


def test_completion_cannot_pass_missing_cases_or_comparisons():
    result = {"cases": {name + "_" + device: {"passed": True, "all_required_gradients_nonzero": True, "targets": 64}
                        for name in audit.ARMS for device in ("cpu", "mps")},
              "comparisons": {name + "_cpu_vs_mps": {"passed": True} for name in audit.ARMS},
              "initialization_equivalence": {"passed": True}, "exact_ir": {"passed": True},
              "sources_unchanged": True, "connectorch_unchanged": True, "inputs_unchanged": True,
              "final_host_guard": {"passed": True}}
    assert audit.required_gates_pass(result)
    assert not audit.required_gates_pass({})
    for key in ("cases", "comparisons", "initialization_equivalence"):
        damaged = copy.deepcopy(result)
        damaged[key] = {}
        assert not audit.required_gates_pass(damaged)
    damaged = copy.deepcopy(result)
    damaged["cases"]["F32rank128bounded_mps"]["targets"] = 0
    assert not audit.required_gates_pass(damaged)
