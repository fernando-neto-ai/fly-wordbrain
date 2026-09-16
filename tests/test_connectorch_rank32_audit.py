"""Rank32 preflight contracts using CPU tensors; no model forward or optimizer."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("rank32_audit", ROOT / "scripts/audit_connectorch_rank32.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def parameters():
    shapes = {"brain.wte.weight": (1024, 32), "brain.in_proj": (8, 32, 1758),
              "brain.gain": (49393,), "brain.rec_gain": (49393,), "brain.bias": (49393,),
              "ln.weight": (49393,), "ln.bias": (49393,),
              "lm_head.0.weight": (32, 49393), "lm_head.1.weight": (1024, 32)}
    return {name: torch.ones(shape, dtype=torch.float32) for name, shape in shapes.items()}


def test_rank32_counts_and_initial_parameters_match_exactly():
    first = parameters()
    second = {name: value.clone() for name, value in first.items()}
    result = audit.verify_initialization(first, second)
    assert result["passed"] and result["cpu_mps_parameter_hashes_equal"]
    assert sum(v.numel() for v in first.values()) == 2343125
    assert first["lm_head.0.weight"].numel() + first["lm_head.1.weight"].numel() == 1613344
    second["lm_head.1.weight"][0, 0] = 2
    with pytest.raises(ValueError, match="initial parameter contents"):
        audit.verify_initialization(first, second)


def test_rank32_rejects_wrong_factor_shape_and_adaptive_parameters():
    first = parameters()
    first["lm_head.0.weight"] = torch.ones(128, 49393)
    with pytest.raises(ValueError, match="factor shape"):
        audit.verify_initialization(first, first)
    first["brain.edge_theta_source"] = torch.zeros(9161)
    with pytest.raises(ValueError, match="parameter names"):
        audit.verify_initialization(first, first)


@pytest.mark.parametrize("factor", ["lm_head.0.weight", "lm_head.1.weight"])
def test_both_readout_factors_require_nonzero_gradients(factor):
    values = parameters()
    record = {"passed": True, "parameter_count": 2343125}
    assert audit.verify_gradients(record, {"gradients": values})["all_required_gradients_nonzero"]
    values[factor].zero_()
    with pytest.raises(ValueError, match="zero or nonfinite"):
        audit.verify_gradients(record, {"gradients": values})


def test_nonfinite_input_gradient_cannot_pass():
    values = parameters()
    values["brain.in_proj"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="brain.in_proj"):
        audit.verify_gradients({"passed": True, "parameter_count": 2343125}, {"gradients": values})


def test_completion_requires_exact_g_cases_and_not_old_rank128_success():
    result = {"cases": {audit.ARM + "_" + device: {"passed": True, "all_required_gradients_nonzero": True, "targets": 64}
                        for device in ("cpu", "mps")},
              "comparisons": {audit.ARM + "_cpu_vs_mps": {"passed": True}},
              "initialization": {"passed": True}, "exact_ir": {"passed": True},
              "sources_unchanged": True, "connectorch_unchanged": True, "inputs_unchanged": True,
              "final_host_guard": {"passed": True}}
    assert audit.required_gates_pass(result)
    assert not audit.required_gates_pass({})
    for field in ("cases", "comparisons", "initialization"):
        broken = copy.deepcopy(result)
        broken[field] = {}
        assert not audit.required_gates_pass(broken)
    broken = copy.deepcopy(result)
    broken["cases"]["E32rank128fixed_mps"] = broken["cases"].pop(audit.ARM + "_mps")
    assert not audit.required_gates_pass(broken)
