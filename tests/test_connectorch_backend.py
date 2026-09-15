"""Exercise the registered backend through the pinned ConnecTorch public API."""
import copy

import numpy as np
import pytest
import torch

ct = pytest.importorskip("connectorch")

from fly_wordbrain.connectorch_backend import (
    ConnectorchCSR, MetalCSRPropagator, exact_connectome_ir,
    register_connectorch_backend,
)


DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Requires actual Apple GPU and native shader compilation"))]


def checkpoint_fixture(alternative=False):
    ptr = np.array([0, 2, 2, 3, 5, 6], np.int64)
    col = np.array([0, 3, 1, 0, 2, 4], np.int64)
    if alternative:
        col = np.array([1, 4, 0, 1, 3, 2], np.int64)
    values = np.array([.4, -.6, 0., .8, -.3, .1], np.float32)
    return {"brain.w_offsets": ptr, "brain.w_indices": col, "brain.w_values": values}, np.array([10, 20, 30, 40, 50])


def fixture_graph(alternative=False):
    checkpoint, ids = checkpoint_fixture(alternative)
    return exact_connectome_ir(checkpoint, ids, ["A", "A", "B", "C", "C"])


def make_model(graph, backend, device="cpu", bounded=False):
    weights = ct.nn.BiologicalWeights(graph, prior="weight", gain_bounds=(.9, 1.1),
                                     share_by="cell_type", dale=False) if bounded else "trainable"
    return ct.nn.ConnectomeRNN(graph, weights=weights, initializer="weight", input_nodes=[10, 40],
                              output_nodes=[30, 10, 50], activation="tanh", leak=.7,
                              bias=True, backend=backend, device=device)


def forbid_native_sparse(*args, **kwargs):
    raise AssertionError("Registered Metal backend attempted torch.sparse.mm or host fallback")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bounded", [False, True])
def test_registered_recurrent_model_matches_independent_dense_backend(device, bounded, monkeypatch):
    receipt = register_connectorch_backend()
    assert receipt["revision"] == "4bbfb645099aeb85bdbf850e1a87cc094769af87"
    graph = fixture_graph()
    dense = make_model(graph, "dense", bounded=bounded)
    native = make_model(graph, "metal_csr", device, bounded)
    # Backend implementation is a derived detail, not an extra checkpoint field.
    native.load_state_dict(dense.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(260915)
    initial = torch.randn(3, 5, generator=generator) * .25
    drive = torch.randn(3, 4, 2, generator=generator) * .3
    coefficients = torch.randn(3, 4, 3, generator=generator)
    cpu_drive, cpu_state = drive.requires_grad_(), initial.requires_grad_()
    gpu_drive = drive.detach().to(device).clone().requires_grad_()
    gpu_state = initial.detach().to(device).clone().requires_grad_()
    cpu_y, cpu_final = dense(cpu_drive, state=cpu_state, return_state=True)
    cpu_loss = (cpu_y * coefficients).sum() + cpu_final.square().sum()
    cpu_loss.backward()
    if device == "mps":
        monkeypatch.setattr(torch.sparse, "mm", forbid_native_sparse)
    gpu_y, gpu_final = native(gpu_drive, state=gpu_state, return_state=True)
    gpu_loss = (gpu_y * coefficients.to(device)).sum() + gpu_final.square().sum()
    gpu_loss.backward()
    for actual, expected in ((gpu_y, cpu_y), (gpu_final, cpu_final), (gpu_loss, cpu_loss),
                             (gpu_drive.grad, cpu_drive.grad), (gpu_state.grad, cpu_state.grad)):
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=2e-6)
    assert dict(native.named_parameters()).keys() == dict(dense.named_parameters()).keys()
    for name, parameter in native.named_parameters():
        expected = dict(dense.named_parameters())[name].grad
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert expected.abs().sum() > 0
        torch.testing.assert_close(parameter.grad.cpu(), expected, rtol=5e-5, atol=3e-6, msg=name)
    assert native.backend == "metal_csr"
    assert native.propagator.verify_frozen()
    if device == "mps":
        assert native.propagator.metadata()["kernel_launch_count"] == {
            "forward": 4, "state_backward": 4, "edge_backward": 4}


@pytest.mark.parametrize("device", DEVICES)
def test_actual_biological_weights_preserve_signed_zero_prior_and_bounds(device):
    register_connectorch_backend()
    graph = fixture_graph()
    model = make_model(graph, "metal_csr", device, bounded=True)
    base = torch.from_numpy(graph.edge_attribute("weight").copy()).to(device)
    torch.testing.assert_close(model.edge_weight, base, rtol=0, atol=0)
    for raw_value in (-100., 100., .6):
        with torch.no_grad():
            model.weights.raw_gain.fill_(raw_value)
        changed = model.edge_weight
        assert torch.equal(torch.sign(changed), torch.sign(base))
        assert changed[base == 0].count_nonzero() == 0
        ratio = changed[base != 0] / base[base != 0]
        assert bool(((ratio >= .9 - 1e-6) & (ratio <= 1.1 + 1e-6)).all())


@pytest.mark.parametrize("device", DEVICES)
def test_state_dict_rebuilds_changed_topology_and_is_backend_portable(device):
    register_connectorch_backend()
    graph_a, graph_b = fixture_graph(), fixture_graph(alternative=True)
    model_a = make_model(graph_a, "metal_csr", device)
    model_b = make_model(graph_b, "metal_csr", device)
    x = torch.tensor([[[.1, .3], [-.2, .2], [.5, -.1]]], device=device)
    initial = torch.tensor([[.1, -.2, .3, -.4, .5]], device=device)
    before = model_a(x, state=initial)
    expected = model_b(x, state=initial)
    assert not torch.allclose(before, expected, atol=1e-6, rtol=1e-6)
    runtime = model_a.propagator._runtime
    state = copy.deepcopy(model_b.state_dict())
    assert not any(name.startswith("propagator.") for name in state)
    model_a.load_state_dict(state, strict=True)
    assert model_a.propagator._runtime is None
    actual = model_a(x, state=initial)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model_a.propagator._runtime is not runtime
    assert model_a.fingerprint == graph_b.fingerprint()
    dense = make_model(graph_a, "dense")
    dense.load_state_dict(state, strict=True)
    torch.testing.assert_close(dense(x.cpu(), state=initial.cpu()), actual.cpu(), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_to_discards_device_specific_runtime_and_retains_exact_output(device):
    register_connectorch_backend()
    model = make_model(fixture_graph(), "metal_csr")
    x = torch.ones(2, 3, 2) * .1
    expected = model(x)
    original_runtime = model.propagator._runtime
    assert original_runtime is not None
    model.to(device)
    assert model.propagator._runtime is None
    actual = model(x.to(device))
    assert model.propagator._runtime is not original_runtime
    assert model.propagator._runtime.device.type == device
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    model.to("cpu")
    assert model.propagator._runtime is None
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_exact_ir_preserves_biological_ids_and_canonical_zero_edges():
    checkpoint, ids = checkpoint_fixture()
    graph = exact_connectome_ir(checkpoint, ids)
    assert graph.num_nodes == 5 and graph.num_edges == 6
    np.testing.assert_array_equal(graph.node_ids, ids)
    np.testing.assert_array_equal(graph.edge_index[0], checkpoint["brain.w_indices"])
    np.testing.assert_array_equal(graph.edge_attribute("weight"), checkpoint["brain.w_values"])
    assert np.sum(graph.edge_attribute("weight") == 0) == 1
    # Node 20 has no incoming edge but is retained in the imported population.
    assert not np.any(graph.edge_index[1] == 1)
    checkpoint["brain.w_indices"][:2] = [3, 0]
    with pytest.raises(ValueError, match="changed canonical"):
        exact_connectome_ir(checkpoint, ids)


def test_registration_is_idempotent_and_does_not_replace_other_backends():
    from connectorch.backends.base import BACKENDS
    before = {name: value for name, value in BACKENDS.items() if name != "metal_csr"}
    register_connectorch_backend()
    register_connectorch_backend()
    assert BACKENDS["metal_csr"] is MetalCSRPropagator
    assert {name: value for name, value in BACKENDS.items() if name != "metal_csr"} == before


def test_bridge_rejects_noninteger_indices_before_casting():
    checkpoint, _ = checkpoint_fixture()
    for name in ("brain.w_offsets", "brain.w_indices"):
        changed = dict(checkpoint)
        changed[name] = changed[name].astype(np.float64)
        with pytest.raises(ValueError, match="integer"):
            ConnectorchCSR(changed["brain.w_offsets"], changed["brain.w_indices"], (5, 5))


@pytest.mark.parametrize("device", DEVICES)
def test_propagator_audit_catches_endpoint_data_mutations_and_explicit_rebuild(device):
    register_connectorch_backend()
    graph_a, graph_b = fixture_graph(), fixture_graph(alternative=True)
    model = make_model(graph_a, "metal_csr", device)
    state = torch.ones(5, 2, device=device)
    model.propagator(state, model.edge_weight)
    model.propagator.edge_index.data[0, 0] = 2
    with pytest.raises(RuntimeError, match="endpoints differ"):
        model.propagator.verify_frozen()
    model.propagator.rebuild(torch.as_tensor(graph_b.edge_index.copy()))
    actual = model.propagator(state, model.edge_weight)
    expected_model = make_model(graph_b, "dense")
    expected = expected_model.propagator(state.cpu(), expected_model.edge_weight)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    assert model.propagator.verify_frozen()
