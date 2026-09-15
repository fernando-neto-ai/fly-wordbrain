"""Independent small CPU oracles for the custom Metal sparse backend.

These tests require actual MPS execution: unsupported hosts skip explicitly,
and native torch.sparse.mm is prohibited during the Metal forward/backward.
Only the tiny fixtures are ever expanded to a dense matrix.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Actual MPS hardware and torch.mps.compile_shader are required",
)


def csr_arrays(rows, columns, values, n):
    order = np.lexsort((columns, rows))
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))].astype(np.int64)
    return ptr, np.asarray(columns[order], np.int64), np.asarray(values[order], np.float32)


def asymmetric_fixture(n=521):
    # The large row spans many SIMD groups/edge-loop iterations. Rows1,3..16,
    # etc. are empty; the matrix differs from its transpose. Zero weights and
    # self-edges are intentionally retained, as well as both signs.
    rows = np.r_[np.zeros(n, np.int64), [2, 2, 2, 17, 17, n - 1, n - 1]]
    columns = np.r_[np.arange(n), [1, 19, n - 1, 0, 38, 2, n - 1]].astype(np.int64)
    values = np.r_[((np.arange(n) % 13) - 6) / 17., [.75, -1.25, 0., -2., .125, .3, -.9]].astype(np.float32)
    incoming = csr_arrays(rows, columns, values, n)
    outgoing = csr_arrays(columns, rows, values, n)
    dense = np.zeros((n, n), np.float64)
    np.add.at(dense, (rows, columns), values.astype(np.float64))
    return incoming, outgoing, torch.from_numpy(dense)


def prohibit_native_sparse(*args, **kwargs):
    raise AssertionError("Metal path called native torch.sparse.mm or attempted CPU fallback")


def state_tensor(values, device, strided):
    values = torch.as_tensor(values, dtype=torch.float32, device=device)
    if not strided:
        return values.clone().detach()
    storage = torch.zeros((values.shape[0], 2 * values.shape[1]), device=device)
    storage[:, ::2] = values
    view = storage[:, ::2]
    assert not view.is_contiguous()
    return view.detach()


def assert_metal_identity(module):
    metadata = module.backend_metadata()
    serialized = json.dumps(metadata, allow_nan=False)
    assert "metal" in serialized.lower() and metadata["backend"] == "metal_custom_csr"
    assert metadata["source_sha256"] == hashlib.sha256(module.METAL_SOURCE.encode()).hexdigest()
    assert metadata["host_fallback"] is False and metadata["per_step_host_transfers"] is False
    assert metadata["dense_adjacency"] is False
    return metadata


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_metal_forward_and_explicit_transpose_gradient_against_dense(batch, strided, monkeypatch):
    from fly_wordbrain import metal_sparse
    from fly_wordbrain.plastic_brain import frozen_sparse_mm

    incoming_arrays, outgoing_arrays, dense = asymmetric_fixture()
    n = dense.shape[0]
    incoming = metal_sparse.DenseMpsCSR(*incoming_arrays, shape=(n, n), device="mps")
    outgoing = metal_sparse.DenseMpsCSR(*outgoing_arrays, shape=(n, n), device="mps")
    assert_metal_identity(metal_sparse)
    generator = torch.Generator().manual_seed(910 + batch)
    values = torch.randn(batch, n, generator=generator)
    upstream = torch.randn(batch, n, generator=generator)
    expected_y = values.double() @ dense.T
    expected_grad = upstream.double() @ dense
    h = state_tensor(values, "mps", strided).requires_grad_()
    grad_output = state_tensor(upstream, "mps", strided)
    before = [tensor.detach().cpu().clone() for matrix in (incoming, outgoing)
              for tensor in (matrix.crow_indices(), matrix.col_indices(), matrix.values())]
    launch_count = metal_sparse.kernel_launch_count()
    monkeypatch.setattr(torch.sparse, "mm", prohibit_native_sparse)
    direct = metal_sparse.mps_csr_mm(h, incoming)
    y = frozen_sparse_mm(h, incoming, outgoing)
    gradient = torch.autograd.grad(y, h, grad_outputs=grad_output)[0]
    torch.mps.synchronize()
    assert metal_sparse.kernel_launch_count() == launch_count + 3
    assert incoming.kernel_launch_count == 2 and outgoing.kernel_launch_count == 1
    assert direct.device.type == y.device.type == gradient.device.type == "mps"
    torch.testing.assert_close(direct.detach().cpu().double(), expected_y, rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(y.detach().cpu().double(), expected_y, rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(gradient.detach().cpu().double(), expected_grad, rtol=2e-5, atol=3e-5)
    assert torch.count_nonzero(y[:, 1]).item() == 0  # empty incoming row
    after = [tensor.detach().cpu() for matrix in (incoming, outgoing)
             for tensor in (matrix.crow_indices(), matrix.col_indices(), matrix.values())]
    for original, final in zip(before, after):
        torch.testing.assert_close(final, original, rtol=0, atol=0)
    assert not incoming.values().requires_grad and not outgoing.values().requires_grad


def write_plastic_fixture(directory):
    n = 12
    pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11], np.int32)
    post = np.array([8, 8, 9, 9, 10, 10, 11, 11, 10, 9, 10, 11, 8], np.int32)
    weight = np.array([.7, -.4, .6, .4, -.5, .8, .9, .3, .5, -.3, .6, .5, -.2], np.float32)
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=n))].astype(np.int64)
    incoming_ids = np.argsort(post, kind="stable")
    incoming_ptr = np.r_[0, np.cumsum(np.bincount(post, minlength=n))].astype(np.int64)
    selected = np.array([0, 2, 4, 6], np.int64)
    np.savez(directory / "graph.npz", ids=np.arange(n, dtype=np.int64), ptr=ptr, post=post, weight=weight,
             incoming_ptr=incoming_ptr, incoming_pre=pre[incoming_ids],
             incoming_weight=weight[incoming_ids], incoming_edge_ids=incoming_ids,
             retina=np.arange(8, dtype=np.int64), descending=np.arange(8, 12, dtype=np.int64),
             candidate_edge_ids=selected, candidate_pre=pre[selected], candidate_post=post[selected],
             candidate_group=np.arange(4, dtype=np.int64))
    (directory / "metadata.json").write_text(json.dumps({"toy_graph": True}))


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_complete_plastic_model_rule_gradients_match_cpu(batch, tmp_path, monkeypatch):
    from fly_wordbrain import metal_sparse
    from fly_wordbrain.plastic_brain import PlasticBrain, PlasticState

    write_plastic_fixture(tmp_path)
    kwargs = dict(vocab_size=8, global_scale=.5, strict_full_graph=False,
                  input_gain=20., internal_steps=4, leak=.5)
    cpu = PlasticBrain(tmp_path, device="cpu", **kwargs)
    mps = PlasticBrain(tmp_path, device="mps", **kwargs)
    assert mps.sparse_backend == "metal_custom_csr"
    assert mps.metadata()["sparse_runtime"]["backend"] == "metal_custom_csr"
    assert_metal_identity(metal_sparse)
    scales = (np.array([.07, .1, .13, .16], np.float32), np.array([.11, .14, .17, .2], np.float32))
    cpu.set_activity_scales(*scales)
    mps.set_activity_scales(*scales)
    with torch.no_grad():
        for name, parameter in mps.named_parameters():
            parameter.copy_(dict(cpu.named_parameters())[name].to("mps"))
    cpu_frozen, mps_frozen = cpu.verify_frozen(), mps.verify_frozen()
    assert cpu_frozen == mps_frozen
    generator = torch.Generator().manual_seed(772 + batch)
    initial_h = .2 + .05 * torch.randn(batch, 12, generator=generator)
    initial_fast = .1 + .02 * torch.randn(batch, 4, generator=generator)
    inputs = [(torch.full((batch,), 2), 4 + torch.arange(batch) % 4)]
    for offset in (1, 2):
        inputs.append((inputs[-1][1], 4 + (torch.arange(batch) + offset) % 4))
    # A fixed small output projection supplies the objective; it has no direct
    # dependence on fast state, so gradients must pass through neural activity.
    coefficients = torch.randn(256, 16, generator=generator) * .2
    targets = torch.arange(batch) % 16

    def rollout(model, device):
        state = PlasticState(state_tensor(initial_h, device, True).requires_grad_(),
                             initial_fast.to(device).clone().requires_grad_())
        initial = state
        observed = []
        for previous, current in inputs:
            features, state = model.step(previous.to(device), current.to(device), state)
            observed.append(features)
        logits = torch.stack(observed).sum(0) @ coefficients.to(device)
        loss = torch.nn.functional.cross_entropy(logits, targets.to(device))
        loss.backward()
        return {"features": torch.stack(observed).detach().cpu(), "h": state.h.detach().cpu(),
                "fast": state.fast.detach().cpu(), "loss": loss.detach().cpu(),
                "initial_h_gradient": initial.h.grad.detach().cpu(),
                "initial_fast_gradient": initial.fast.grad.detach().cpu(),
                "rule_gradients": {name: p.grad.detach().cpu() for name, p in model.named_parameters()}}

    expected = rollout(cpu, "cpu")
    monkeypatch.setattr(torch.sparse, "mm", prohibit_native_sparse)
    launch_count = metal_sparse.kernel_launch_count()
    actual = rollout(mps, "mps")
    torch.mps.synchronize()
    assert metal_sparse.kernel_launch_count() == launch_count + 24
    for name in ("features", "h", "fast", "loss", "initial_h_gradient", "initial_fast_gradient"):
        torch.testing.assert_close(actual[name], expected[name], rtol=4e-4, atol=2e-7, msg=name)
    assert sum(p.numel() for p in mps.parameters()) == 20
    for name, gradient in expected["rule_gradients"].items():
        assert gradient.abs().sum().item() > 1e-12, name
        assert actual["rule_gradients"][name].abs().sum().item() > 1e-12, name
        torch.testing.assert_close(actual["rule_gradients"][name], gradient, rtol=8e-4, atol=2e-8, msg=name)
    extreme = torch.tensor([[-1e6, -2., 2., 1e6]], device="mps")
    effective = mps.effective_candidate_weights(extreme)
    torch.testing.assert_close(torch.sign(effective[0]).cpu(), torch.sign(mps.candidate_weight).cpu(), rtol=0, atol=0)
    ratios = effective / mps.candidate_weight
    assert bool(torch.all((ratios >= .5) & (ratios <= 2.)).item())
    assert cpu.verify_frozen() == cpu_frozen
    assert mps.verify_frozen() == mps_frozen
