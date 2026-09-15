"""Independent dense oracles for dynamic CSR and its native edge gradients."""
import numpy as np
import pytest
import torch

from fly_wordbrain.metal_sparse_trainable import TrainableCSR


DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Requires actual Apple GPU and native shader compilation"))]


def fixture():
    # Rectangular, unsorted, empty rows/columns, retained zero and negative edges.
    ptr = np.array([0, 3, 3, 4, 6], dtype=np.int64)
    col = np.array([4, 0, 2, 1, 4, 0], dtype=np.int64)
    row = np.array([0, 0, 0, 2, 3, 3], dtype=np.int64)
    return ptr, col, row, torch.tensor([.25, -1.5, 0., .75, -.4, .6]), (4, 6)


def prohibit_sparse(*args, **kwargs):
    raise AssertionError("MPS dynamic CSR attempted native sparse execution or CPU fallback")


def strided(value, device):
    storage = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype, device=device)
    storage[..., ::2] = value.to(device)
    return storage[..., ::2].detach()


def dense_oracle(h, value, row, col, shape):
    # index_put gives an independently differentiable matrix, used only for tiny fixtures.
    matrix = torch.zeros(shape, dtype=h.dtype, device=h.device)
    matrix = matrix.index_put((torch.as_tensor(row, device=h.device), torch.as_tensor(col, device=h.device)), value)
    return h @ matrix.T


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [1, 8, 33])
def test_forward_input_and_edge_gradients(device, batch, monkeypatch):
    ptr, col, row, initial, shape = fixture()
    graph = TrainableCSR(ptr, col, shape, device=device)
    rng = torch.Generator().manual_seed(1981 + batch)
    h_cpu = torch.randn(batch, shape[1], generator=rng).requires_grad_()
    v_cpu = initial.clone().requires_grad_()
    upstream = torch.randn(batch, shape[0], generator=rng)
    expected = dense_oracle(h_cpu, v_cpu, row, col, shape)
    expected_grad = torch.autograd.grad(expected, (h_cpu, v_cpu), upstream)
    h = strided(h_cpu.detach(), device).requires_grad_()
    v = strided(initial, device).requires_grad_()
    assert not h.is_contiguous() and not v.is_contiguous()
    if device == "mps":
        monkeypatch.setattr(torch.sparse, "mm", prohibit_sparse)
    actual = graph.mm(h, v)
    actual_grad = torch.autograd.grad(actual, (h, v), strided(upstream, device))
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    for result, target in zip(actual_grad, expected_grad):
        torch.testing.assert_close(result.cpu(), target, rtol=2e-5, atol=2e-6)
    assert actual[:, 1].count_nonzero().item() == 0
    assert actual_grad[0][:, 3].count_nonzero().item() == 0
    # A stored zero-valued edge remains learnable, with the same canonical index.
    assert abs(actual_grad[1][2].item()) > 1e-5
    assert graph.verify_frozen()
    if device == "mps":
        assert graph.kernel_launch_count == {"forward": 1, "state_backward": 1, "edge_backward": 1}


@pytest.mark.parametrize("device", DEVICES)
def test_long_unsorted_row_spans_simd_reduction(device):
    n = 521
    col = np.random.default_rng(82).permutation(n)
    ptr = [0, n, n, n]
    row = np.zeros(n, dtype=np.int64)
    shape = (3, n)
    values = torch.linspace(-.9, .8, n)
    rng = torch.Generator().manual_seed(44)
    h = torch.randn(8, n, generator=rng)
    upstream = torch.randn(8, 3, generator=rng)
    cpu_h, cpu_v = h.double().requires_grad_(), values.double().requires_grad_()
    expected = dense_oracle(cpu_h, cpu_v, row, col, shape)
    expected_grad = torch.autograd.grad(expected, (cpu_h, cpu_v), upstream.double())
    device_h, device_v = h.to(device).requires_grad_(), values.to(device).requires_grad_()
    actual = TrainableCSR(ptr, col, shape, device).mm(device_h, device_v)
    actual_grad = torch.autograd.grad(actual, (device_h, device_v), upstream.to(device))
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=3e-5, atol=2e-5)
    for result, target in zip(actual_grad, expected_grad):
        torch.testing.assert_close(result.cpu().double(), target, rtol=3e-5, atol=2e-6)


def test_cpu_double_finite_difference():
    ptr, col, _, initial, shape = fixture()
    graph = TrainableCSR(ptr, col, shape)
    h = torch.randn(2, shape[1], dtype=torch.float64, requires_grad=True)
    v = initial.double().requires_grad_()
    assert torch.autograd.gradcheck(graph.mm, (h, v), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_multiple_optimizer_updates_and_grouped_value_chain(device):
    ptr, col, row, base, shape = fixture()
    group = torch.tensor([2, 0, 1, 0, 2, 1])
    graph = TrainableCSR(ptr, col, shape, device)
    rng = torch.Generator().manual_seed(768)
    h0 = torch.randn(8, shape[1], generator=rng)
    target = torch.randn(8, shape[0], generator=rng)
    cpu_h = torch.nn.Parameter(h0.clone())
    cpu_g = torch.nn.Parameter(torch.tensor([.03, -.05, .12]))
    gpu_h = torch.nn.Parameter(h0.to(device))
    gpu_g = torch.nn.Parameter(cpu_g.detach().to(device).clone())
    cpu_opt = torch.optim.AdamW([cpu_h, cpu_g], lr=.015)
    gpu_opt = torch.optim.AdamW([gpu_h, gpu_g], lr=.015)
    snapshots = []
    for _ in range(4):
        cpu_opt.zero_grad()
        gpu_opt.zero_grad()
        cpu_v = base * (1. + .1 * torch.tanh(cpu_g[group]))
        gpu_v = base.to(device) * (1. + .1 * torch.tanh(gpu_g[group.to(device)]))
        cpu_y = dense_oracle(cpu_h, cpu_v, row, col, shape)
        gpu_y = graph.mm(gpu_h, gpu_v)
        cpu_loss = ((cpu_y - target) ** 2).mean()
        gpu_loss = ((gpu_y - target.to(device)) ** 2).mean()
        cpu_loss.backward()
        gpu_loss.backward()
        for expected, actual in ((cpu_h, gpu_h), (cpu_g, gpu_g)):
            torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-5, atol=2e-6)
        cpu_opt.step()
        gpu_opt.step()
        torch.testing.assert_close(gpu_g.cpu(), cpu_g, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(gpu_h.cpu(), cpu_h, rtol=2e-5, atol=2e-6)
        snapshots.append(gpu_v.detach().cpu())
    assert not torch.equal(snapshots[0], snapshots[-1])
    assert graph.verify_frozen()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("empty_batch,empty_edges", [(True, False), (False, True), (True, True)])
def test_empty_shapes_produce_zero_values_and_gradients(device, empty_batch, empty_edges):
    ptr, col, _, v, shape = fixture()
    if empty_edges:
        ptr = [0] * (shape[0] + 1)
        col = np.array([], dtype=np.int64)
        v = torch.tensor([])
    graph = TrainableCSR(ptr, col, shape, device)
    h = torch.ones(0 if empty_batch else 2, shape[1], device=device, requires_grad=True)
    v = v.to(device).requires_grad_()
    y = graph.mm(h, v)
    dh, dv = torch.autograd.grad(y.sum(), (h, v))
    assert y.shape == (h.shape[0], shape[0])
    assert y.count_nonzero().item() == dh.count_nonzero().item() == dv.count_nonzero().item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_saved_tensors_are_states_and_values_only(device):
    ptr, col, _, v, shape = fixture()
    graph = TrainableCSR(ptr, col, shape, device)
    h = torch.ones(8, shape[1], device=device, requires_grad=True)
    v = v.to(device).requires_grad_()
    saved = []
    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.numel()))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        graph.mm(h, v).sum().backward()
    assert saved == [(tuple(h.shape), h.numel()), (tuple(v.shape), v.numel())]
    assert not graph.metadata()["edge_batch_intermediate"]


@pytest.mark.parametrize("device", DEVICES)
def test_selective_gradients_and_higher_order_guard(device):
    ptr, col, _, base, shape = fixture()
    graph = TrainableCSR(ptr, col, shape, device)
    h = torch.ones(2, shape[1], device=device, requires_grad=True)
    v = base.to(device)
    assert torch.autograd.grad(graph.mm(h, v).sum(), h)[0].shape == h.shape
    v = v.requires_grad_()
    assert torch.autograd.grad(graph.mm(h.detach(), v).sum(), v)[0].shape == v.shape
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(graph.mm(h, v).sum(), h, create_graph=True)


@pytest.mark.parametrize("device", DEVICES)
def test_topology_copies_inputs_and_detects_mutation(device):
    ptr, col, _, v, shape = fixture()
    graph = TrainableCSR(ptr, col, shape, device)
    ptr[:] = 0
    col[:] = 0
    assert graph.verify_frozen()
    graph._buffers["transpose_edge"][0] = 1
    with pytest.raises(RuntimeError, match="cache was mutated"):
        graph.mm(torch.ones(2, shape[1], device=device), v.to(device))
    graph = TrainableCSR(*fixture()[:2], shape, device)
    graph._buffers["col"].data[0] = 5
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        graph.verify_frozen()


@pytest.mark.parametrize("ptr,col,shape,error", [
    ([0, 1, 2], [1, 0], (2, 2), None),
    ([0, 2, 2], [1, 1], (2, 2), "Duplicate"),
    ([0, 1, 2], [1., 0.], (2, 2), "integer array"),
    ([0, 3, 2], [1, 0], (2, 2), "inconsistent"),
    ([0, 1], [1, 0], (2, 2), "inconsistent"),
    ([1, 1, 2], [1, 0], (2, 2), "inconsistent"),
    ([0, 1, 2], [2, 0], (2, 2), "out of bounds"),
    ([0, 1, 2], [-1, 0], (2, 2), "nonnegative"),
    ([0, 1, 2], [1, 0], (0, 2), "positive integer"),
    ([0, 1, 2], [1, 0], (2., 2), "positive integer"),
])
def test_topology_validation(ptr, col, shape, error):
    if error is None:
        assert TrainableCSR(ptr, col, shape).verify_frozen()
    else:
        with pytest.raises(ValueError, match=error):
            TrainableCSR(ptr, col, shape)


@pytest.mark.parametrize("device", DEVICES)
def test_input_validation(device):
    ptr, col, _, v, shape = fixture()
    graph = TrainableCSR(ptr, col, shape, device)
    h, v = torch.ones(2, shape[1], device=device), v.to(device)
    for bad_h, bad_v, error in [(h.half(), v.half(), "float32"), (h, v[:-1], "Expected"),
                                 (h[:, :-1], v, "Expected"), (h, v.reshape(2, -1), "Expected")]:
        with pytest.raises(ValueError, match=error):
            graph.mm(bad_h, bad_v)
    if device == "mps":
        with pytest.raises(ValueError, match="resident"):
            graph.mm(h.cpu(), v)
