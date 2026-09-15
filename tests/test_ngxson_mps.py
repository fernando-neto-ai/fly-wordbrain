"""Numerical and isolation checks; pinned upstream fixture never downloads weights."""
import hashlib
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_wordbrain import ngxson_mps as adapter
from fly_wordbrain.metal_sparse import DenseMpsCSR
from fly_wordbrain.plastic_brain import FrozenSparseMM


PTR = np.array([0, 2, 2, 4, 5, 7], np.int32)
COL = np.array([4, 1, 3, 0, 2, 4, 1], np.int32)
WEIGHT = np.array([.25, -.5, .75, -.25, .5, .125, -.75], np.float32)
MPS = torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
needs_mps = pytest.mark.skipif(not MPS, reason="Native MPS shader runtime unavailable")


def dense_oracle(ptr=PTR, col=COL, weight=WEIGHT, shape=(5, 5)):
    result = np.zeros(shape, np.float32)
    for row in range(shape[0]):
        for edge in range(ptr[row], ptr[row + 1]):
            result[row, col[edge]] += weight[edge]
    return torch.from_numpy(result)


@pytest.fixture(scope="module")
def upstream():
    # This is the source downloaded by the project's explicit preparation
    # command. Tests do not fetch either code or a model checkpoint.
    path = Path(__file__).resolve().parents[1] / "data" / "ngxson-fly-llm-hf" / adapter.UPSTREAM_REVISION
    if not (path / "modeling_fly.py").is_file():
        pytest.skip("Pinned upstream source absent; prepare the upstream artifact first")
    pytest.importorskip("transformers")
    assert hashlib.sha256((path / "modeling_fly.py").read_bytes()).hexdigest() == adapter.MODELING_SHA256
    package = types.ModuleType("_ngxson_pinned_adapter_tests")
    package.__path__ = [str(path)]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + ".modeling_fly")


def tiny_model(upstream):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        config = upstream.FlyConfig(n_neurons=5, n_edges=7, n_in=4, n_out=3,
            delay_k=2, d_embed=4, vocab_size=11, leak=.7, input_scale=.3)
        model = upstream.FlyForCausalLM(config)
    with torch.no_grad():
        model.brain.w_offsets.copy_(torch.from_numpy(PTR))
        model.brain.w_indices.copy_(torch.from_numpy(COL))
        model.brain.w_values.copy_(torch.from_numpy(WEIGHT))
        model.brain.in_index.copy_(torch.tensor([1, 4, 0, 3]))
        model.brain.out_index.copy_(torch.tensor([0, 2, 4]))
        model.brain.gain.copy_(torch.linspace(.8, 1.2, 5))
        model.brain.rec_gain.copy_(torch.linspace(.7, 1.1, 5))
        model.brain.bias.copy_(torch.linspace(-.1, .1, 5))
    return model


def test_sorted_pair_preserves_original_edges_and_empty_rows():
    arrays = [x.copy() for x in (PTR, COL, WEIGHT)]
    forward, transpose = adapter._sorted_csr_pair(*arrays, (5, 5))
    expected = dense_oracle()
    assert torch.equal(dense_oracle(*forward), expected)
    assert torch.equal(dense_oracle(*transpose), expected.t())
    assert all(np.array_equal(a, b) for a, b in zip(arrays, (PTR, COL, WEIGHT)))
    assert np.array_equal(forward[2], WEIGHT[[1, 0, 3, 2, 4, 6, 5]])
    assert forward[0][1] == forward[0][2]


def test_empty_connectome_pair():
    ptr = np.zeros(4, np.int32)
    forward, transpose = adapter._sorted_csr_pair(ptr, np.empty(0, np.int32), np.empty(0, np.float32), (3, 2))
    assert forward[0].tolist() == [0, 0, 0, 0]
    assert transpose[0].tolist() == [0, 0, 0]


@pytest.mark.parametrize("kind", ["duplicates", "bad_pointer", "bad_column", "nan", "double", "float_index"])
def test_malformed_or_unsupported_csr_rejected(kind):
    p, c, w = PTR.copy(), COL.copy(), WEIGHT.copy()
    if kind == "duplicates":
        c[0] = c[1]
    elif kind == "bad_pointer":
        p[-1] -= 1
    elif kind == "bad_column":
        c[0] = 5
    elif kind == "nan":
        w[0] = np.nan
    elif kind == "double":
        w = w.astype(np.float64)
    else:
        c = c.astype(np.float32)
    with pytest.raises(ValueError):
        adapter._sorted_csr_pair(p, c, w, (5, 5))


def test_cpu_fixed_sparse_gradient_independent_dense_oracle():
    forward, transpose = adapter._sorted_csr_pair(PTR, COL, WEIGHT, (5, 5))
    csr = lambda arrays: torch.sparse_csr_tensor(*(torch.from_numpy(x) for x in arrays), size=(5, 5))
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5).requires_grad_()
    reference = x.detach().clone().requires_grad_()
    actual = FrozenSparseMM.apply(x, csr(forward), csr(transpose))
    expected = reference @ dense_oracle().t()
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, reference.grad)


def test_private_function_copy_does_not_mutate_globals():
    function = types.FunctionType((lambda self, x=None: x).__code__, {"_SpMM": object()}, argdefs=(None,))
    function.__kwdefaults__ = {"example": True}
    function.__annotations__ = {"return": object}
    original = function.__globals__["_SpMM"]
    private = adapter._private_forward(function)
    assert private.__code__ is function.__code__
    assert function.__globals__["_SpMM"] is original
    assert private.__globals__["_SpMM"] is adapter._MetalSpMM
    assert private.__globals__ is not function.__globals__
    assert private.__defaults__ == function.__defaults__
    assert private.__kwdefaults__ == function.__kwdefaults__
    assert private.__annotations__ == function.__annotations__


def test_pinned_guard_rejects_changed_instance_forward(upstream):
    model = tiny_model(upstream)
    assert adapter._verify_upstream(model.brain) is upstream.FlyModel.forward
    model.brain.forward = types.MethodType(lambda self, **kwargs: None, model.brain)
    with pytest.raises(ValueError, match="supports only|replaced"):
        adapter._verify_upstream(model.brain)


def test_canonical_buffers_must_be_fixed():
    weights = torch.from_numpy(WEIGHT.copy()).requires_grad_()
    with pytest.raises(ValueError, match="fixed CPU buffer"):
        adapter._sorted_csr_pair(torch.from_numpy(PTR), torch.from_numpy(COL), weights, (5, 5))


@needs_mps
def test_mps_rectangular_spmm_forward_backward_and_repeat():
    ptr = np.array([0, 2, 2, 4, 5], np.int32)
    col = np.array([2, 0, 2, 1, 0], np.int32)
    weights = np.array([.25, -.5, .75, -.125, .5], np.float32)
    pair = adapter._sorted_csr_pair(ptr, col, weights, (4, 3))
    forward = DenseMpsCSR(*pair[0], (4, 3))
    transpose = DenseMpsCSR(*pair[1], (3, 4))
    xcpu = torch.arange(6, dtype=torch.float32).reshape(2, 3).t().requires_grad_()
    x = xcpu.detach().to("mps").requires_grad_()
    assert not x.is_contiguous()
    actual = adapter._MetalSpMM.apply(x, forward, transpose)
    reference = dense_oracle(ptr, col, weights, (4, 3)) @ xcpu
    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=1e-6)
    assert torch.equal(actual, adapter._MetalSpMM.apply(x, forward, transpose))
    assert actual.is_contiguous()
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(x.grad.cpu(), xcpu.grad, rtol=0, atol=1e-6)
    assert forward.kernel_launch_count == 2 and transpose.kernel_launch_count == 1


@needs_mps
def test_pinned_tiny_model_logits_cache_loss_and_all_parameter_gradients(upstream):
    cpu = tiny_model(upstream)
    gpu = tiny_model(upstream)
    before = {name: value.clone() for name, value in gpu.state_dict().items()}
    original_spmm = upstream._SpMM
    original_forward = upstream.FlyModel.forward
    assert adapter.enable_mps(gpu) is gpu
    assert adapter.enable_mps(gpu) is gpu
    assert set(gpu.state_dict()) == set(before)
    assert upstream._SpMM is original_spmm and upstream.FlyModel.forward is original_forward
    assert gpu.brain.forward.__func__.__code__ is original_forward.__code__
    assert cpu.brain.forward.__func__ is original_forward
    assert gpu.brain.forward.__func__.__globals__ is not original_forward.__globals__
    assert not any("ngxson" in name for name, _ in gpu.named_modules())
    ids = torch.tensor([[1, 4, 5, 6], [1, 7, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    labels = ids.clone()
    labels[mask == 0] = -100
    expected = cpu(ids, attention_mask=mask, labels=labels, use_cache=True, output_hidden_states=True)
    actual = gpu(ids.to("mps"), attention_mask=mask.to("mps"), labels=labels.to("mps"),
                 use_cache=True, output_hidden_states=True)
    torch.testing.assert_close(actual.logits.cpu(), expected.logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.loss.cpu(), expected.loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.cache_params.state.cpu(), expected.cache_params.state, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.hidden_states[0].cpu(), expected.hidden_states[0], rtol=2e-5, atol=2e-6)
    assert torch.equal(actual.cache_params.last_tokens.cpu(), expected.cache_params.last_tokens)
    assert actual.cache_params.seq_len == expected.cache_params.seq_len == 4
    actual.loss.backward()
    expected.loss.backward()
    for (name, parameter), (other, reference) in zip(gpu.named_parameters(), cpu.named_parameters()):
        assert name == other and parameter.grad is not None and reference.grad is not None
        torch.testing.assert_close(parameter.grad.cpu(), reference.grad, rtol=5e-4, atol=5e-6, msg=name)
    metadata = adapter.mps_metadata(gpu, verify=True)
    assert metadata["canonical_and_runtime_buffers_verified"]
    assert metadata["forward_kernel_calls"] > 0 and metadata["backward_kernel_calls"] > 0
    for name, value in gpu.state_dict().items():
        assert torch.equal(value.cpu(), before[name])
    adapter.disable_mps(gpu)
    assert gpu.brain.forward.__func__ is original_forward
    assert gpu.brain.connectome.__func__ is upstream.FlyModel.connectome
    gpu.cpu()
    torch.testing.assert_close(gpu(ids, attention_mask=mask).logits, expected.logits, rtol=0, atol=0)


@needs_mps
def test_pinned_cached_sequence_matches_whole_sequence(upstream):
    model = adapter.enable_mps(tiny_model(upstream))
    ids = torch.tensor([[1, 3, 4, 8, 5]], device="mps")
    with torch.no_grad():
        whole = model(ids, use_cache=True)
        first = model(ids[:, :3], use_cache=True)
        second = model(ids[:, 3:], cache_params=first.cache_params, use_cache=True)
    torch.testing.assert_close(torch.cat((first.logits, second.logits), dim=1), whole.logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(second.cache_params.state, whole.cache_params.state, rtol=2e-5, atol=2e-6)
    assert torch.equal(second.cache_params.last_tokens, whole.cache_params.last_tokens)
    assert second.cache_params.seq_len == whole.cache_params.seq_len


@needs_mps
def test_graph_mutation_and_runtime_cache_corruption_detected(upstream):
    model = adapter.enable_mps(tiny_model(upstream))
    with torch.no_grad():
        model.brain.w_values[0].add_(1)
    with pytest.raises(RuntimeError, match="buffers moved or changed"):
        model.brain.connectome()
    model = adapter.enable_mps(tiny_model(upstream))
    runtime = getattr(model.brain, adapter._RUNTIME_ATTRIBUTE)
    with torch.no_grad():
        runtime.forward.values()[0].add_(1)
    with pytest.raises(RuntimeError, match="Derived Metal"):
        adapter.mps_metadata(model, verify=True)


@needs_mps
def test_unsupported_half_precision_rejected_before_model_move(upstream):
    model = tiny_model(upstream).half()
    with pytest.raises(ValueError, match="float32"):
        adapter.enable_mps(model)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert not hasattr(model.brain, adapter._RUNTIME_ATTRIBUTE)


@needs_mps
def test_standalone_embedding_padding_backward_regression(record_property):
    # Reproduces native torch2.11's MPS padding gradient without any sparse
    # matrix, upstream model, masking, or transformer dependencies. A future
    # runtime that already fixes the bug also satisfies the compatibility test.
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    ids = torch.tensor([0, 1, 0, 3])
    models = [torch.nn.Embedding(5, 4, padding_idx=0) for _ in range(3)]
    for model in models:
        with torch.no_grad():
            model.weight.copy_(weight)  # PAD deliberately has nonzero values.
    models[1].to("mps")
    models[2].to("mps")
    handle = adapter._padding_gradient_hook(models[2])
    for model in models:
        device = model.weight.device
        output = model(ids.to(device))
        assert torch.equal(output.cpu(), weight[ids])
        output.sum().backward()
    native = models[1].weight.grad.cpu()
    record_property("native_mps_padding_gradient_max_abs", float(native[0].abs().max()))
    record_property("torch_version", torch.__version__)
    assert torch.equal(native[1:], models[0].weight.grad[1:])
    assert torch.equal(models[2].weight.grad.cpu(), models[0].weight.grad)
    assert torch.count_nonzero(models[2].weight.grad[0]) == 0
    assert torch.equal(models[2].weight.detach().cpu(), weight)
    handle.remove()


@needs_mps
def test_padding_hook_lifecycle_frozen_parameter_and_nonzero_forward(upstream):
    model = tiny_model(upstream)
    with torch.no_grad():
        model.brain.wte.weight[0].fill_(.125)
    model.brain.wte.weight.requires_grad_(False)
    before = model.brain.wte.weight.detach().clone()
    before_hooks = len(model.brain.wte.weight._backward_hooks or {})
    adapter.enable_mps(model)
    adapter.enable_mps(model)
    assert not model.brain.wte.weight.requires_grad
    assert len(model.brain.wte.weight._backward_hooks) == before_hooks + 1
    assert torch.equal(model.brain.wte.weight.detach().cpu(), before)
    model.brain.wte.weight.requires_grad_(True)
    output = model.brain.wte(torch.tensor([0, 1], device="mps"))
    assert torch.equal(output[0].cpu(), before[0])
    output.sum().backward()
    assert torch.count_nonzero(model.brain.wte.weight.grad[0]) == 0
    adapter.disable_mps(model)
    adapter.disable_mps(model)
    assert len(model.brain.wte.weight._backward_hooks or {}) == before_hooks
    assert not hasattr(model.brain, adapter._RUNTIME_ATTRIBUTE)


@needs_mps
def test_pinned_adamw_steps_and_momentum_match_cpu(upstream):
    cpu = tiny_model(upstream)
    gpu = tiny_model(upstream)
    # Nonzero padding weights still undergo normal AdamW weight decay,
    # matching upstream CPU behavior; the adapter does not alter forward data.
    with torch.no_grad():
        cpu.brain.wte.weight[0].fill_(.125)
        gpu.brain.wte.weight[0].fill_(.125)
    adapter.enable_mps(gpu)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1, foreach=False)
                  for model in (cpu, gpu)]
    ids = torch.tensor([[1, 3, 4, 6], [1, 5, 7, 8]])
    for _ in range(2):
        for model, optimizer in zip((cpu, gpu), optimizers):
            optimizer.zero_grad(set_to_none=True)
            device = next(model.parameters()).device
            model(ids.to(device), labels=ids.to(device)).loss.backward()
            optimizer.step()
        for (name, expected), (other, actual) in zip(cpu.named_parameters(), gpu.named_parameters()):
            assert name == other
            torch.testing.assert_close(actual.detach().cpu(), expected.detach(), rtol=2e-5, atol=2e-6, msg=name)
            left, right = optimizers[0].state[expected], optimizers[1].state[actual]
            assert set(left) == set(right)
            for state_name in left:
                torch.testing.assert_close(right[state_name].cpu(), left[state_name], rtol=5e-4,
                                           atol=1e-8, msg=name + ":" + state_name)
        assert torch.count_nonzero(gpu.brain.wte.weight.grad[0]) == 0
        assert torch.count_nonzero(optimizers[1].state[gpu.brain.wte.weight]["exp_avg"][0]) == 0


@needs_mps
def test_embedding_replacement_requires_rebinding_guard(upstream):
    model = adapter.enable_mps(tiny_model(upstream))
    model.set_input_embeddings(torch.nn.Embedding(11, 4, padding_idx=0, device="mps"))
    with pytest.raises(RuntimeError, match="embedding replaced"):
        model.brain.connectome()
