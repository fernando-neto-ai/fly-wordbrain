"""Instance-scoped Metal sparse backend for the pinned ngxson Fly LLM.

Load the original checkpoint on CPU in float32, then call ``enable_mps(model)``.
Only that model's brain gets a private copy of its original forward function
with a different fixed-matrix multiplication primitive. Parameters, canonical
buffers, delay line, activation, masking, cache, and language head are unchanged.
Runtime CSR caches are deliberately absent from the original state_dict.

On the tested torch2.11 MPS runtime, nn.Embedding backward does not suppress
the padding_idx row (also reproduced without sparse operations). A weight
gradient hook restores the original CPU semantics for that row only. It does
not zero, rewrite, or detach the padding embedding in the forward pass.
"""
import hashlib
import inspect
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn

from .metal_sparse import DenseMpsCSR, _validate_cpu_csr, backend_metadata
from .plastic_brain import FrozenSparseMM


UPSTREAM_REPOSITORY = "ngxson/fly-llm-hf"
UPSTREAM_REVISION = "65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
MODELING_SHA256 = "43efb042f2809fe87eb172718511aa884b7b9f7c356181bede9f70148bb87319"
_RUNTIME_ATTRIBUTE = "_ngxson_mps_runtime"
_CANONICAL_NAMES = ("w_offsets", "w_indices", "w_values")


def _digest(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _numpy_fixed(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError("%s must be a fixed CPU buffer before enabling MPS" % name)
        value = value.detach().numpy()
    return np.asarray(value)


def _sorted_csr_pair(offsets, indices, values, shape):
    """Derive sorted W and W.T without rewriting or coalescing canonical edges."""
    ptr = _numpy_fixed(offsets, "w_offsets")
    col = _numpy_fixed(indices, "w_indices")
    weight = _numpy_fixed(values, "w_values")
    if (len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape)
            or ptr.dtype.kind not in "iu" or col.dtype.kind not in "iu"
            or ptr.ndim != 1 or col.ndim != 1 or weight.ndim != 1
            or weight.dtype != np.float32):
        raise ValueError("Require integer CSR indices, float32 weights, and a positive matrix shape")
    rows, columns = map(int, shape)
    if (ptr.shape != (rows + 1,) or ptr[0] != 0 or ptr[-1] != len(col)
            or len(weight) != len(col) or np.any(np.diff(ptr.astype(np.int64)) < 0)
            or not np.isfinite(weight).all()
            or (len(col) and (col.min() < 0 or col.max() >= columns))):
        raise ValueError("Malformed upstream CSR pointers, columns, or weights")
    if max(rows, columns, len(col)) >= np.iinfo(np.int32).max:
        raise ValueError("Metal CSR dimensions and edge counts must be below 2^31-1")
    row = np.repeat(np.arange(rows, dtype=np.int32), np.diff(ptr.astype(np.int64)))
    order = np.lexsort((col, row))
    forward = (ptr.astype(np.int64, copy=True), col[order].astype(np.int64, copy=True), weight[order].copy())
    _validate_cpu_csr(*forward, (rows, columns))
    # Sorting the reversed endpoints preserves each original edge and value.
    transpose_order = np.lexsort((row, col))
    transpose_ptr = np.r_[0, np.cumsum(np.bincount(col.astype(np.int64), minlength=columns))].astype(np.int64)
    transpose = (transpose_ptr, row[transpose_order].astype(np.int64), weight[transpose_order].copy())
    _validate_cpu_csr(*transpose, (columns, rows))
    return forward, transpose


class _MetalSpMM:
    @staticmethod
    def apply(state, matrix, transpose):
        """Match upstream W @ state, where state has shape [neurons,batch]."""
        if not isinstance(matrix, DenseMpsCSR) or not isinstance(transpose, DenseMpsCSR):
            raise TypeError("The scoped ngxson Metal primitive requires its fixed CSR pair")
        if state.ndim != 2 or state.shape[0] != matrix.shape[1]:
            raise ValueError("Upstream sparse state must have shape [matrix_columns,batch]")
        return FrozenSparseMM.apply(state.transpose(0, 1), matrix, transpose).transpose(0, 1).contiguous()


def _brain(model):
    brain = getattr(model, "brain", model)
    if not isinstance(model, nn.Module) or not isinstance(brain, nn.Module):
        raise TypeError("Expected the original FlyForCausalLM or FlyModel instance")
    if any(not hasattr(brain, name) for name in _CANONICAL_NAMES):
        raise TypeError("Model has no upstream Fly connectome buffers")
    return brain


def _verify_upstream(brain):
    forward = getattr(brain.forward, "__func__", None)
    connectome = getattr(brain.connectome, "__func__", None)
    module = sys.modules.get(getattr(forward, "__module__", ""))
    path = inspect.getsourcefile(forward) if forward is not None else None
    if (path is None or not Path(path).is_file()
            or hashlib.sha256(Path(path).read_bytes()).hexdigest() != MODELING_SHA256):
        raise ValueError("MPS adapter supports only modeling_fly.py from %s at %s (source SHA256 %s)"
                         % (UPSTREAM_REPOSITORY, UPSTREAM_REVISION, MODELING_SHA256))
    if (module is None or not hasattr(module, "FlyModel") or not isinstance(brain, module.FlyModel)
            or forward is not module.FlyModel.forward or connectome is not module.FlyModel.connectome
            or forward.__globals__.get("_SpMM") is not module._SpMM):
        raise ValueError("The pinned FlyModel forward/connectome bindings have been replaced")
    return forward


def _signature(brain):
    return tuple((name, value.data_ptr(), value._version, str(value.device), value.dtype, tuple(value.shape))
                 for name in _CANONICAL_NAMES for value in (getattr(brain, name),))


def _private_forward(function):
    namespace = dict(function.__globals__)
    namespace["_SpMM"] = _MetalSpMM
    cloned = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__annotations__ = dict(function.__annotations__)
    cloned.__doc__, cloned.__module__, cloned.__qualname__ = function.__doc__, function.__module__, function.__qualname__
    return cloned


class _Runtime:
    def __init__(self, forward, transpose, shape, canonical_sha256, device):
        self.forward = DenseMpsCSR(*forward, shape, device=device)
        self.transpose = DenseMpsCSR(*transpose, tuple(reversed(shape)), device=device)
        self.canonical_sha256 = canonical_sha256
        self.runtime_sha256 = (_digest(forward), _digest(transpose))
        self.signature = None
        self.original_attributes = {}
        self.padding_hook = None
        self.padding_index = None
        self.padding_weight = None

    def check(self, brain):
        if _signature(brain) != self.signature:
            raise RuntimeError("Fly connectome buffers moved or changed after enable_mps; disable the adapter and rebuild from CPU")
        if brain.wte.weight is not self.padding_weight or brain.wte.padding_idx != self.padding_index:
            raise RuntimeError("Fly input embedding replaced after enable_mps; disable the adapter and rebuild from CPU")

    def verify(self, brain):
        self.check(brain)
        if _digest([getattr(brain, name) for name in _CANONICAL_NAMES]) != self.canonical_sha256:
            raise RuntimeError("Canonical upstream connectome buffer contents changed")
        for matrix, expected in zip((self.forward, self.transpose), self.runtime_sha256):
            matrix.verify_frozen()
            if _digest([matrix.crow_indices(), matrix.col_indices(), matrix.values()]) != expected:
                raise RuntimeError("Derived Metal connectome buffer contents changed")
        return True


def _connectome(self):
    runtime = getattr(self, _RUNTIME_ATTRIBUTE)
    runtime.check(self)
    return runtime.forward, runtime.transpose


def _padding_gradient_hook(embedding):
    """Keep upstream padding_idx's zero-gradient semantics on native MPS.

    Installed on one existing Parameter, including when it is initially
    frozen. Restore requires_grad immediately; later unfreezing remains safe.
    The fixed one-element index lives on the parameter's current device.
    """
    if embedding.padding_idx is None:
        return None
    index = torch.tensor([embedding.padding_idx], dtype=torch.long, device=embedding.weight.device)
    original_requires_grad = embedding.weight.requires_grad
    try:
        embedding.weight.requires_grad_(True)
        return embedding.weight.register_hook(lambda gradient: gradient.index_fill(0, index, 0))
    finally:
        embedding.weight.requires_grad_(original_requires_grad)


def enable_mps(model, device="mps"):
    """Move a fully loaded CPU float32 upstream model to MPS and return it.

    No model class/global function is patched. Dense canonical graph buffers
    keep their original values and ordering. Fixed sorted runtime matrices are
    derived once, with no per-step CPU transfers. Repeated calls are idempotent.
    Call disable_mps before moving to CPU, replacing buffers, or changing dtype.
    """
    device = torch.device(device)
    if device.type != "mps":
        raise ValueError("enable_mps only accepts an MPS device")
    brain = _brain(model)
    if hasattr(brain, _RUNTIME_ATTRIBUTE):
        getattr(brain, _RUNTIME_ATTRIBUTE).check(brain)
        return model
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("enable_mps requires native MPS and torch.mps.compile_shader; no CPU fallback")
    forward_function = _verify_upstream(brain)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Load the original model fully on CPU before enable_mps: " + name)
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError("This Metal backend preserves float32 math only; load dtype=torch.float32: " + name)
    for name in _CANONICAL_NAMES:
        if getattr(brain, name).requires_grad:
            raise ValueError("Upstream connectome buffers must remain fixed")
    shape = (int(brain.config.n_neurons),) * 2
    canonical = [getattr(brain, name) for name in _CANONICAL_NAMES]
    canonical_sha = _digest(canonical)
    pair = _sorted_csr_pair(*canonical, shape)
    runtime = _Runtime(*pair, shape, canonical_sha, device)
    runtime.original_attributes = {name: (name in brain.__dict__, brain.__dict__.get(name))
                                   for name in ("forward", "connectome")}
    model.to(device)
    runtime.signature = _signature(brain)
    runtime.padding_index = brain.wte.padding_idx
    runtime.padding_weight = brain.wte.weight
    runtime.padding_hook = _padding_gradient_hook(brain.wte)
    # A plain runtime object avoids registering derived matrices in state_dict.
    object.__setattr__(brain, _RUNTIME_ATTRIBUTE, runtime)
    brain.connectome = types.MethodType(_connectome, brain)
    brain.forward = types.MethodType(_private_forward(forward_function), brain)
    return model


def disable_mps(model):
    """Restore original instance bindings; subsequently model.cpu() is safe."""
    brain = _brain(model)
    runtime = getattr(brain, _RUNTIME_ATTRIBUTE, None)
    if runtime is None:
        return model
    if runtime.padding_hook is not None:
        runtime.padding_hook.remove()
    for name, (existed, value) in runtime.original_attributes.items():
        if existed:
            setattr(brain, name, value)
        else:
            delattr(brain, name)
    delattr(brain, _RUNTIME_ATTRIBUTE)
    return model


def mps_metadata(model, verify=False):
    """Backend/provenance receipt; verify=True explicitly audits GPU buffers."""
    brain = _brain(model)
    runtime = getattr(brain, _RUNTIME_ATTRIBUTE, None)
    if runtime is None:
        raise ValueError("The model has no enabled ngxson MPS adapter")
    runtime.check(brain)
    verified = runtime.verify(brain) if verify else None
    return {"upstream_repository": UPSTREAM_REPOSITORY, "upstream_revision": UPSTREAM_REVISION,
            "modeling_sha256": MODELING_SHA256, "canonical_connectome_sha256": runtime.canonical_sha256,
            "shape": list(runtime.forward.shape), "edges": runtime.forward.nnz,
            "derived_csr_sha256": list(runtime.runtime_sha256), "canonical_and_runtime_buffers_verified": verified,
            "scope": "one brain instance; original forward code with private fixed SpMM primitive",
            "padding_gradient_guard": {"installed": runtime.padding_hook is not None,
                "padding_idx": runtime.padding_index, "forward_values_unchanged": True,
                "reason": "Match CPU nn.Embedding padding_idx zero-gradient semantics; native torch2.11 MPS regression reproduced"},
            "state_dict_unchanged": True, "dtype": "float32", "base_edge_gradients": False,
            "first_order_state_gradients": True, "higher_order_gradients_supported": False,
            "forward_kernel_calls": runtime.forward.kernel_launch_count,
            "backward_kernel_calls": runtime.transpose.kernel_launch_count,
            "backend": backend_metadata()}
