"""Fixed sparse CSR multiplication on Apple GPUs using a custom Metal kernel.

This is a backend primitive, not a new neuron model. It computes h @ A.T for
h[batch,columns] with FP32 values/accumulation. The existing custom autograd
wrapper supplies the explicitly transposed graph for state gradients. No edge
gradients, dense adjacency, per-step CPU fallback or per-step host transfers.

PyTorch 2.8 already provides torch.mps.compile_shader. Its supported custom
kernel dispatch interface is exercised in PyTorch's TestMetalLibrary tests.
"""
from __future__ import annotations

import hashlib
import threading

import numpy as np
import torch
from torch import nn


METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// One 32-lane Apple SIMD group owns one (batch,row). Each lane walks every
// 32nd edge; the SIMD reduction writes a single result without atomics.
kernel void fixed_csr_spmm_simd(
    device float* output,
    const device float* input,
    const device int* row_ptr,
    const device int* column,
    const device float* weight,
    const device int* shape,
    uint thread_id [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint rows = uint(shape[0]);
    const uint columns = uint(shape[1]);
    const uint output_index = thread_id / 32;
    const uint row = output_index % rows;
    const uint batch = output_index / rows;
    float partial = 0.0f;
    for (int edge = row_ptr[row] + int(lane); edge < row_ptr[row + 1]; edge += 32) {
        partial += weight[edge] * input[batch * columns + uint(column[edge])];
    }
    const float total = simd_sum(partial);
    if (lane == 0) {
        output[output_index] = total;
    }
}
"""

METAL_SOURCE_SHA256 = hashlib.sha256(METAL_SOURCE.encode()).hexdigest()
_LIBRARY = None
_COMPILE_LOCK = threading.Lock()
_KERNEL_LAUNCH_COUNT = 0


def _compiled_library():
    global _LIBRARY
    if _LIBRARY is None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Custom Metal sparse backend requires an available MPS device")
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("This PyTorch runtime lacks torch.mps.compile_shader; no CPU fallback is provided")
        with _COMPILE_LOCK:
            if _LIBRARY is None:
                _LIBRARY = torch.mps.compile_shader(METAL_SOURCE)
    return _LIBRARY


def _cpu_array(value, name):
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError(f"Fixed CSR buffer must not require gradients: {name}")
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_cpu_csr(ptr, indices, values, shape):
    if len(shape) != 2 or any(int(size) != size or size < 1 for size in shape):
        raise ValueError("CSR shape must contain two positive integer dimensions")
    rows, columns = map(int, shape)
    if ptr.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
        raise ValueError("CSR pointers and column indices must be integer arrays")
    if ptr.ndim != 1 or indices.ndim != 1 or values.ndim != 1:
        raise ValueError("CSR buffers must be one-dimensional")
    if ptr.shape != (rows + 1,) or ptr[0] != 0 or ptr[-1] != len(indices) or len(values) != len(indices):
        raise ValueError("CSR pointer/column/value lengths are inconsistent")
    if np.any(np.diff(ptr.astype(np.int64)) < 0):
        raise ValueError("CSR row pointers must be nondecreasing")
    if not np.isfinite(values).all():
        raise ValueError("CSR values must be finite")
    if len(indices) and (indices.min() < 0 or indices.max() >= columns):
        raise ValueError("CSR column index out of bounds")
    limit = np.iinfo(np.int32).max
    if max(rows, columns, len(indices)) >= limit:
        raise ValueError("Metal CSR runtime requires dimensions and edge count below2^31-1")
    if len(indices) > 1:
        same_row = np.ones(len(indices) - 1, dtype=bool)
        boundaries = ptr[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < len(indices))]
        same_row[boundaries.astype(np.int64) - 1] = False
        if np.any(same_row & (indices[1:] <= indices[:-1])):
            raise ValueError("CSR columns must be sorted and distinct inside each row; no coalescing is allowed")


class DenseMpsCSR(nn.Module):
    """CSR semantics stored as immutable dense MPS index/value buffers.

    Public int64 indices match the CPU/ROCm model fingerprint. Compact int32
    index caches are uploaded once and used by Metal. Construction performs
    CPU validation; numerical calls never copy matrix/state data to the host.
    """

    def __init__(self, ptr, indices, values, shape, device="mps"):
        super().__init__()
        if torch.device(device).type != "mps":
            raise ValueError("DenseMpsCSR is specifically an MPS backend; it has no CPU fallback")
        ptr = _cpu_array(ptr, "crow_indices")
        indices = _cpu_array(indices, "col_indices")
        values = _cpu_array(values, "values")
        _validate_cpu_csr(ptr, indices, values, shape)
        self.shape = torch.Size(tuple(map(int, shape)))
        self.nnz = int(len(indices))
        self.register_buffer("_crow", torch.tensor(ptr.astype(np.int64, copy=False), dtype=torch.int64, device=device))
        self.register_buffer("_col", torch.tensor(indices.astype(np.int64, copy=False), dtype=torch.int64, device=device))
        self.register_buffer("_values", torch.tensor(values.astype(np.float32, copy=False), dtype=torch.float32, device=device))
        self.register_buffer("_metal_crow", torch.tensor(ptr.astype(np.int32, copy=False), dtype=torch.int32, device=device))
        self.register_buffer("_metal_col", torch.tensor(indices.astype(np.int32, copy=False), dtype=torch.int32, device=device))
        self.register_buffer("_metal_shape", torch.tensor(tuple(self.shape), dtype=torch.int32, device=device))
        self.kernel_launch_count = 0

    @property
    def device(self):
        return self._values.device

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def requires_grad(self):
        return False

    def crow_indices(self):
        return self._crow

    def col_indices(self):
        return self._col

    def values(self):
        return self._values

    def _nnz(self):
        return self.nnz

    def verify_frozen(self):
        """Explicit audit only; verifies the GPU index caches used by Metal."""
        if not torch.equal(self._metal_crow.to(torch.int64), self._crow):
            raise RuntimeError("Metal CSR row-pointer cache differs from public canonical index buffer")
        if not torch.equal(self._metal_col.to(torch.int64), self._col):
            raise RuntimeError("Metal CSR column cache differs from public canonical index buffer")
        expected = torch.tensor(tuple(self.shape), dtype=torch.int32, device=self.device)
        if not torch.equal(self._metal_shape, expected):
            raise RuntimeError("Metal CSR shape cache changed")
        return True

    def metadata(self):
        return {**backend_metadata(), "shape": list(self.shape), "edges": self.nnz,
                "public_index_dtype": "int64", "kernel_index_dtype": "int32",
                "value_dtype": "float32", "matrix_kernel_launch_count": self.kernel_launch_count}


def make_mps_csr(ptr, indices, values, shape, device="mps"):
    return DenseMpsCSR(ptr, indices, values, shape, device=device)


def mps_csr_mm(h, matrix):
    """Return h @ matrix.T on GPU; autograd belongs to FrozenSparseMM.

    Noncontiguous h is copied to contiguous storage on the GPU. Every kernel
    writes each output exactly once, including empty rows. No atomics or dense
    adjacency are used. Matrix buffers must remain fixed across trajectories.
    """
    global _KERNEL_LAUNCH_COUNT
    if not isinstance(matrix, DenseMpsCSR):
        raise TypeError("mps_csr_mm requires DenseMpsCSR")
    if h.device.type != "mps" or matrix.device.type != "mps":
        raise ValueError("Metal sparse multiplication requires MPS-resident state and matrix")
    if h.dtype != torch.float32 or h.ndim != 2 or h.shape[1] != matrix.shape[1]:
        raise ValueError("Metal sparse input must be float32[batch,matrix_columns]")
    if matrix.values().requires_grad:
        raise ValueError("No base edge weight gradients are supported")
    rows = matrix.shape[0]
    output = torch.empty((h.shape[0], rows), dtype=torch.float32, device=h.device)
    if h.shape[0] == 0:
        return output
    # 32 lanes per output, four independent SIMD groups per threadgroup.
    # Dispatch metadata uses shapes already known to Python, never tensor.item.
    _compiled_library().fixed_csr_spmm_simd(
        output, h.contiguous(), matrix._metal_crow, matrix._metal_col,
        matrix._values, matrix._metal_shape,
        threads=int(h.shape[0] * rows * 32), group_size=128,
    )
    _KERNEL_LAUNCH_COUNT += 1
    matrix.kernel_launch_count += 1
    return output


def kernel_launch_count():
    return _KERNEL_LAUNCH_COUNT


def backend_metadata():
    return {
        "backend": "metal_custom_csr", "implementation_version": 1,
        "shader_compiler": "torch.mps.compile_shader", "torch_version": str(torch.__version__),
        "source_sha256": METAL_SOURCE_SHA256,
        "kernel": "fixed_csr_spmm_simd", "strategy": "one32-laneSIMDgroupper(batch,row); tree reduction; no atomics",
        "precision": "float32 weights, state and accumulation",
        "host_fallback": False, "per_step_host_transfers": False,
        "dense_adjacency": False, "edge_weight_gradients": False,
        "state_gradient": "same GPU kernel with explicitly transposed fixed CSR",
        "kernel_launch_count": _KERNEL_LAUNCH_COUNT,
    }
