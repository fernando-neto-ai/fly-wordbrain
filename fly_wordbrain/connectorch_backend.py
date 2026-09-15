"""Registered ConnecTorch backend using our native trainable Apple CSR kernels.

Call register_connectorch_backend(), then construct ConnectomeRNN with
backend='metal_csr'. CPU is an explicit correctness reference; MPS never falls
back to CPU. ConnectorchCSR provides the same propagation backend to the exact
ngxson dynamics, which have neuron gains absent from stock ConnectomeRNN.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

import connectorch
from connectorch.backends.base import BACKENDS, Propagator, build_propagator

from .metal_sparse_trainable import TrainableCSR

ROOT = Path(__file__).resolve().parents[1]


def connectorch_receipt():
    lock = json.loads((ROOT / "connectorch-source.json").read_text())
    package = Path(connectorch.__file__).resolve().parent
    observed = {}
    for name, expected in lock["python_files"].items():
        path = package / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Installed Connectorch differs from pinned source: " + name)
        observed[name] = actual
    return {"repository": lock["repository"], "revision": lock["revision"],
            "package_path": str(package), "python_files_sha256": observed}


class MetalCSRPropagator(Propagator):
    """ConnecTorch's node-major propagator contract with O(E) edge backward."""
    backend_name = "metal_csr"
    supports_sparse_backward = True

    def __init__(self, edge_index, num_nodes):
        super().__init__(edge_index, num_nodes)
        self.register_buffer("edge_index", edge_index.detach().clone(), persistent=False)
        self._runtime = None
        self._signature = None
        self._validate_edges()
        self.register_load_state_dict_post_hook(_clear_loaded_layout)

    def _validate_edges(self):
        e = self.edge_index.detach().cpu().numpy()
        if e.dtype.kind not in "iu" or (e.size and (e.min() < 0 or e.max() >= self.num_nodes)):
            raise ValueError("Invalid ConnecTorch endpoint dtype or range")
        # Connectorch canonical order is target then source. Fixed target order
        # is necessary for the supplied edge_weight vector to be CSR-aligned.
        if e.shape[1] > 1 and np.any(np.diff(e[1]) < 0):
            raise ValueError("ConnecTorch edge_index must be target-sorted")

    def _apply(self, fn, recurse=True):
        self._runtime = None
        self._signature = None
        return super()._apply(fn, recurse=recurse)

    def rebuild(self, edge_index):
        device = self.edge_index.device
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Expected edge_index[2,E]")
        self.edge_index = edge_index.detach().clone().to(device)
        self.num_edges = int(edge_index.shape[1])
        self._runtime = None
        self._signature = None
        self._validate_edges()

    def _get_runtime(self):
        signature = (id(self.edge_index), self.edge_index._version, self.edge_index.device,
                     self.num_nodes, self.num_edges)
        if self._runtime is not None and signature != self._signature:
            raise RuntimeError("ConnecTorch topology changed without rebuilding the propagator")
        if self._runtime is None:
            self._validate_edges()
            edges = self.edge_index.detach().cpu().numpy()
            ptr = np.r_[0, np.cumsum(np.bincount(edges[1], minlength=self.num_nodes))]
            self._runtime = TrainableCSR(ptr, edges[0], (self.num_nodes, self.num_nodes),
                                         device=self.edge_index.device)
            self._signature = signature
        return self._runtime

    def forward(self, h, edge_weight):
        if h.ndim != 2 or h.shape[0] != self.num_nodes:
            raise ValueError("ConnecTorch state must be [nodes,batch]")
        return self._get_runtime().mm(h.T, edge_weight).T.contiguous()

    def verify_frozen(self):
        runtime = self._get_runtime()
        current = self.edge_index.detach().cpu().numpy()
        rows = runtime._buffers["row"].detach().cpu().numpy()
        columns = runtime._buffers["col"].detach().cpu().numpy()
        if not np.array_equal(current[0], columns) or not np.array_equal(current[1], rows):
            raise RuntimeError("Propagator endpoints differ from the execution layout")
        return runtime.verify_frozen()

    def metadata(self):
        return {**self._get_runtime().metadata(), "connectorch_backend": self.backend_name}


def register_connectorch_backend():
    """Explicit opt-in registration; existing backend choices remain unchanged."""
    receipt = connectorch_receipt()
    previous = BACKENDS.get("metal_csr")
    if previous is not None and previous is not MetalCSRPropagator:
        raise RuntimeError("Another extension already registered metal_csr")
    BACKENDS["metal_csr"] = MetalCSRPropagator
    return receipt


def _clear_loaded_layout(module, incompatible_keys):
    module._runtime = None
    module._signature = None
    module.num_edges = int(module.edge_index.shape[1])
    module._validate_edges()


class ConnectorchCSR:
    """Batch-major bridge to ConnecTorch's registered propagation backend."""
    def __init__(self, ptr, indices, shape, device="cpu"):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Connectome propagation requires a square graph")
        register_connectorch_backend()
        ptr = np.asarray(torch.as_tensor(ptr).cpu())
        indices = np.asarray(torch.as_tensor(indices).cpu())
        if ptr.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
            raise ValueError("CSR topology arrays must contain integers")
        ptr, indices = ptr.astype(np.int64), indices.astype(np.int64)
        if (ptr.shape != (shape[0] + 1,) or ptr[0] != 0 or ptr[-1] != len(indices)
                or np.any(np.diff(ptr) < 0)):
            raise ValueError("Malformed canonical CSR pointers")
        rows = np.repeat(np.arange(shape[0], dtype=np.int64), np.diff(ptr))
        edge_index = torch.from_numpy(np.stack([indices, rows]))
        self.propagator = build_propagator("metal_csr", edge_index, int(shape[0]), trainable=True).to(device)
        self.propagator._get_runtime()  # Upload layouts once before timed work.
        self.shape = torch.Size(shape)
        self.nnz = len(indices)

    @property
    def device(self):
        return self.propagator.edge_index.device

    def mm(self, h, values):
        return self.propagator(h.T, values).T.contiguous()

    def verify_frozen(self):
        return self.propagator.verify_frozen()

    def metadata(self):
        return self.propagator.metadata()


def exact_connectome_ir(checkpoint, node_ids, node_type_labels=None):
    """Import exact checkpoint weights; never invoke dataset/sign defaults."""
    def array(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    ptr, col, weight = (array(checkpoint[k]) for k in ("brain.w_offsets", "brain.w_indices", "brain.w_values"))
    node_ids = np.asarray(node_ids)
    n = len(node_ids)
    if ptr.shape != (n + 1,) or np.any(np.diff(node_ids) <= 0):
        raise ValueError("Require sorted unique biological IDs aligned with checkpoint rows")
    row = np.repeat(np.arange(n), np.diff(ptr))
    nodes = {"node_id": node_ids}
    if node_type_labels is not None:
        nodes["cell_type"] = np.asarray(node_type_labels)
    graph = connectorch.Connectome(nodes=nodes,
        edges={"source": node_ids[col], "target": node_ids[row], "weight": weight},
        aggregate_parallel_edges=False,
        provenance={"source": "ngxson/fly-llm-hf frozen CSR", "normalization": "none; exact checkpoint values"})
    if (graph.num_nodes != n or graph.num_edges != len(col)
            or not np.array_equal(graph.node_ids, node_ids)
            or not np.array_equal(graph.edge_index[0], col)
            or not np.array_equal(graph.edge_index[1], row)
            or not np.array_equal(graph.edge_attribute("weight"), weight)):
        raise ValueError("ConnecTorch IR import changed canonical topology, ordering or values")
    return graph
