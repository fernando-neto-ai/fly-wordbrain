"""Fixed-connectome rate model with small per-story synaptic state.

This deliberately replaces Doomfly's spiking physiology with smooth leaky tanh
rate dynamics. Graph endpoints, neuron identities and base synaptic signs stay
fixed. Only 20 shared plasticity-rule parameters can learn; the decoder lives
outside this module. No physical milliseconds are assigned to rate updates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from .brain import OutputProjection, array_digest
from .pair_brain import OrderedPairEncoder


class PlasticState(NamedTuple):
    h: torch.Tensor
    fast: torch.Tensor


class FrozenSparseMM(torch.autograd.Function):
    """Differentiate only neuron state through a fixed sparse matrix.

    Matrices have explicit incoming and outgoing CSR orientations. Autograd
    retains no per-edge activation products and computes no base-weight grads.
    """

    @staticmethod
    def forward(ctx, h, incoming, outgoing):
        if h.device.type == "mps":
            from .metal_sparse import mps_csr_mm
            # DenseMpsCSR is a module, not a sparse Tensor. Only h has a
            # backward result; the fixed transpose stays on the Metal device.
            ctx.mps_outgoing = outgoing
            return mps_csr_mm(h, incoming)
        ctx.save_for_backward(outgoing)
        return torch.sparse.mm(incoming, h.transpose(0, 1)).transpose(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        if hasattr(ctx, "mps_outgoing"):
            from .metal_sparse import mps_csr_mm
            return mps_csr_mm(grad_output, ctx.mps_outgoing), None, None
        (outgoing,) = ctx.saved_tensors
        grad_h = torch.sparse.mm(outgoing, grad_output.transpose(0, 1)).transpose(0, 1)
        return grad_h, None, None


def frozen_sparse_mm(h, incoming, outgoing):
    return FrozenSparseMM.apply(h, incoming, outgoing)


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _csr(ptr, columns, values, n, device):
    # Both compressed pointers and column indices must use one index dtype.
    # Sorted, distinct columns are required by validated CSR/rocSPARSE paths.
    if torch.device(device).type == "mps":
        from .metal_sparse import DenseMpsCSR
        return DenseMpsCSR(ptr, columns, values, (n, n), device=device)
    return torch.sparse_csr_tensor(
        torch.as_tensor(ptr.astype(np.int64, copy=False), device=device),
        torch.as_tensor(columns.astype(np.int64, copy=False), device=device),
        torch.as_tensor(values.astype(np.float32, copy=False), device=device),
        size=(n, n), dtype=torch.float32, device=device, check_invariants=True,
    )


def sorted_csr_columns(ptr, columns, values):
    """Stable runtime reordering only: preserve every canonical edge and value."""
    rows = np.repeat(np.arange(len(ptr) - 1, dtype=np.int32), np.diff(ptr))
    order = np.lexsort((columns, rows))
    sorted_columns = columns[order]
    if len(order) > 1:
        same_row = rows[order][1:] == rows[order][:-1]
        if np.any(same_row & (sorted_columns[1:] == sorted_columns[:-1])):
            raise ValueError("Parallel directed graph entries cannot be coalesced by this fixed-edge model")
    return sorted_columns, values[order], order


class PlasticBrain(nn.Module):
    def __init__(self, graph_path, vocab_size, global_scale, internal_steps=8,
                 leak=.5, plasticity=True, device="cpu", feature_dim=256,
                 strict_full_graph=True, input_gain=1., input_center=0.,
                 input_high=.02, seed=1729, initial_write_strength=.1,
                 max_write_strength=.5, initial_retention=.95):
        super().__init__()
        requested_device = torch.device(device)
        if requested_device.type == "mps":
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
                raise RuntimeError("Custom Metal execution requires PYTORCH_ENABLE_MPS_FALLBACK disabled")
            if not torch.backends.mps.is_available():
                raise RuntimeError("Custom Metal CSR requires an available native MPS device")
        self.sparse_backend = "metal_custom_csr" if requested_device.type == "mps" else "torch_sparse_csr"
        path = Path(graph_path)
        if path.is_dir():
            path = path / "graph.npz"
        self.graph_path = path.resolve()
        self.graph_file_sha256 = _file_hash(path)
        metadata_path = path.with_name("metadata.json")
        self.source_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        self.source_metadata_sha256 = _file_hash(metadata_path) if metadata_path.exists() else None
        if strict_full_graph and not metadata_path.exists():
            raise ValueError("Full graph mode requires adjacent metadata.json provenance")
        if not math.isfinite(global_scale) or global_scale <= 0:
            raise ValueError("global_scale must be explicit, finite and positive")
        if int(internal_steps) != internal_steps or internal_steps < 1 or not 0 < leak <= 1:
            raise ValueError("Require positive integer internal_steps and leak within (0,1]")
        if not all(math.isfinite(float(v)) for v in (input_gain, input_center, input_high)) or input_high <= 0:
            raise ValueError("Input current settings must be finite and input_high positive")
        if not 0 < abs(initial_write_strength) < max_write_strength or not 0 < initial_retention < 1:
            raise ValueError("Require nonzero bounded initial writing and retention within (0,1)")
        self.global_scale, self.leak = float(global_scale), float(leak)
        self.internal_steps, self.feature_dim = int(internal_steps), int(feature_dim)
        self.plasticity_enabled, self.strict_full_graph = bool(plasticity), bool(strict_full_graph)
        self.input_gain, self.input_center = float(input_gain), float(input_center)
        self.input_high, self.seed = float(input_high), int(seed)
        self.max_write_strength = float(max_write_strength)
        self.vocab_size = int(vocab_size)
        with np.load(path, allow_pickle=False) as source:
            required = ("ids", "ptr", "post", "weight", "incoming_ptr", "incoming_pre",
                        "incoming_weight", "incoming_edge_ids", "retina", "descending",
                        "candidate_edge_ids", "candidate_pre", "candidate_post", "candidate_group")
            arrays = {key: source[key].copy() for key in required}
        self.neurons, self.edges = len(arrays["ids"]), len(arrays["weight"])
        self.candidates = len(arrays["candidate_edge_ids"])
        if strict_full_graph and (self.neurons != 166700 or self.edges != 25582938 or len(arrays["descending"]) != 1314):
            raise ValueError("Expected all166700neurons/25582938edges/1314descending outputs")
        self._validate_graph(arrays)
        # The canonical outgoing CSR has unsorted columns. Retain its raw
        # arrays/edge IDs, but give the sparse backward kernel a sorted runtime
        # transpose. Sorting changes neither graph endpoints nor values.
        sorted_post, sorted_weight, outgoing_order = sorted_csr_columns(
            arrays["ptr"], arrays["post"], arrays["weight"])
        self.canonical_outgoing_sha256 = array_digest([arrays["ptr"], arrays["post"], arrays["weight"]])
        self.runtime_outgoing_order_sha256 = array_digest([outgoing_order])
        for name, key in (("raw_outgoing_ptr", "ptr"), ("raw_outgoing_post", "post"), ("raw_outgoing_weight", "weight")):
            self.register_buffer(name, torch.as_tensor(arrays[key], device=device))
        incoming = _csr(arrays["incoming_ptr"], arrays["incoming_pre"], arrays["incoming_weight"], self.neurons, device)
        outgoing = _csr(arrays["ptr"], sorted_post, sorted_weight, self.neurons, device)
        if requested_device.type == "mps":
            # MPS has no sparse CSR Tensor storage. Child modules hold dense
            # compressed arrays consumed directly by the Metal kernels.
            self.incoming, self.outgoing = incoming, outgoing
        else:
            self.register_buffer("incoming", incoming)
            self.register_buffer("outgoing", outgoing)
        for name, source_name in (("neuron_ids", "ids"), ("retina", "retina"),
                                  ("descending", "descending"), ("candidate_edge_ids", "candidate_edge_ids"),
                                  ("candidate_pre", "candidate_pre"), ("candidate_post", "candidate_post"),
                                  ("candidate_group", "candidate_group")):
            self.register_buffer(name, torch.as_tensor(arrays[source_name].astype(np.int64, copy=False), device=device))
        self.register_buffer("candidate_weight", torch.as_tensor(arrays["weight"][arrays["candidate_edge_ids"]],
                                                                   dtype=torch.float32, device=device))
        encoder = OrderedPairEncoder(vocab_size, len(arrays["retina"]), seed, input_high)
        self.encoder_metadata = encoder.metadata()
        for name, values in (("previous_codes", encoder.previous_codes), ("current_codes", encoder.current_codes)):
            self.register_buffer(name, torch.tensor(values.copy(), dtype=torch.float32, device=device))
        self.register_buffer("previous_neurons", self.retina[torch.as_tensor(encoder.previous_bank.astype(np.int64), device=device)])
        self.register_buffer("current_neurons", self.retina[torch.as_tensor(encoder.current_bank.astype(np.int64), device=device)])
        projection = OutputProjection(arrays["descending"], bins=feature_dim, seed=741)
        self.projection_sha256 = projection.sha256
        self.register_buffer("readout_bucket", torch.tensor(projection.bucket, dtype=torch.long, device=device))
        self.register_buffer("readout_sign", torch.tensor(projection.sign, dtype=torch.float32, device=device))
        self.register_buffer("readout_scale", torch.tensor(projection.scale, dtype=torch.float32, device=device))
        self.register_buffer("pre_scale", torch.ones(self.candidates, dtype=torch.float32, device=device))
        self.register_buffer("post_scale", torch.ones(self.candidates, dtype=torch.float32, device=device))
        self.write_strength = nn.Parameter(torch.full((4,), math.atanh(initial_write_strength / max_write_strength), device=device))
        self.retention_logit = nn.Parameter(torch.full((4,), math.log(initial_retention / (1 - initial_retention)), device=device))
        self.gate_bias = nn.Parameter(torch.full((4,), -1., device=device))
        self.gate_pre = nn.Parameter(torch.full((4,), .1, device=device))
        self.gate_post = nn.Parameter(torch.full((4,), .1, device=device))
        if not plasticity:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        self.initial_graph_fingerprint = self.frozen_fingerprint()
        self.initial_interface_fingerprint = self.interface_fingerprint()

    def _validate_graph(self, a):
        n, e, k = self.neurons, self.edges, self.candidates
        if n < 1 or e < 1 or k < 1:
            raise ValueError("Graph and selected plasticity edge set must be nonempty")
        for name in ("ptr", "incoming_ptr"):
            ptr = a[name]
            if ptr.shape != (n + 1,) or ptr[0] != 0 or ptr[-1] != e or np.any(np.diff(ptr) < 0):
                raise ValueError(f"Invalid CSR row pointer: {name}")
        for name in ("post", "incoming_pre", "incoming_weight", "incoming_edge_ids"):
            if a[name].shape != (e,):
                raise ValueError(f"Wrong edge-array shape: {name}")
        if not np.isfinite(a["weight"]).all() or not np.isfinite(a["incoming_weight"]).all():
            raise ValueError("Base weights must be finite")
        for name in ("post", "incoming_pre", "candidate_pre", "candidate_post", "retina", "descending"):
            if np.any(a[name] < 0) or np.any(a[name] >= n):
                raise ValueError(f"Neuron index out of bounds: {name}")
        for name in ("retina", "descending"):
            if len(np.unique(a[name])) != len(a[name]) or len(a[name]) == 0:
                raise ValueError(f"Empty/duplicate population membership: {name}")
        incoming_ids = a["incoming_edge_ids"]
        if incoming_ids.min() < 0 or incoming_ids.max() >= e or not np.all(np.bincount(incoming_ids, minlength=e) == 1):
            raise ValueError("Incoming edge mapping is not a permutation of all original edges")
        outgoing_pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(a["ptr"]))
        incoming_post = np.repeat(np.arange(n, dtype=np.int32), np.diff(a["incoming_ptr"]))
        if (not np.array_equal(a["incoming_pre"], outgoing_pre[incoming_ids])
                or not np.array_equal(incoming_post, a["post"][incoming_ids])
                or not np.array_equal(a["incoming_weight"], a["weight"][incoming_ids])):
            raise ValueError("Incoming CSR differs from original outgoing graph orientation/weights")
        selected = a["candidate_edge_ids"]
        if (selected.shape != (k,) or len(np.unique(selected)) != k or selected.min() < 0 or selected.max() >= e
                or any(a[name].shape != (k,) for name in ("candidate_pre", "candidate_post", "candidate_group"))):
            raise ValueError("Invalid candidate edge selection")
        if (not np.array_equal(a["candidate_pre"], outgoing_pre[selected])
                or not np.array_equal(a["candidate_post"], a["post"][selected])
                or np.any(a["candidate_group"] < 0) or np.any(a["candidate_group"] >= 4)):
            raise ValueError("Candidate identities/endpoints/groups differ from original graph")

    @property
    def device(self):
        return self.candidate_weight.device

    def initial_state(self, batch):
        if int(batch) != batch or batch < 1:
            raise ValueError("Batch size must be a positive integer")
        return PlasticState(torch.zeros((int(batch), self.neurons), device=self.device),
                            torch.zeros((int(batch), self.candidates), device=self.device))

    @staticmethod
    def reset_activity(state):
        return PlasticState(torch.zeros_like(state.h), state.fast)

    @staticmethod
    def reset_fast(state):
        return PlasticState(state.h, torch.zeros_like(state.fast))

    @staticmethod
    def detach_state(state):
        return PlasticState(state.h.detach(), state.fast.detach())

    def set_activity_scales(self, pre, post):
        """Install positive training-only scales; length K takes precedence over groups."""
        for name, values in (("pre_scale", pre), ("post_scale", post)):
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device).detach()
            if values.ndim == 0:
                values = values.expand(self.candidates)
            elif values.shape == (4,) and self.candidates != 4:
                values = values[self.candidate_group]
            if values.shape != (self.candidates,) or not torch.isfinite(values).all() or not torch.all(values > 0):
                raise ValueError("Activity scales must be finite positive scalar, per-edgeK, or four-group values")
            getattr(self, name).copy_(values)

    def retinal_current(self, previous_ids, current_ids):
        previous = torch.as_tensor(previous_ids, dtype=torch.long, device=self.device)
        current = torch.as_tensor(current_ids, dtype=torch.long, device=self.device)
        if previous.ndim != 1 or current.shape != previous.shape:
            raise ValueError("Word input IDs must be equally shaped one-dimensional batches")
        if torch.any(previous < 0) or torch.any(previous >= self.vocab_size) or torch.any(current < 0) or torch.any(current >= self.vocab_size):
            raise ValueError("Word input ID out of vocabulary")
        result = torch.zeros((len(previous), self.neurons), dtype=torch.float32, device=self.device)
        result[:, self.previous_neurons] = self.input_gain * (self.previous_codes[previous] - self.input_center)
        result[:, self.current_neurons] = self.input_gain * (self.current_codes[current] - self.input_center)
        return result

    def effective_candidate_weights(self, fast):
        return self.candidate_weight * torch.exp(math.log(2.) * torch.tanh(fast))

    def readout(self, h):
        if h.ndim != 2 or h.shape[1] != self.neurons:
            raise ValueError("Readout requires [batch,neurons] activity")
        values = h[:, self.descending] * self.readout_sign
        features = torch.zeros((h.shape[0], self.feature_dim), dtype=h.dtype, device=h.device)
        return features.index_add(1, self.readout_bucket, values) / self.readout_scale

    def step(self, previous_ids, current_ids, state, plasticity_override=None):
        current = self.retinal_current(previous_ids, current_ids)
        batch = len(current)
        if state.h.shape != (batch, self.neurons) or state.fast.shape != (batch, self.candidates):
            raise ValueError("State shapes differ from input batch or graph")
        enabled = self.plasticity_enabled if plasticity_override is None else bool(plasticity_override)
        h, fast = state
        group = self.candidate_group
        if enabled:
            eta = self.max_write_strength * torch.tanh(self.write_strength[group]) / self.internal_steps
            retention = torch.sigmoid(self.retention_logit[group]).pow(1. / self.internal_steps)
        for _ in range(self.internal_steps):
            recurrent = frozen_sparse_mm(h, self.incoming, self.outgoing)
            if enabled:
                correction = h[:, self.candidate_pre] * (self.effective_candidate_weights(fast) - self.candidate_weight)
                recurrent = recurrent.index_add(1, self.candidate_post, correction)
            new_h = (1 - self.leak) * h + self.leak * torch.tanh(self.global_scale * recurrent + current)
            if enabled:
                norm_pre = torch.tanh(h[:, self.candidate_pre] / self.pre_scale)
                norm_post = torch.tanh(new_h[:, self.candidate_post] / self.post_scale)
                gate = torch.sigmoid(self.gate_bias[group] + self.gate_pre[group] * norm_pre
                                     + self.gate_post[group] * norm_post)
                fast = retention * fast + eta * gate * norm_pre * norm_post
            h = new_h
        return self.readout(h), PlasticState(h, fast)

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def frozen_fingerprint(self):
        arrays = [self.neuron_ids, self.outgoing.crow_indices(), self.outgoing.col_indices(), self.outgoing.values(),
                  self.raw_outgoing_ptr, self.raw_outgoing_post, self.raw_outgoing_weight,
                  self.incoming.crow_indices(), self.incoming.col_indices(), self.incoming.values(),
                  self.candidate_edge_ids, self.candidate_pre, self.candidate_post, self.candidate_group,
                  self.candidate_weight, self.retina, self.descending]
        return array_digest([value.detach().cpu().numpy() for value in arrays])

    def verify_frozen(self):
        if self.sparse_backend == "metal_custom_csr":
            self.incoming.verify_frozen()
            self.outgoing.verify_frozen()
        value = self.frozen_fingerprint()
        if value != self.initial_graph_fingerprint:
            raise RuntimeError("Fixed connectome arrays changed")
        if self.interface_fingerprint() != self.initial_interface_fingerprint:
            raise RuntimeError("Fixed sensory/readout interface arrays changed")
        return value

    def interface_fingerprint(self):
        arrays = [self.previous_codes, self.current_codes, self.previous_neurons, self.current_neurons,
                  self.readout_bucket, self.readout_sign, self.readout_scale]
        return array_digest([value.detach().cpu().numpy() for value in arrays])

    def metadata(self):
        return {
            "model": "full_connectome_smooth_rate_with_optional_fast_synaptic_state",
            "physiology": "Leaky tanh rate update; explicitly differs from Doomfly spiking/LIF dynamics",
            "time_units": "Dimensionless internal rate updates; no physical milliseconds assigned",
            "neurons": self.neurons, "edges": self.edges, "candidate_edges": self.candidates,
            "candidate_groups": ["hDeltaB->hDeltaH", "hDeltaB->hDeltaA", "hDeltaB->hDeltaI", "hDeltaB->hDeltaG"],
            "strict_full_graph": self.strict_full_graph, "graph_file_sha256": self.graph_file_sha256,
            "frozen_graph_sha256": self.initial_graph_fingerprint,
            "canonical_outgoing_sha256": self.canonical_outgoing_sha256,
            "sparse_runtime": {"backend": self.sparse_backend,
                               "index_dtype": "int64 for both public crow and col", "invariant_checks": True,
                               "metal_index_cache": "int32 exact copies, verified at explicit frozen audits" if self.sparse_backend == "metal_custom_csr" else None,
                               "metal_source_sha256": _file_hash(Path(__file__).with_name("metal_sparse.py")) if self.sparse_backend == "metal_custom_csr" else None,
                               "outgoing_columns": "stable sorted runtime transpose; canonical raw edge IDs retained",
                               "outgoing_edge_order_sha256": self.runtime_outgoing_order_sha256},
            "fixed_interface_sha256": self.initial_interface_fingerprint,
            "source_metadata_sha256": self.source_metadata_sha256,
            "source_metadata": self.source_metadata,
            "global_scale": self.global_scale, "internal_steps": self.internal_steps, "leak": self.leak,
            "input_gain": self.input_gain, "input_center_on_retina": self.input_center,
            "input_high": self.input_high, "input_encoding": self.encoder_metadata,
            "feature_dimensions": self.feature_dim, "readout_neurons": len(self.descending),
            "readout": "Fixed signed pooling of every descending neuron's current rate",
            "readout_projection_sha256": self.projection_sha256,
            "plasticity": self.plasticity_enabled, "trainable_rule_parameters": self.trainable_parameter_count(),
            "gain": "exp(log(2)*tanh(fast)); strictly positive in[0.5,2]; existing endpoints/signs preserved",
            "write": "H'=sigmoid(retention_logit)^(1/internal_steps)*H + eta/internal_steps*sigmoid(g0+gpre*tanh(pre/scale)+gpost*tanh(post/scale))*tanh(pre/scale)*tanh(post/scale)",
            "max_absolute_write_strength": self.max_write_strength,
            "activity_scales_sha256": array_digest([self.pre_scale.detach().cpu().numpy(), self.post_scale.detach().cpu().numpy()]),
            "override_off": "Disables both fast-weight reading and writing; supplied fast state is retained unchanged",
            "autograd": "Fixed incoming CSR forward and explicit outgoing CSR backward; no base edge weight gradients or edge-by-batch-by-time activations",
        }
