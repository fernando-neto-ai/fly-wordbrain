"""Reference-shaped fly LM with smaller interfaces and bounded edge adaptation.

Canonical incoming CSR buffers retain the released edge order, signs and zeros.
Only derived sparse execution caches are device specific. The primary experiment
keeps the reference's unconstrained neuron gain and rec_gain parameters: a 10%
bound on new edge multipliers is *not* a bound on effective recurrent dynamics.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from types import SimpleNamespace
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


GRAPH_NAMES = ("w_offsets", "w_indices", "w_values", "in_index", "out_index", "edge_group_index", "node_type_index")


def _hash_tensors(tensors):
    digest = hashlib.sha256()
    for value in tensors:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _signature(brain):
    return tuple((name, value.data_ptr(), value._version, str(value.device), value.dtype, tuple(value.shape))
                 for name in GRAPH_NAMES for value in (getattr(brain, name),))


def _stats(value):
    value = value.detach().cpu().float()
    if not value.numel():
        return {"count": 0, "min": None, "max": None, "mean": None, "rms": None, "negative_fraction": None}
    return {"count": value.numel(), "min": value.min().item(), "max": value.max().item(),
            "mean": value.mean().item(), "rms": value.square().mean().sqrt().item(),
            "negative_fraction": (value < 0).float().mean().item()}


@dataclass
class FlyCache:
    state: torch.Tensor
    last_tokens: torch.Tensor
    seq_len: int = 0
    is_compileable = False

    def get_seq_length(self, layer_idx=0):
        return self.seq_len

    def reorder_cache(self, beam_idx):
        self.state = self.state.index_select(0, beam_idx.to(self.state.device))
        self.last_tokens = self.last_tokens.index_select(0, beam_idx.to(self.last_tokens.device))


class ConnectorchBrain(nn.Module):
    def __init__(self, reference, config, groups, node_types, csr_factory=None):
        super().__init__()
        self.config = config
        self.n_in_slot = config.n_in // config.delay_k
        self.wte = nn.Embedding(config.vocab_size, config.d_embed, padding_idx=config.pad_token_id)
        self.in_proj = nn.Parameter(torch.empty(config.delay_k, config.d_embed, self.n_in_slot))
        self.gain = nn.Parameter(torch.ones(config.n_neurons))
        self.rec_gain = nn.Parameter(torch.ones(config.n_neurons))
        self.bias = nn.Parameter(torch.zeros(config.n_neurons))
        for name in GRAPH_NAMES[:-2]:
            self.register_buffer(name, getattr(reference, name).detach().cpu().clone())
        self.register_buffer("edge_group_index", groups.clone())
        self.register_buffer("node_type_index", node_types.clone())
        labels = node_types if node_types.numel() else groups
        self.group_count = int(labels.max()) + 1 if labels.numel() else 0
        adaptive = config.plasticity == "bounded10"
        factorized = config.edge_parameterization == "factorized_cell_types"
        self.edge_theta = nn.Parameter(torch.zeros(self.group_count)) if adaptive and not factorized else None
        self.edge_theta_source = nn.Parameter(torch.zeros(self.group_count)) if adaptive and factorized else None
        self.edge_theta_destination = nn.Parameter(torch.zeros(self.group_count)) if adaptive and factorized else None
        # The leak is the fraction of state replaced each step, one scalar for all 49,393
        # neurons in the pinned model. At 0.9 nothing survives ~26 steps, which is shorter
        # than the 32-token chunk language is scored over. Trainable, each neuron picks its
        # own time constant. Parameterised as a per-neuron offset from the fixed value in
        # logit space, so it starts EXACTLY as the fixed arm and weight decay pulls it back
        # toward the fixed arm rather than toward leak = 0.5. None unless the config asks,
        # so every existing checkpoint keeps its exact parameter set and restores unchanged.
        self.leak_delta = (nn.Parameter(torch.zeros(config.n_neurons))
                           if getattr(config, "leak_mode", "fixed") == "trainable" else None)
        self._edge_targets = None
        self._edge_targets_signature = None
        self._edge_targets_sha256 = None
        self.lag_map = tuple(config.lag_map)
        self._csr_factory = csr_factory
        self._csr = None
        self._canonical_sha256 = _hash_tensors(getattr(self, name) for name in GRAPH_NAMES)
        self._canonical_signature = _signature(self)
        # Torch 2.11 MPS embedding backward does not suppress padding_idx.
        # Applying the same zero-gradient hook on CPU is harmless and retains
        # ordinary AdamW weight decay (including a nonzero loaded PAD vector).
        self._padding_hook = self.wte.weight.register_hook(self._zero_padding_gradient)

    def _zero_padding_gradient(self, gradient):
        if self.wte.padding_idx is None:
            return gradient
        index = torch.tensor([self.wte.padding_idx], dtype=torch.long, device=gradient.device)
        return gradient.index_fill(0, index, 0)

    def _check_frozen(self):
        if _signature(self) != self._canonical_signature:
            raise RuntimeError("Canonical connectome or group buffers changed; construct a new model for a new graph")
        if self._edge_targets is not None:
            targets = self._edge_targets
            signature = (targets.data_ptr(), targets._version, str(targets.device), targets.dtype, tuple(targets.shape))
            if signature != self._edge_targets_signature:
                raise RuntimeError("Derived canonical edge-target cache changed")

    def effective_leak(self):
        """The per-step replacement fraction, in the form the recurrence should use.

        Fixed: the config scalar itself, a python float, so the update is byte-identical to
        the pinned model. Trainable: sigmoid(logit(leak) + delta) per neuron, shaped to
        broadcast over the batch axis of the neuron-major state.
        """
        if self.leak_delta is None:
            return self.config.leak
        base = math.log(self.config.leak / (1.0 - self.config.leak))
        return torch.sigmoid(base + self.leak_delta)[:, None]

    def verify_frozen(self):
        self._check_frozen()
        if _hash_tensors(getattr(self, name) for name in GRAPH_NAMES) != self._canonical_sha256:
            raise RuntimeError("Canonical connectome or group buffer contents changed")
        if self._csr is not None:
            self._csr.verify_frozen()
        if self._edge_targets is not None and _hash_tensors([self._edge_targets]) != self._edge_targets_sha256:
            raise RuntimeError("Derived canonical edge-target cache contents changed")
        return True

    def _apply(self, fn, recurse=True):
        # Device transfers are permitted; graph edits and reduced precision are
        # rejected. Runtime CSR caches are rebuilt on their destination device.
        self.verify_frozen()
        probe = fn(torch.empty(0, dtype=self.w_values.dtype, device=self.w_values.device))
        if probe.dtype != torch.float32 or probe.device.type not in ("cpu", "mps"):
            raise ValueError("Connectorch fly backend supports float32 CPU or native MPS only")
        self._csr = None
        self._edge_targets = None
        result = super()._apply(fn, recurse=recurse)
        self._canonical_signature = _signature(self)
        return result

    def sparse_runtime(self):
        self._check_frozen()
        if self._csr is None:
            factory = self._csr_factory
            if factory is None:
                from .connectorch_backend import ConnectorchCSR
                factory = ConnectorchCSR
            self._csr = factory(self.w_offsets.detach().cpu(), self.w_indices.detach().cpu(),
                                (self.config.n_neurons, self.config.n_neurons), device=self.w_values.device)
        return self._csr

    def edge_multipliers(self):
        if self.config.plasticity == "fixed":
            return torch.ones_like(self.w_values)
        if self.edge_theta_source is not None:
            if self._edge_targets is None:
                counts = (self.w_offsets[1:] - self.w_offsets[:-1]).detach().cpu().long()
                targets = torch.repeat_interleave(torch.arange(self.config.n_neurons, dtype=torch.int32), counts)
                self._edge_targets_sha256 = _hash_tensors([targets])
                self._edge_targets = targets.to(self.w_values.device)
                targets = self._edge_targets
                self._edge_targets_signature = (targets.data_ptr(), targets._version, str(targets.device), targets.dtype, tuple(targets.shape))
            source = self.edge_theta_source[self.node_type_index]
            destination = self.edge_theta_destination[self.node_type_index]
            theta = source[self.w_indices] + destination[self._edge_targets]
        else:
            theta = self.edge_theta[self.edge_group_index]
        return 1 + .1 * torch.tanh(theta)

    def effective_values(self):
        if self.config.plasticity == "fixed":
            return self.w_values
        return self.w_values * self.edge_multipliers()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                cache_params=None, use_cache=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        cfg = self.config
        use_cache = cfg.use_cache if use_cache is None else use_cache
        return_dict = getattr(cfg, "return_dict", True) if return_dict is None else return_dict
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        batch, length, _ = inputs_embeds.shape
        if length < 1:
            raise ValueError("At least one input token is required")
        k, neurons = cfg.delay_k, cfg.n_neurons
        if cache_params is None:
            state = inputs_embeds.new_zeros(batch, neurons)
            prev_ids = torch.full((batch, k), cfg.pad_token_id, dtype=torch.long, device=inputs_embeds.device)
            seq_len = 0
        else:
            state, prev_ids, seq_len = cache_params.state, cache_params.last_tokens, cache_params.seq_len
        ext = torch.cat([self.wte(prev_ids), inputs_embeds], dim=1)
        drive = torch.cat([ext[:, k - lag:k - lag + length] @ self.in_proj[j]
                           for j, lag in enumerate(self.lag_map)], dim=-1)
        runtime, values = self.sparse_runtime(), self.effective_values()
        # Retain the original neuron-major order and input addition semantics.
        state = state.t().contiguous()
        gain, bias, rec_gain = self.gain[:, None], self.bias[:, None], self.rec_gain[:, None]
        mask = attention_mask.to(state.dtype).t() if attention_mask is not None else None
        drive = drive.permute(1, 2, 0).contiguous()
        outs = []
        leak = self.effective_leak()
        for time in range(length):
            recurrent = runtime.mm(state.t().contiguous(), values).t().contiguous()
            pre = (rec_gain * recurrent).index_add(0, self.in_index, drive[time])
            new = (1 - leak) * state + leak * torch.tanh(gain * pre + bias)
            if mask is not None:
                m = mask[time][None, :]
                new = m * new + (1 - m) * state
            state = new
            outs.append(state[self.out_index])
        hidden = torch.stack(outs, dim=0).permute(2, 0, 1)
        state = state.t().contiguous()
        cache = None
        if use_cache:
            ids = torch.cat([prev_ids, input_ids], dim=1)[:, -k:] if input_ids is not None else prev_ids
            cache = FlyCache(state, ids, seq_len + length)
        hidden_states = (hidden,) if output_hidden_states else None
        if not return_dict:
            return tuple(v for v in (hidden, cache, hidden_states) if v is not None)
        return SimpleNamespace(last_hidden_state=hidden, cache_params=cache, hidden_states=hidden_states)


class ConnectorchFlyForCausalLM(nn.Module):
    def __init__(self, reference, config, groups, node_types, csr_factory=None):
        super().__init__()
        self.config = config
        self.brain = ConnectorchBrain(reference.brain, config, groups, node_types, csr_factory)
        self.ln = nn.LayerNorm(config.n_out, eps=reference.ln.eps)
        self.lm_head = (nn.Linear(config.n_out, config.vocab_size, bias=False) if not config.readout_rank else
                        nn.Sequential(nn.Linear(config.n_out, config.readout_rank, bias=False),
                                      nn.Linear(config.readout_rank, config.vocab_size, bias=False)))

    def get_input_embeddings(self):
        return self.brain.wte

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                cache_params=None, labels=None, use_cache=None, output_hidden_states=None,
                return_dict=None, logits_to_keep=0, **kwargs):
        return_dict = getattr(self.config, "return_dict", True) if return_dict is None else return_dict
        out = self.brain(input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                         cache_params=cache_params, use_cache=use_cache,
                         output_hidden_states=output_hidden_states, return_dict=True)
        sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(self.ln(out.last_hidden_state[:, sl]))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                   labels[:, 1:].reshape(-1).to(logits.device), ignore_index=-100)
        if not return_dict:
            return tuple(v for v in (loss, logits, out.cache_params, out.hidden_states) if v is not None)
        return SimpleNamespace(loss=loss, logits=logits, cache_params=out.cache_params, hidden_states=out.hidden_states)

    def verify_frozen(self):
        return self.brain.verify_frozen()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if assign:
            raise ValueError("assign=True would replace guarded parameters; use ordinary checkpoint loading")
        self.verify_frozen()
        for name in GRAPH_NAMES:
            key = "brain." + name
            if key in state_dict and not torch.equal(state_dict[key].detach().cpu(), getattr(self.brain, name).detach().cpu()):
                raise ValueError("Checkpoint would replace canonical buffer: " + key)
        result = super().load_state_dict(state_dict, strict=strict, assign=False)
        self.brain._canonical_signature = _signature(self.brain)
        self.brain._csr = None
        self.brain._edge_targets = None
        return result

    def parameter_counts(self):
        return {"embedding": self.brain.wte.weight.numel(), "input_projection": self.brain.in_proj.numel(),
                "neuron_parameters": sum(getattr(self.brain, name).numel() for name in ("gain", "rec_gain", "bias"))
                                     + (0 if self.brain.leak_delta is None else self.brain.leak_delta.numel()),
                "layer_norm": sum(p.numel() for p in self.ln.parameters()),
                "readout": sum(p.numel() for p in self.lm_head.parameters()),
                "edge_gains": sum(p.numel() for name, p in self.brain.named_parameters() if name.startswith("edge_theta")),
                "total": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def backend_metadata(self):
        return {"canonical_graph_and_groups_sha256": self.brain._canonical_sha256,
                "plasticity": self.config.plasticity, "group_count": self.brain.group_count,
                "edge_parameterization": self.config.edge_parameterization,
                "lag_map": list(self.brain.lag_map), "d_embed": self.config.d_embed,
                "readout_rank": self.config.readout_rank, "neuron_gain_policy": "reference_unconstrained",
                "leak_mode": getattr(self.config, "leak_mode", "fixed"), "leak": self.config.leak,
                "sparse_backend": self.brain.sparse_runtime().metadata()}

    @torch.no_grad()
    def adaptation_report(self):
        self.verify_frozen()
        b = self.brain
        base = b.w_values.detach().cpu()
        groups = b.edge_group_index.detach().cpu()
        counts = (b.w_offsets[1:] - b.w_offsets[:-1]).detach().cpu().long()
        row = torch.repeat_interleave(torch.arange(b.config.n_neurons), counts)
        group_gain = None
        if b.edge_theta_source is not None:
            types = b.node_type_index.detach().cpu()
            source = b.edge_theta_source.detach().cpu()[types]
            destination = b.edge_theta_destination.detach().cpu()[types]
            edge_gain = 1 + .1 * torch.tanh(source[b.w_indices.detach().cpu().long()] + destination[row])
        elif b.edge_theta is not None:
            group_gain = 1 + .1 * b.edge_theta.detach().cpu().tanh()
            edge_gain = group_gain[groups]
        else:
            edge_gain = torch.ones_like(base)
        current = base * edge_gain
        original_incoming = torch.zeros(b.config.n_neurons).index_add_(0, row, base.abs())
        current_incoming = torch.zeros_like(original_incoming).index_add_(0, row, current.abs())
        active = original_incoming > 0
        relative_incoming = current_incoming[active] / original_incoming[active] - 1
        changed = {str(tol): float(((edge_gain - 1).abs()[base != 0] > tol).float().mean())
                   if bool((base != 0).any()) else 0.0 for tol in (1e-4, 1e-3, 1e-2)}
        return {"canonical_buffers_verified": True, "neuron_gain_policy": "reference_unconstrained",
                "group_count": b.group_count, "edge_parameterization": self.config.edge_parameterization,
                "group_multiplier": None if group_gain is None else _stats(group_gain),
                "group_displacement": None if group_gain is None else _stats(group_gain - 1),
                "source_type_theta": None if b.edge_theta_source is None else _stats(b.edge_theta_source),
                "destination_type_theta": None if b.edge_theta_destination is None else _stats(b.edge_theta_destination),
                "edge_multiplier": _stats(edge_gain),
                "edge_displacement": _stats(edge_gain - 1), "changed_nonzero_edge_fraction": changed,
                "incoming_absolute_strength_relative_drift": _stats(relative_incoming),
                "zero_edges_preserved": bool(torch.equal(current[base == 0], base[base == 0])),
                "nonzero_signs_preserved": bool(torch.equal(current.sign(), base.sign())),
                "gain": _stats(b.gain), "rec_gain": _stats(b.rec_gain),
                "gain_displacement_from_initial": _stats(b.gain.detach().cpu() - self._initial_gain),
                "rec_gain_displacement_from_initial": _stats(b.rec_gain.detach().cpu() - self._initial_rec_gain),
                "effective_neuron_recurrent_multiplier": _stats(b.gain * b.rec_gain),
                "effective_neuron_recurrent_multiplier_displacement": _stats(
                    (b.gain * b.rec_gain).detach().cpu() - self._initial_gain * self._initial_rec_gain),
                "leak_mode": getattr(self.config, "leak_mode", "fixed"),
                "leak_delta": None if b.leak_delta is None else _stats(b.leak_delta),
                "effective_leak": (None if b.leak_delta is None
                                   else _stats(b.effective_leak().detach().cpu().squeeze(1))),
                "note": "Edge multipliers are bounded; original gain and rec_gain remain unconstrained."}


def _validate_groups(edge_group_index, edges, plasticity):
    if edge_group_index is None:
        if plasticity == "bounded10":
            raise ValueError("bounded10 requires verified canonical-edge-order group indices")
        return torch.empty(0, dtype=torch.long)
    groups = torch.as_tensor(edge_group_index).detach().cpu()
    if groups.ndim != 1 or groups.numel() != edges or groups.dtype not in (torch.int32, torch.int64):
        raise ValueError("edge_group_index must contain one integer per stored canonical CSR edge")
    groups = groups.long().clone()
    unique = torch.unique(groups, sorted=True)
    if unique.numel() and not torch.equal(unique, torch.arange(unique.numel())):
        raise ValueError("Group identifiers must be contiguous from zero, with no unused groups")
    return groups


def build_connectorch_model(reference, *, d_embed=None, plasticity="fixed", leak="fixed", edge_group_index=None,
                            readout_rank=0, lag_map=None, seed=42, copy_reference_parameters=False, node_type_index=None,
                            device="cpu", csr_factory: Callable | None = None):
    """Build an independent model from CPU reference buffers; never mutate reference.

    ``csr_factory`` optionally supplies a Connectorch-aware sparse runtime with
    the TrainableCSR constructor/mm/metadata/verify_frozen interface. Base graph
    loading and annotation joins remain external, so defaults cannot silently
    replace the reference graph, normalization, or transmitter signs.
    """
    if plasticity not in ("fixed", "bounded10"):
        raise ValueError("plasticity must be fixed or bounded10")
    if leak not in ("fixed", "trainable"):
        raise ValueError("leak must be fixed or trainable")
    if any(value.device.type != "cpu" for value in reference.parameters()):
        raise ValueError("Load the reference on CPU before constructing an independent model")
    for name in GRAPH_NAMES[:-2]:
        value = getattr(reference.brain, name)
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError("Reference graph/interface buffers must be fixed on CPU")
    if reference.brain.w_values.dtype != torch.float32 or not torch.isfinite(reference.brain.w_values).all():
        raise ValueError("Reference weights must be finite float32")
    cfg = copy.deepcopy(reference.config)
    cfg.d_embed = cfg.d_embed if d_embed is None else d_embed
    if not isinstance(cfg.d_embed, int) or cfg.d_embed < 1:
        raise ValueError("d_embed must be a positive integer")
    if not isinstance(readout_rank, int) or readout_rank < 0 or readout_rank > min(cfg.n_out, cfg.vocab_size):
        raise ValueError("readout_rank must be zero or a valid positive matrix rank")
    cfg.plasticity, cfg.readout_rank, cfg.leak_mode = plasticity, readout_rank, leak
    cfg.lag_map = list(range(cfg.delay_k)) if lag_map is None else list(lag_map)
    if (len(cfg.lag_map) != cfg.delay_k or any(not isinstance(lag, int) or lag < 0 or lag >= cfg.delay_k for lag in cfg.lag_map)):
        raise ValueError("lag_map must assign a lag in [0, delay_k) to every original input group")
    if len(reference.brain.in_index) != cfg.delay_k * (cfg.n_in // cfg.delay_k):
        raise ValueError("Input-neuron coverage does not match the reference projection groups")
    if node_type_index is not None and edge_group_index is not None:
        raise ValueError("Choose factorized node types or full edge pair groups, not both")
    node_types = _validate_groups(node_type_index, cfg.n_neurons, "fixed")
    groups = _validate_groups(edge_group_index, cfg.n_edges, plasticity if node_type_index is None else "fixed")
    cfg.edge_parameterization = ("factorized_cell_types" if node_types.numel() else
                                 "cell_type_pairs" if groups.numel() else "none")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ConnectorchFlyForCausalLM(reference, cfg, groups, node_types, csr_factory)
        # Reset after construction to match the reference trainer's RNG order.
        torch.manual_seed(seed)
        with torch.no_grad():
            b = model.brain
            b.wte.weight.normal_(std=1.)
            b.wte.weight[cfg.pad_token_id].zero_()
            b.in_proj.normal_(std=1 / math.sqrt(cfg.d_embed))
            b.gain.fill_(1.)
            b.bias.zero_()
            row = torch.repeat_interleave(torch.arange(cfg.n_neurons), (b.w_offsets[1:] - b.w_offsets[:-1]).long())
            sums = torch.zeros(cfg.n_neurons).index_add_(0, row, b.w_values.abs())
            b.rec_gain.copy_(torch.where(sums > 0, cfg.rec_target / sums.clamp_min(1e-30), 1.))
            model.ln.weight.fill_(1.)
            model.ln.bias.zero_()
            if not readout_rank:
                model.lm_head.weight.normal_(std=min(.02, 1 / math.sqrt(cfg.n_out)))
            else:
                for layer in model.lm_head:
                    layer.weight.normal_(std=min(.02, 1 / math.sqrt(layer.in_features)))
    if copy_reference_parameters:
        if cfg.d_embed != reference.config.d_embed or readout_rank:
            raise ValueError("Reference parameter copying requires original input width and full readout")
        source = dict(reference.named_parameters())
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name.startswith("brain.edge_theta"):
                    continue
                if name not in source or value.shape != source[name].shape:
                    raise ValueError("Reference parameter structure differs: " + name)
                value.copy_(source[name])
    # These are small audit references, not learned values or checkpoint buffers.
    # Rebuilding a run with the same seed reconstructs them before parameter resume.
    model._initial_gain = model.brain.gain.detach().cpu().clone()
    model._initial_rec_gain = model.brain.rec_gain.detach().cpu().clone()
    return model.to(device)
