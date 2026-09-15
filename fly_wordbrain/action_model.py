"""Candidate-aware, top-K action readout of the unchanged full fly circuit.

This is a separate experimental sensory interface. Existing pair-input
calibration is not valid for it. The frozen-brain prototype adds one affine
256-to-K correction to the count model's log probabilities; it does not train
an embedding, alter graph connections, or supply target labels as inputs.
"""
from typing import NamedTuple
from numbers import Integral

import numpy as np
import torch
from torch import nn

from .brain import array_digest
from .plastic_brain import PlasticBrain, PlasticState


ACTION_COUNT = 5
ROLE_NAMES = ("previous_word", "current_word", "candidate_1", "candidate_2",
              "candidate_3", "candidate_4", "candidate_5")


class CandidateInput(NamedTuple):
    previous: torch.Tensor
    candidates: torch.Tensor
    probabilities: torch.Tensor


def _word_ids(values, name, device):
    result = torch.as_tensor(values, device=device)
    if result.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(name + " must contain integer word IDs")
    return result.to(dtype=torch.long)


class CandidateActionBrain(PlasticBrain):
    """Context and K candidate roles using the parent's exact neural dynamics.

All K candidates must be distinct vocabulary IDs. Their probabilities are
the original count-model probabilities, not top-K-renormalized values.
They must be strictly positive, finite, and sum to at most one. Candidate order
defines action order. Inputs must remain valid even in inactive padding rows.
"""

    def __init__(self, graph_path, vocab_size, global_scale, top_k=5, **brain_kwargs):
        if not isinstance(top_k, Integral) or isinstance(top_k, bool) or top_k < 2:
            raise ValueError("top_k must be an integer of at least two")
        self.action_count = int(top_k)
        self.role_names = ("previous_word", "current_word") + tuple(
            "candidate_" + str(i + 1) for i in range(self.action_count))
        if brain_kwargs.pop("plasticity", False):
            raise ValueError("The action prototype requires a frozen brain")
        if brain_kwargs.get("feature_dim", 256) != 256:
            raise ValueError("The action prototype requires 256 brain features")
        if vocab_size < self.action_count + 2:
            raise ValueError("Need top_k distinct actions in addition to PAD/BOS")
        super().__init__(graph_path, vocab_size, global_scale,
                         plasticity=False, **brain_kwargs)
        if len(self.retina) < len(self.role_names):
            raise ValueError("Need at least top_k + 2 retinal neurons for disjoint roles")

        order = np.random.default_rng(self.seed + 1907).permutation(len(self.retina))
        banks = np.array_split(order, len(self.role_names))
        self.bank_sizes = tuple(len(bank) for bank in banks)
        self.lit_counts = tuple(max(1, int(round(len(bank) * .25))) for bank in banks)
        # Store only vocabulary x receptors values, divided across seven roles.
        # There is no vocabulary^K table.
        codes = np.zeros((self.vocab_size, len(self.retina)), dtype=np.float32)
        offsets = [0]
        for role, (bank, count) in enumerate(zip(banks, self.lit_counts)):
            rng = np.random.default_rng(self.seed + role)
            start = offsets[-1]
            for word in range(self.vocab_size):
                codes[word, start + rng.choice(len(bank), count, replace=False)] = self.input_high
            offsets.append(start + len(bank))
        self.bank_offsets = tuple(offsets)
        neurons = self.retina[torch.as_tensor(order, dtype=torch.long, device=self.device)]
        self.register_buffer("action_codes", torch.as_tensor(codes, device=self.device))
        self.register_buffer("action_neurons", neurons.clone())
        # The parent initialized its pair interface. Remove unused tables so
        # the live encoder stores exactly vocab_size * retinal_count values.
        for name in ("previous_codes", "current_codes", "previous_neurons", "current_neurons"):
            delattr(self, name)
        self.encoder_metadata = {
            "type": "fixed_candidate_role_sensory_code", "version": 2,
            "roles": list(self.role_names), "top_k": self.action_count, "seed": self.seed,
            "partition_seed": self.seed + 1907,
            "role_code_seeds": [self.seed + i for i in range(len(self.role_names))],
            "bank_sizes": list(self.bank_sizes), "illuminated_per_role": list(self.lit_counts),
            "stored_code_values": int(codes.size), "stored_code_bytes": int(codes.nbytes),
            "high": self.input_high, "learned": False,
            "candidate_amplitude": "input_high * (0.5 + 0.5 * original_candidate_probability)",
            "candidate_order": "input slot defines sensory bank and output action",
            "labels_are_inputs": False,
            "calibration": "Requires new training-only calibration; pair-input calibration is invalid",
        }
        self.initial_interface_fingerprint = self.interface_fingerprint()
        self.encoder_metadata["sha256"] = self.initial_interface_fingerprint

    def interface_fingerprint(self):
        # PlasticBrain.__init__ calls this method before our new buffers exist.
        if not hasattr(self, "action_codes"):
            return super().interface_fingerprint()
        arrays = [self.action_codes, self.action_neurons, self.readout_bucket,
                  self.readout_sign, self.readout_scale]
        return array_digest([value.detach().cpu().numpy() for value in arrays])

    def validate_inputs(self, previous, current, candidate_ids, probabilities):
        previous = _word_ids(previous, "previous", self.device)
        current = _word_ids(current, "current", self.device)
        candidates = _word_ids(candidate_ids, "candidates", self.device)
        probabilities = torch.as_tensor(probabilities, dtype=torch.float32, device=self.device)
        if previous.ndim != 1 or not len(previous) or current.shape != previous.shape:
            raise ValueError("Previous/current words require equally shaped nonempty [batch] IDs")
        expected = (len(previous), self.action_count)
        if candidates.shape != expected or probabilities.shape != expected:
            raise ValueError("Candidates/probabilities require [batch,top_k] shape")
        for name, ids in (("previous", previous), ("current", current), ("candidates", candidates)):
            if torch.any(ids < 0) or torch.any(ids >= self.vocab_size):
                raise ValueError(name + " word ID outside vocabulary")
        ordered = candidates.sort(dim=1).values
        if torch.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError("Each row must contain top_k distinct candidates")
        if torch.any((candidates == 0) | (candidates == 2)):
            raise ValueError("PAD/BOS cannot be candidate actions")
        if (not torch.isfinite(probabilities).all() or torch.any(probabilities <= 0)
                or torch.any(probabilities > 1) or torch.any(probabilities.sum(dim=1) > 1 + 1e-6)):
            raise ValueError("Original candidate probabilities must be finite, positive, and sum to at most one")
        # Fixed sensory inputs cannot become another trainable path.
        return previous, current, candidates, probabilities.detach()

    def retinal_current(self, bundle, current_ids):
        if not isinstance(bundle, CandidateInput):
            raise TypeError("CandidateActionBrain requires its candidate input bundle")
        previous, current, candidates, probabilities = self.validate_inputs(
            bundle.previous, current_ids, bundle.candidates, bundle.probabilities)
        word_roles = torch.cat((previous[:, None], current[:, None], candidates), dim=1)
        amplitudes = torch.cat((torch.ones((len(previous), 2), device=self.device),
                                .5 + .5 * probabilities), dim=1)
        result = torch.zeros((len(previous), self.neurons), dtype=torch.float32, device=self.device)
        for role, (start, end) in enumerate(zip(self.bank_offsets[:-1], self.bank_offsets[1:])):
            values = self.action_codes[word_roles[:, role], start:end] * amplitudes[:, role, None]
            result[:, self.action_neurons[start:end]] = self.input_gain * (values - self.input_center)
        return result

    def step(self, previous, current, candidate_ids, probabilities, state, active=None):
        """Return current candidate-aware features and causally advanced state.

        An inactive row preserves both state tensors exactly. It still requires
        valid sensory input; its returned logits/features must be loss-masked by
        the caller. No state is detached or reset implicitly between words.
        """
        bundle = CandidateInput(previous, candidate_ids, probabilities)
        if active is not None:
            active = torch.as_tensor(active, device=self.device)
            if active.dtype != torch.bool or active.shape != (state.h.shape[0],):
                raise ValueError("active must be a boolean [batch] mask")
        # Only retinal_current is overridden. The sparse graph, recurrence,
        # physiological approximation, and readout pooling are the parent code.
        features, next_state = super().step(bundle, current, state, plasticity_override=False)
        if active is not None:
            next_state = PlasticState(torch.where(active[:, None], next_state.h, state.h),
                                      torch.where(active[:, None], next_state.fast, state.fast))
            features = self.readout(next_state.h)
        return features, next_state


class FlyActionSelector(nn.Module):
    """Count log-probabilities plus a single zero-initialized fly action head."""

    def __init__(self, brain, feature_mean=None, feature_std=None):
        super().__init__()
        if not isinstance(brain, CandidateActionBrain):
            raise TypeError("Selector requires the candidate-aware sensory brain")
        if brain.feature_dim != 256 or brain.trainable_parameter_count() != 0 or brain.plasticity_enabled:
            raise ValueError("Prototype requires a frozen 256-feature brain")
        self.brain = brain
        self.action_count = brain.action_count
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("Provide both training-only feature mean/std or neither")
        self.features_calibrated = feature_mean is not None
        mean = torch.zeros(256) if feature_mean is None else torch.as_tensor(feature_mean)
        std = torch.ones(256) if feature_std is None else torch.as_tensor(feature_std)
        mean = mean.to(device=brain.device, dtype=torch.float32).detach().clone()
        std = std.to(device=brain.device, dtype=torch.float32).detach().clone()
        if (mean.shape != (256,) or std.shape != (256,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or torch.any(std <= 0)):
            raise ValueError("Feature mean/std must be finite [256] vectors with positive std")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.readout = nn.Linear(256, self.action_count, device=brain.device)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def initial_state(self, batch):
        return self.brain.initial_state(batch)

    def correction(self, features):
        return self.readout((features - self.feature_mean) / self.feature_std)

    def step(self, previous, current, candidate_ids, probabilities, state, active=None):
        previous, current, candidate_ids, probabilities = self.brain.validate_inputs(
            previous, current, candidate_ids, probabilities)
        features, next_state = self.brain.step(previous, current, candidate_ids, probabilities, state, active)
        logits = probabilities.log() + self.correction(features)
        return logits, next_state

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def metadata(self):
        return {
            "model": "frozen_connectome_candidate_residual_action_selector",
            "brain": self.brain.metadata(), "actions": self.action_count,
            "head": "single affine 256-to-K, weight and bias initialized to zero",
            "head_trainable_parameters": sum(p.numel() for p in self.readout.parameters()),
            "total_trainable_parameters": self.trainable_parameter_count(),
            "logits": "log(original_candidate_probability) + affine(normalized_brain_features)",
            "initial_prediction": "Exactly the count model's top candidate, including input-order tie breaking",
            "features_calibrated": self.features_calibrated,
            "evaluation": "Report accuracy over all positions; absent top-K targets remain misses",
        }
