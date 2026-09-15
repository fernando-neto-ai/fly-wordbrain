"""Seven delayed observations write temporary weights before an eighth forecast.

Prediction and observation are deliberately separate operations. A proposal is
formed before its word is observed, neural prediction activity supplies an
eligibility trace, and only the subsequently observed word can write fast
weights. Every base edge, endpoint and sign stays fixed.
"""
import math
from typing import NamedTuple, Optional

import numpy as np
import torch
from torch import nn

from .action_model import CandidateActionBrain, CandidateInput, _word_ids
from .brain import array_digest
from .plastic_brain import frozen_sparse_mm


def fixed_identity_codes(vocab_size, seed):
    """Seeded categorical identities with exactly unique, unit-norm 16-bit rows."""
    if not 1 <= vocab_size <= 65536:
        raise ValueError("Sixteen-bit identity codes support at most 65536 vocabulary items")
    packed = np.random.default_rng(seed).choice(65536, size=vocab_size, replace=False).astype(np.uint32)
    bits = ((packed[:, None] >> np.arange(16, dtype=np.uint32)) & 1).astype(np.float32)
    return (2 * bits - 1) / 4.


class FeedbackState(NamedTuple):
    h: torch.Tensor
    fast: torch.Tensor
    eligibility: torch.Tensor
    observations: int = 0
    pending: bool = False
    proposal_ids: Optional[torch.Tensor] = None
    proposal_probabilities: Optional[torch.Tensor] = None
    writing_enabled: bool = True


class FeedbackActionBrain(CandidateActionBrain):
    """Original circuit, with a small shared rule writing selected existing edges.

    ``predict`` reads but never writes fast weights. ``observe`` writes fast
    weights but does not advance neural activity. Seven observe calls are
    permitted, each bound to the immediately preceding proposal. A final eighth
    prediction has no corresponding observation within this episode.
    """

    def __init__(self, graph_path, vocab_size, global_scale, top_k=10,
                 initial_eligibility_retention=.9, identity_dimensions=16, **brain_kwargs):
        if top_k != 10:
            raise ValueError("The feedback experiment requires ten candidates plus an OTHER feedback bucket")
        if identity_dimensions != 16:
            raise ValueError("The feedback experiment uses a fixed 16-dimensional identity code")
        if vocab_size > 65536:
            raise ValueError("Sixteen-bit identity codes support at most 65536 vocabulary items")
        if not 0 < initial_eligibility_retention < 1:
            raise ValueError("Eligibility retention must be within (0,1)")
        super().__init__(graph_path, vocab_size, global_scale, top_k=top_k, **brain_kwargs)
        if self.strict_full_graph and self.candidates != 347:
            raise ValueError("Feedback experiment requires the original 347 selected existing edges")
        self.identity_dimensions = identity_dimensions
        self.feedback_dimensions = self.action_count + 1 + identity_dimensions + 1
        self.plasticity_enabled = True
        # CandidateActionBrain explicitly froze all parent parameters. Unfreeze
        # only these five four-group rule terms, all used below.
        self.rule_parameter_names = ("write_strength", "retention_logit", "gate_bias", "gate_pre", "gate_post",
                                     "feedback_modulation", "eligibility_retention_logit")
        for name in self.rule_parameter_names[:5]:
            getattr(self, name).requires_grad_(True)
        self.eligibility_retention_logit = nn.Parameter(torch.full((4,),
            math.log(initial_eligibility_retention / (1 - initial_eligibility_retention)), device=self.device))
        # Seeded nonzero modulation allows rank/identity/position feedback to
        # affect temporary weights before the slow rule has learned anything.
        rng = np.random.default_rng(self.seed + 7319)
        modulation = rng.normal(0., .1, (4, self.feedback_dimensions)).astype(np.float32)
        self.feedback_modulation = nn.Parameter(torch.as_tensor(modulation, device=self.device))
        identity = fixed_identity_codes(self.vocab_size, self.seed + 7320)
        self.register_buffer("observed_word_codes", torch.as_tensor(identity, device=self.device))
        self.observed_word_codes_sha256 = array_digest([identity])
        # Preserve the original signed pooling map, but give each output bin
        # its own fixed, padded gather row. A reduction over that row avoids
        # collisions between floating-point index_add writers on MPS.
        bucket = self.readout_bucket.detach().cpu().numpy()
        descending = self.descending.detach().cpu().numpy()
        signs = self.readout_sign.detach().cpu().numpy()
        counts = np.bincount(bucket, minlength=self.feature_dim)
        width = max(1, int(counts.max()))
        indices = np.zeros((self.feature_dim, width), dtype=np.int64)
        weights = np.zeros((self.feature_dim, width), dtype=np.float32)
        occupied = np.zeros(self.feature_dim, dtype=np.int64)
        for neuron, destination, sign in zip(descending, bucket, signs):
            offset = occupied[destination]
            indices[destination, offset] = neuron
            weights[destination, offset] = sign
            occupied[destination] += 1
        self.register_buffer("ordered_readout_indices", torch.as_tensor(indices, device=self.device))
        self.register_buffer("ordered_readout_signs", torch.as_tensor(weights, device=self.device))
        self.initial_interface_fingerprint = self.interface_fingerprint()
        if self.trainable_parameter_count() != 136:
            raise RuntimeError("Feedback rule must contain exactly 136 shared trainable parameters")

    def interface_fingerprint(self):
        base = super().interface_fingerprint()
        if not hasattr(self, "observed_word_codes"):
            return base
        arrays = [np.frombuffer(bytes.fromhex(base), dtype=np.uint8),
                  self.observed_word_codes.detach().cpu().numpy()]
        if hasattr(self, "ordered_readout_indices"):
            arrays.extend((self.ordered_readout_indices.detach().cpu().numpy(),
                           self.ordered_readout_signs.detach().cpu().numpy()))
        return array_digest(arrays)

    def readout(self, h):
        """The original fixed signed map, reduced in fixed per-bin rows.

        Each descending neuron appears once in its original bucket, with its
        original sign. Zero-sign padding contributes no value or gradient.
        All values remain on the input device, including state gradients.
        """
        if h.ndim != 2 or h.shape[1] != self.neurons:
            raise ValueError("Readout requires [batch,neurons] activity")
        values = h[:, self.ordered_readout_indices] * self.ordered_readout_signs
        return values.sum(dim=-1) / self.readout_scale

    def initial_state(self, batch):
        parent = super().initial_state(batch)
        return FeedbackState(parent.h, parent.fast, torch.zeros_like(parent.fast))

    def _validate_state(self, state, batch):
        if not isinstance(state, FeedbackState):
            raise TypeError("Feedback prediction requires FeedbackState")
        if (state.h.shape != (batch, self.neurons) or state.fast.shape != (batch, self.candidates)
                or state.eligibility.shape != state.fast.shape):
            raise ValueError("Feedback state shapes differ from batch or graph")
        if not 0 <= state.observations <= 7:
            raise ValueError("Feedback state observation count outside this eight-word episode")

    def predict(self, previous, current, candidate_ids, probabilities, state, plasticity=True):
        previous, current_ids, candidate_ids, probabilities = self.validate_inputs(
            previous, current, candidate_ids, probabilities)
        self._validate_state(state, len(previous))
        if state.pending:
            raise ValueError("Observe the pending proposal before predicting another word")
        current = self.retinal_current(CandidateInput(previous, candidate_ids, probabilities), current_ids)
        h, fast = state.h, state.fast
        eligibility_sum = torch.zeros_like(state.eligibility)
        group = self.candidate_group
        for _ in range(self.internal_steps):
            recurrent = frozen_sparse_mm(h, self.incoming, self.outgoing)
            if plasticity:
                correction = h[:, self.candidate_pre] * (self.effective_candidate_weights(fast) - self.candidate_weight)
                recurrent = recurrent.index_add(1, self.candidate_post, correction)
            new_h = (1 - self.leak) * h + self.leak * torch.tanh(self.global_scale * recurrent + current)
            # Both factors belong to prediction-time dynamics, before the
            # actual word or its surprise is supplied to observe().
            pre = torch.tanh(h[:, self.candidate_pre] / self.pre_scale)
            post = torch.tanh(new_h[:, self.candidate_post] / self.post_scale)
            eligibility_sum = eligibility_sum + pre * post / self.internal_steps
            h = new_h
        retention = torch.sigmoid(self.eligibility_retention_logit[group])
        eligibility = retention * state.eligibility + (1 - retention) * eligibility_sum
        result = FeedbackState(h, fast, eligibility, state.observations, True,
                               candidate_ids.detach().clone(), probabilities.detach().clone(), bool(plasticity))
        return self.readout(h), result

    def feedback_features(self, candidate_ids, probabilities, observed_ids, position):
        """Return [B,28]: 11-way rank error, fixed word identity, and position.

        OTHER retains all probability outside the ten proposals. Word IDs are
        categorical lookups, never treated as numerical semantic coordinates.
        """
        observed = _word_ids(observed_ids, "observed", self.device)
        if observed.ndim != 1 or not len(observed):
            raise ValueError("Observed words require a nonempty [batch] vector")
        dummy = torch.full_like(observed, 2)
        _, _, ids, probabilities = self.validate_inputs(dummy, dummy, candidate_ids, probabilities)
        if (torch.any(observed < 1) or torch.any(observed >= self.vocab_size)
                or torch.any((observed == 2) | (observed == 3))):
            raise ValueError("Observed context words must be lexical/UNK IDs, never PAD/BOS/EOS")
        positions = torch.as_tensor(position, device=self.device)
        if positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("Observation position must be an integer from 1 through 7")
        if positions.ndim == 0:
            positions = positions.expand(len(observed))
        if positions.shape != observed.shape or torch.any(positions < 1) or torch.any(positions > 7):
            raise ValueError("Observation position must be scalar or [batch], within 1 through 7")
        matches = ids == observed[:, None]
        ranks = torch.where(matches.any(dim=-1), matches.to(torch.long).argmax(dim=-1),
                            torch.full_like(observed, self.action_count))
        one_hot = torch.nn.functional.one_hot(ranks, self.action_count + 1).to(torch.float32)
        other = (1 - probabilities.sum(dim=1, keepdim=True)).clamp(min=0.)
        errors = one_hot - torch.cat((probabilities, other), dim=1)
        position_code = (positions.to(torch.float32)[:, None] - 1.) / 6.
        return torch.cat((errors, self.observed_word_codes[observed], position_code), dim=1)

    def observe(self, state, candidate_ids, probabilities, observed_ids, position):
        observed = _word_ids(observed_ids, "observed", self.device)
        if observed.ndim != 1:
            raise ValueError("Observed IDs must have shape [batch]")
        self._validate_state(state, len(observed))
        if not state.pending or state.proposal_ids is None or state.proposal_probabilities is None:
            raise ValueError("Prediction must precede observation")
        if state.observations == 7:
            raise ValueError("The eighth word is hidden; this episode permits only seven observations")
        dummy = torch.full_like(observed, 2)
        _, _, ids, probabilities = self.validate_inputs(dummy, dummy, candidate_ids, probabilities)
        if not torch.equal(ids, state.proposal_ids) or not torch.equal(probabilities, state.proposal_probabilities):
            raise ValueError("Observation must refer to the unchanged pending proposal")
        features = self.feedback_features(ids, probabilities, observed, position)
        positions = torch.as_tensor(position, device=self.device)
        if not bool(torch.all(positions == state.observations + 1)):
            raise ValueError("Observation positions must advance causally from 1 through 7")
        fast = state.fast
        if state.writing_enabled:
            group = self.candidate_group
            modulation = torch.tanh(features @ self.feedback_modulation.transpose(0, 1))[:, group]
            pre = torch.tanh(state.h[:, self.candidate_pre] / self.pre_scale)
            post = torch.tanh(state.h[:, self.candidate_post] / self.post_scale)
            gate = torch.sigmoid(self.gate_bias[group] + self.gate_pre[group] * pre + self.gate_post[group] * post)
            retention = torch.sigmoid(self.retention_logit[group])
            eta = self.max_write_strength * torch.tanh(self.write_strength[group])
            fast = retention * fast + eta * gate * modulation * state.eligibility
        return FeedbackState(state.h, fast, state.eligibility, state.observations + 1, False,
                             None, None, state.writing_enabled)

    def step(self, *args, **kwargs):
        raise RuntimeError("Use separate predict() and observe() for causal feedback")

    def metadata(self):
        result = super().metadata()
        result.update(model="full_connectome_delayed_observation_feedback_fast_weights",
            readout="Same fixed signed descending-neuron pooling via padded per-bin gather and sum",
            readout_reduction_order="Canonical descending-array order inside each original bucket; zero-sign padding",
            readout_cache_shape=list(self.ordered_readout_indices.shape),
            episode="Seven observed context words followed by one hidden-word prediction",
            trainable_rule_parameters=self.trainable_parameter_count(),
            rule_parameter_names=list(self.rule_parameter_names),
            feedback={"dimensions": self.feedback_dimensions, "rank_error_dimensions": 11,
                      "observed_identity_dimensions": self.identity_dimensions,
                      "observed_word_code_seed": self.seed + 7320,
                      "observed_word_code_construction": "Unique seeded 16-bit words sampled without replacement; signed bits divided by 4",
                      "observed_word_codes_sha256": self.observed_word_codes_sha256,
                      "observed_word_code_values": int(self.observed_word_codes.numel()),
                      "position": "(observed word position - 1) / 6, positions 1..7",
                      "rank_error": "onehot(observed rank or OTHER) - [original probabilities, other mass]"},
            eligibility="E'=sigmoid(lambda)*E+(1-sigmoid(lambda))*mean_internal(tanh(pre/scale)*tanh(post/scale))",
            write="H'=sigmoid(retention)*H + max_eta*tanh(eta)*sigmoid(g0+gpre*pre+gpost*post)*tanh(U*feedback)*E",
            fast_write_timing="Only observe(); prediction only reads frozen base plus bounded fast gain",
            override_off="Disables fast reading and writing; activity and eligibility still advance",
            reset="All neural activity, fast weights and eligibility reset for every eight-word window",
            autograd="BPTT through seven observed feedback updates; base sparse matrices remain frozen")
        return result


class FeedbackSelector(nn.Module):
    """Tiny residual action head; final targets are not part of this API."""

    def __init__(self, brain, feature_mean=None, feature_std=None):
        super().__init__()
        if not isinstance(brain, FeedbackActionBrain):
            raise TypeError("FeedbackSelector requires FeedbackActionBrain")
        self.brain = brain
        self.action_count = brain.action_count
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("Provide both feature mean and std, or neither")
        self.features_calibrated = feature_mean is not None
        mean = torch.zeros(256) if feature_mean is None else torch.as_tensor(feature_mean)
        std = torch.ones(256) if feature_std is None else torch.as_tensor(feature_std)
        mean = mean.to(device=brain.device, dtype=torch.float32).detach().clone()
        std = std.to(device=brain.device, dtype=torch.float32).detach().clone()
        if (mean.shape != (256,) or std.shape != (256,) or not bool(torch.isfinite(mean).all())
                or not bool(torch.isfinite(std).all()) or not bool(torch.all(std > 0))):
            raise ValueError("Feature calibration requires finite [256] vectors and positive std")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.readout = nn.Linear(256, self.action_count, device=brain.device)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def initial_state(self, batch):
        return self.brain.initial_state(batch)

    @staticmethod
    def _field(batch, name):
        return batch[name] if isinstance(batch, dict) else getattr(batch, name)

    def _inputs(self, batch):
        previous = _word_ids(self._field(batch, "previous"), "previous", self.brain.device)
        current = _word_ids(self._field(batch, "current"), "current", self.brain.device)
        candidates = _word_ids(self._field(batch, "candidates"), "candidates", self.brain.device)
        probabilities = torch.as_tensor(self._field(batch, "probabilities"), dtype=torch.float32, device=self.brain.device)
        observed = _word_ids(self._field(batch, "observed_ids"), "observed", self.brain.device)
        if (previous.ndim != 2 or previous.shape[1] != 8 or not len(previous)
                or current.shape != previous.shape or observed.shape != (len(previous), 7)
                or candidates.shape != (len(previous), 8, self.action_count)
                or probabilities.shape != candidates.shape):
            raise ValueError("Require previous/current [B,8], candidates/probabilities [B,8,10], observed_ids [B,7]")
        if (not bool(torch.all(previous[:, :2] == 2)) or not bool(torch.all(current[:, 0] == 2))
                or not torch.equal(previous[:, 2:], observed[:, :6])
                or not torch.equal(current[:, 1:], observed)):
            raise ValueError("Context must contain exactly the already observed prefix, starting with BOS/BOS")
        return previous, current, candidates, probabilities, observed

    def features(self, batch, plasticity=True):
        """Return final features/state, preserving BPTT through all observations."""
        previous, current, candidates, probabilities, observed = self._inputs(batch)
        state = self.initial_state(len(previous))
        for position in range(8):
            features, state = self.brain.predict(previous[:, position], current[:, position],
                candidates[:, position], probabilities[:, position], state, plasticity=plasticity)
            if position < 7:
                state = self.brain.observe(state, candidates[:, position], probabilities[:, position],
                                           observed[:, position], position + 1)
        return features, state

    def correction(self, features):
        return self.readout((features - self.feature_mean) / self.feature_std)

    def forward(self, batch, plasticity=True):
        features, _ = self.features(batch, plasticity=plasticity)
        probabilities = torch.as_tensor(self._field(batch, "probabilities"),
                                         dtype=torch.float32, device=self.brain.device).detach()
        return probabilities[:, 7].log() + self.correction(features)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def metadata(self):
        return {"model": "seven_observations_then_eighth_word_feedback_selector", "brain": self.brain.metadata(),
                "actions": self.action_count, "head_trainable_parameters": sum(p.numel() for p in self.readout.parameters()),
                "rule_trainable_parameters": self.brain.trainable_parameter_count(),
                "total_trainable_parameters": self.trainable_parameter_count(),
                "head": "One zero-initialized affine 256-to-10 residual on count log probabilities",
                "features_calibrated": self.features_calibrated,
                "final_target_input": False,
                "gradient_startup": "First head update has zero rule gradients; after head warmup verify final-loss rule gradients",
                "scoring": "Final eighth-word candidate accuracy; out-of-candidate gold remains a miss"}
