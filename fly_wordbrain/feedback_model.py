"""Seven delayed observations write temporary weights before an eighth forecast.

Prediction and observation are deliberately separate operations. A proposal is
formed before its word is observed, neural prediction activity supplies an
eligibility trace, and only the subsequently observed word can write fast
weights. Every base edge, endpoint and sign stays fixed.
"""
import math
from pathlib import Path
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
                 initial_eligibility_retention=.9, identity_dimensions=16,
                 structure_path=None, trainable_susceptibility=False, **brain_kwargs):
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
        self.expanded_structure = structure_path is not None
        self.structure_path = str(Path(structure_path).resolve()) if self.expanded_structure else None
        self.structure_metadata = None
        self.group_names = ["hDeltaH", "hDeltaA", "hDeltaI", "hDeltaG"]
        self.original_candidate_count = self.candidates
        self.original_selection_inclusive_fingerprint = self.initial_graph_fingerprint
        self.base_graph_sha256_before_selection = self.base_graph_fingerprint()
        if self.expanded_structure:
            self._install_structure(structure_path)
        self.group_count = len(self.group_names)
        self.base_graph_sha256_after_selection = self.base_graph_fingerprint() if self.expanded_structure else self.base_graph_sha256_before_selection
        if self.base_graph_sha256_after_selection != self.base_graph_sha256_before_selection:
            raise RuntimeError("Installing a fast-weight selection changed the canonical base graph")
        if self.expanded_structure:
            # This fingerprint includes the verified selection as well as the
            # full base graph; changing selection is not changing the graph.
            self.initial_graph_fingerprint = self.frozen_fingerprint()
            self._build_candidate_incidence()
        self.identity_dimensions = identity_dimensions
        self.feedback_dimensions = self.action_count + 1 + identity_dimensions + 1
        self.plasticity_enabled = True
        # CandidateActionBrain explicitly froze all parent parameters. Unfreeze
        # only the rule terms actually used below. Additional groups inherit
        # the same initial rule values as the original four groups.
        self.rule_parameter_names = ("write_strength", "retention_logit", "gate_bias", "gate_pre", "gate_post",
                                     "feedback_modulation", "eligibility_retention_logit")
        for name in self.rule_parameter_names[:5]:
            original = getattr(self, name)
            if self.group_count != 4:
                expanded = torch.cat((original.detach(), original.detach()[:1].expand(self.group_count - 4)))
                setattr(self, name, nn.Parameter(expanded.clone()))
            else:
                original.requires_grad_(True)
        self.eligibility_retention_logit = nn.Parameter(torch.full((self.group_count,),
            math.log(initial_eligibility_retention / (1 - initial_eligibility_retention)), device=self.device))
        # Seeded nonzero modulation allows rank/identity/position feedback to
        # affect temporary weights before the slow rule has learned anything.
        rng = np.random.default_rng(self.seed + 7319)
        modulation = rng.normal(0., .1, (self.group_count, self.feedback_dimensions)).astype(np.float32)
        self.feedback_modulation = nn.Parameter(torch.as_tensor(modulation, device=self.device))
        self.trainable_susceptibility = bool(trainable_susceptibility)
        if self.trainable_susceptibility:
            self.slow_susceptibility = nn.Parameter(torch.zeros(self.candidates, device=self.device))
            self.rule_parameter_names += ("slow_susceptibility",)
        else:
            self.register_parameter("slow_susceptibility", None)
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
        expected = 34 * self.group_count + (self.candidates if self.trainable_susceptibility else 0)
        if self.trainable_parameter_count() != expected:
            raise RuntimeError("Feedback parameter count differs from 34 per group plus optional per-edge susceptibility")

    def base_graph_fingerprint(self):
        """Canonical in-memory identities/endpoints/weights, excluding selection."""
        return array_digest([value.detach().cpu().numpy() for value in
            (self.neuron_ids, self.raw_outgoing_ptr, self.raw_outgoing_post, self.raw_outgoing_weight)])

    def _install_structure(self, structure_path):
        from .fast_structure import load_structure
        graph = {name: value.detach().cpu().numpy().copy() for name, value in (
            ("ptr", self.raw_outgoing_ptr), ("post", self.raw_outgoing_post),
            ("weight", self.raw_outgoing_weight), ("candidate_edge_ids", self.candidate_edge_ids),
            ("candidate_group", self.candidate_group))}
        arrays, metadata = load_structure(structure_path, graph, self.graph_file_sha256)
        for name in ("candidate_edge_ids", "candidate_group", "candidate_pre", "candidate_post", "candidate_weight"):
            dtype = torch.float32 if name == "candidate_weight" else torch.long
            setattr(self, name, torch.as_tensor(arrays[name].copy(), dtype=dtype, device=self.device))
        self.candidates = len(arrays["candidate_edge_ids"])
        self.pre_scale = torch.ones(self.candidates, dtype=torch.float32, device=self.device)
        self.post_scale = torch.ones_like(self.pre_scale)
        self.group_names = list(metadata["group_names"])
        self.structure_metadata = metadata

    def _build_candidate_incidence(self):
        """Fixed U-by-E sum and E-by-U transpose, using only O(E+U) storage.

        Each selected edge occurs exactly once in its destination's CSR row.
        Both CPU CSR and Metal's per-row kernel avoid colliding scatter sums.
        """
        post = self.candidate_post.detach().cpu().numpy()
        destinations, inverse = np.unique(post, return_inverse=True)
        order = np.argsort(inverse, kind="stable").astype(np.int64)
        ptr = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(destinations)))].astype(np.int64)
        transpose_ptr = np.arange(self.candidates + 1, dtype=np.int64)
        values = np.ones(self.candidates, dtype=np.float32)
        self.register_buffer("candidate_unique_posts", torch.as_tensor(destinations, dtype=torch.long, device=self.device))
        for name, row_ptr, columns, shape in (
            ("fast_incidence", ptr, order, (len(destinations), self.candidates)),
            ("fast_incidence_transpose", transpose_ptr, inverse.astype(np.int64), (self.candidates, len(destinations)))):
            if self.device.type == "mps":
                from .metal_sparse import DenseMpsCSR
                setattr(self, name, DenseMpsCSR(row_ptr, columns, values, shape, device=self.device))
            else:
                matrix = torch.sparse_csr_tensor(torch.as_tensor(row_ptr, device=self.device),
                    torch.as_tensor(columns, device=self.device), torch.as_tensor(values.copy(), device=self.device),
                    size=shape, dtype=torch.float32, device=self.device, check_invariants=True)
                self.register_buffer(name, matrix)

    def aggregate_candidate_corrections(self, correction):
        """Return one sum per unique selected postsynaptic neuron."""
        if not self.expanded_structure or correction.ndim != 2 or correction.shape[1] != self.candidates:
            raise ValueError("Expanded candidate correction must have shape [batch,selected_edges]")
        return frozen_sparse_mm(correction, self.fast_incidence, self.fast_incidence_transpose)

    def set_activity_scales(self, pre, post):
        if not getattr(self, "expanded_structure", False):
            return super().set_activity_scales(pre, post)
        for name, values in (("pre_scale", pre), ("post_scale", post)):
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device).detach()
            if values.ndim == 0:
                values = values.expand(self.candidates)
            elif values.shape == (self.group_count,) and self.candidates != self.group_count:
                values = values[self.candidate_group]
            if values.shape != (self.candidates,) or not bool(torch.isfinite(values).all()) or not bool(torch.all(values > 0)):
                raise ValueError("Activity scales must be finite positive scalar, per-edge, or per-group values")
            getattr(self, name).copy_(values)

    def verify_frozen(self):
        if getattr(self, "expanded_structure", False) and self.device.type == "mps":
            self.fast_incidence.verify_frozen()
            self.fast_incidence_transpose.verify_frozen()
        return super().verify_frozen()

    def interface_fingerprint(self):
        base = super().interface_fingerprint()
        if not hasattr(self, "observed_word_codes"):
            return base
        arrays = [np.frombuffer(bytes.fromhex(base), dtype=np.uint8),
                  self.observed_word_codes.detach().cpu().numpy()]
        if hasattr(self, "ordered_readout_indices"):
            arrays.extend((self.ordered_readout_indices.detach().cpu().numpy(),
                           self.ordered_readout_signs.detach().cpu().numpy()))
        if hasattr(self, "candidate_unique_posts"):
            arrays.append(self.candidate_unique_posts.detach().cpu().numpy())
            for matrix in (self.fast_incidence, self.fast_incidence_transpose):
                arrays.append(np.asarray(matrix.shape, dtype=np.int64))
                arrays.extend(value.detach().cpu().numpy() for value in
                    (matrix.crow_indices(), matrix.col_indices(), matrix.values()))
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
                if self.expanded_structure:
                    delta = self.candidate_weight * torch.expm1(math.log(2.) * torch.tanh(fast))
                    correction = h[:, self.candidate_pre] * delta
                    aggregated = self.aggregate_candidate_corrections(correction)
                    recurrent = recurrent.index_copy(1, self.candidate_unique_posts,
                        recurrent[:, self.candidate_unique_posts] + aggregated)
                else:
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
            if self.slow_susceptibility is not None:
                eta = eta * (2. * torch.sigmoid(self.slow_susceptibility))
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
            group_count=self.group_count, group_names=list(self.group_names),
            candidate_groups=list(self.group_names) if self.expanded_structure else result["candidate_groups"],
            trainable_shared_rule_parameters=34 * self.group_count,
            trainable_susceptibility_parameters=self.candidates if self.trainable_susceptibility else 0,
            trainable_susceptibility=self.trainable_susceptibility,
            susceptibility="2*sigmoid(slow_susceptibility); bounded positive write multiplier, initialized exactly one" if self.trainable_susceptibility else None,
            expanded_structure=self.expanded_structure, structure_path=self.structure_path, structure=self.structure_metadata,
            original_candidate_edges=self.original_candidate_count,
            base_graph_sha256_before_selection=self.base_graph_sha256_before_selection,
            base_graph_sha256_after_selection=self.base_graph_sha256_after_selection,
            base_graph_digest_definition="In-memory canonical neuron_ids, raw outgoing ptr/post/weight; excludes selected plastic edges",
            original_selection_inclusive_fingerprint=self.original_selection_inclusive_fingerprint,
            selection_inclusive_fingerprint=self.initial_graph_fingerprint,
            graph_fingerprint_definition="Full fixed base graph plus selected edge IDs/endpoints/groups/weights and populations",
            candidate_correction="weight*expm1(log(2)*tanh(H)); fixed CSR sum to unique posts" if self.expanded_structure else "effective_weight-base_weight; original index_add",
            feedback={"dimensions": self.feedback_dimensions, "rank_error_dimensions": 11,
                      "observed_identity_dimensions": self.identity_dimensions,
                      "observed_word_code_seed": self.seed + 7320,
                      "observed_word_code_construction": "Unique seeded 16-bit words sampled without replacement; signed bits divided by 4",
                      "observed_word_codes_sha256": self.observed_word_codes_sha256,
                      "observed_word_code_values": int(self.observed_word_codes.numel()),
                      "position": "(observed word position - 1) / 6, positions 1..7",
                      "rank_error": "onehot(observed rank or OTHER) - [original probabilities, other mass]"},
            eligibility="E'=sigmoid(lambda)*E+(1-sigmoid(lambda))*mean_internal(tanh(pre/scale)*tanh(post/scale))",
            write="H'=sigmoid(retention)*H + max_eta*tanh(eta)*sigmoid(g0+gpre*pre+gpost*post)*tanh(U*feedback)*E"
                  + (" * (2*sigmoid(slow_susceptibility))" if self.trainable_susceptibility else ""),
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
