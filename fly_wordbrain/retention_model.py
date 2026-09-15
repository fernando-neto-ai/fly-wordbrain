"""Task heads and differentiable delayed-memory supervision on existing edges.

The fly receives the same causal word/proposal inputs as the frozen language
selector. Probe task IDs and labels enter only the supervised loss. Auxiliary
earlier losses may shape training, but the final head reads only prediction
eight's 256 features; it has no external store of earlier neural responses.
"""
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .brain import array_digest
from .feedback_model import FeedbackSelector


PREDICTION_POSITIONS = (4, 5, 6, 7, 8)
TASK_NAMES = ("identity", "order")
STD_FLOOR = 1e-12


def _task_index(task):
    if isinstance(task, str) and task in TASK_NAMES:
        return TASK_NAMES.index(task)
    if isinstance(task, Integral) and not isinstance(task, bool) and task in (0, 1):
        return int(task)
    raise ValueError("Task must be identity/0 or order/1")


class RetentionReadout(nn.Module):
    """Identical lightweight heads for cached control and differentiable fly.

    ``means`` and ``stds`` are fixed training-only moments for predictions4..8.
    No graph or neural state is allocated by this class.
    """

    def __init__(self, means, stds, device="cpu", seed=0):
        super().__init__()
        if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Head seed must be a nonnegative integer")
        self.seed = int(seed)
        mean = torch.as_tensor(means, dtype=torch.float32, device=device).detach().clone()
        std = torch.as_tensor(stds, dtype=torch.float32, device=device).detach().clone()
        if (mean.shape != (5, 256) or std.shape != (5, 256)
                or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all())
                or not bool(torch.all(std >= 0))):
            raise ValueError("Training normalization needs finite means and nonnegative stds of shape [5,256]")
        self.register_buffer("feature_means", mean)
        self.register_buffer("feature_stds", std.clamp_min(STD_FLOOR))
        # Initialize on CPU with an isolated generator, then move the small
        # heads. Nonzero weights allow rule gradients on the first loss.
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        with torch.random.fork_rng(devices=[]):
            self.identity_head = nn.Linear(256, 10, device="cpu", dtype=torch.float32)
            self.order_head = nn.Linear(256, 2, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                for head in (self.identity_head, self.order_head):
                    head.weight.copy_(.02 * torch.randn(head.weight.shape, generator=generator))
                    head.bias.zero_()
        self.identity_head.to(self.device)
        self.order_head.to(self.device)

    @property
    def device(self):
        return self.feature_means.device

    def head_parameters(self):
        return list(self.identity_head.parameters()) + list(self.order_head.parameters())

    def logits(self, features, task):
        """Map [B,5,256] to all times, or [B,256] to the final-time logits."""
        index = _task_index(task)
        if not isinstance(features, torch.Tensor) or features.device != self.device:
            raise ValueError("Features must be a tensor on the readout device")
        if features.ndim == 3 and features.shape[1:] == (5, 256):
            normalized = (features - self.feature_means) / self.feature_stds
        elif features.ndim == 2 and features.shape[1] == 256:
            normalized = (features - self.feature_means[-1]) / self.feature_stds[-1]
        else:
            raise ValueError("Features require [B,5,256], or [B,256] for prediction eight")
        return (self.identity_head if index == 0 else self.order_head)(normalized)

    def memory_loss(self, features, task_ids, labels, time_weights):
        """Task-balanced CE with normalized nonnegative weights over times4..8.

        Task IDs are 0=identity, 1=order. Each present task contributes equally,
        regardless of its row count. Within a task, rows are averaged. Use
        [0,0,0,0,1] for final-only supervision. Labels never enter the fly.
        """
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[1:] != (5, 256) or not len(features)):
            raise ValueError("Memory loss requires nonempty [B,5,256] features")
        ids = torch.as_tensor(task_ids, device=self.device)
        targets = torch.as_tensor(labels, device=self.device)
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if (ids.shape != (len(features),) or targets.shape != ids.shape
                or ids.dtype not in integer_dtypes or targets.dtype not in integer_dtypes
                or bool(torch.any((ids < 0) | (ids > 1)))):
            raise ValueError("Task IDs and labels must be integer [B]; task IDs are 0 or 1")
        ids, targets = ids.long(), targets.long()
        weights = torch.as_tensor(time_weights, dtype=torch.float32, device=self.device).detach()
        if (weights.shape != (5,) or not bool(torch.isfinite(weights).all())
                or bool(torch.any(weights < 0)) or not bool(torch.isfinite(weights.sum()))
                or not bool(weights.sum() > 0)):
            raise ValueError("Time weights must be five finite nonnegative values with positive sum")
        weights = weights / weights.sum()
        losses = []
        for task, classes in ((0, 10), (1, 2)):
            mask = ids == task
            if not bool(mask.any()):
                continue
            selected = targets[mask]
            if bool(torch.any((selected < 0) | (selected >= classes))):
                raise ValueError("Label outside task class range")
            scores = self.logits(features[mask], task)
            repeated = selected[:, None].expand(-1, 5)
            ce = F.cross_entropy(scores.reshape(-1, classes), repeated.reshape(-1), reduction="none")
            losses.append((ce.reshape(-1, 5).mean(dim=0) * weights).sum())
        return torch.stack(losses).mean()

    def metadata(self):
        normalization = array_digest([self.feature_means.detach().cpu().numpy(),
                                      self.feature_stds.detach().cpu().numpy()])
        return {"kind": "retention_readout", "prediction_positions": list(PREDICTION_POSITIONS),
            "task_ids": {"identity": 0, "order": 1},
            "trainable_head_parameters": sum(value.numel() for value in self.head_parameters()),
            "head_seed": self.seed, "head_initialization": "Independent seeded normal weights std0.02; zero biases",
            "normalization_sha256": normalization, "normalization_std_floor": STD_FLOOR,
            "normalization": "Fixed training-only moments separately for prediction times4..8; no feature mask",
            "final_prediction": "Each final head sees only the256features at prediction eight",
            "labels_enter_neural_inputs": False}


class RetentionModel(RetentionReadout):
    """Consume one selector, enabling only its declared existing-edge rules.

    Use RetentionReadout for cached controls without a second copy of the graph.
    The original language readout stays frozen and is never called here.
    """

    def __init__(self, frozen_selector, means, stds, train_rules=True, seed=0):
        if not isinstance(frozen_selector, FeedbackSelector):
            raise TypeError("RetentionModel requires a FeedbackSelector")
        if not isinstance(train_rules, bool):
            raise ValueError("train_rules must be an explicit Boolean")
        super().__init__(means, stds, device=frozen_selector.brain.device, seed=seed)
        self.selector = frozen_selector
        self.selector.requires_grad_(False)
        self.selector.eval()
        self.train_rules = train_rules
        declared = tuple(self.brain.rule_parameter_names)
        available = dict(self.brain.named_parameters())
        if len(set(declared)) != len(declared) or any(name not in available for name in declared):
            raise ValueError("Brain rule_parameter_names must uniquely identify actual parameters")
        self.rule_parameter_names = declared
        if train_rules:
            for name in declared:
                available[name].requires_grad_(True)
        enabled = {name for name, parameter in self.brain.named_parameters() if parameter.requires_grad}
        if enabled != (set(declared) if train_rules else set()):
            raise RuntimeError("Only declared existing-edge rules may be trainable")

    @property
    def brain(self):
        return self.selector.brain

    def train(self, mode=True):
        super().train(mode)
        # eval does not disable gradients; the existing brain has no dropout
        # or batch-dependent training behavior to introduce here.
        self.selector.eval()
        return self

    def rule_parameters(self):
        available = dict(self.brain.named_parameters())
        return [available[name] for name in self.rule_parameter_names if available[name].requires_grad]

    def forward_features(self, batch, plasticity=True, clear_fast_before_final=False, return_state=False):
        """Raw differentiable [B,5,256] after predictions 4..8.

        The optional erasure intervention zeros only H immediately before
        prediction eight, retaining neural activity and eligibility. With
        return_state=True, return (features, final FeedbackState).
        """
        if any(not isinstance(value, bool) for value in (plasticity, clear_fast_before_final, return_state)):
            raise ValueError("Trajectory controls must be explicit Booleans")
        previous, current, candidates, probabilities, observed = self.selector._inputs(batch)
        state = self.brain.initial_state(len(previous))
        outputs = []
        for position in range(8):
            if position == 7 and clear_fast_before_final:
                state = state._replace(fast=torch.zeros_like(state.fast))
            features, state = self.brain.predict(previous[:, position], current[:, position],
                candidates[:, position], probabilities[:, position], state, plasticity=plasticity)
            if position >= 3:
                outputs.append(features)
            if position < 7:
                state = self.brain.observe(state, candidates[:, position], probabilities[:, position],
                    observed[:, position], position + 1)
        features = torch.stack(outputs, dim=1)
        return (features, state) if return_state else features

    def forward(self, batch, plasticity=True, clear_fast_before_final=False, return_state=False):
        return self.forward_features(batch, plasticity, clear_fast_before_final, return_state)

    def metadata(self):
        return {**super().metadata(), "kind": "existing_edge_delayed_retention_model",
            "train_rules": self.train_rules,
            "trainable_rule_parameters": sum(value.numel() for value in self.rule_parameters()),
            "rule_parameter_names": list(self.rule_parameter_names), "old_language_head_frozen": True,
            "erasure_control": "Zero only fast state immediately before prediction eight; preserve h and eligibility"}


def memory_loss(model, features, task_ids, labels, time_weights):
    """Functional spelling of RetentionModel.memory_loss for training scripts."""
    return model.memory_loss(features, task_ids, labels, time_weights)
