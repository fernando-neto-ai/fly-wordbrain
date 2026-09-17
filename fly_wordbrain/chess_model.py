"""A chess head on the fly brain, sharing the language model's recurrence exactly.

The point of this task is to ask what the *brain* can do, so the brain must be the same
object it is for language: the same frozen graph, the same per-neuron gain, rec_gain and
bias, and the same update

    x <- (1 - leak) x + leak tanh(gain (rec_gain (W x) + drive) + bias)

Only the way drive arrives and the way the state is read out differ. `settle` re-implements
that loop because a board is not a sequence -- the same current is held across every step
rather than one token arriving per step -- and `tests/test_chess_model.py` checks the
re-implementation against the pinned model's own forward pass, step for step, so the two
cannot drift apart silently.

A position is one settling run, so the encoder writes a current onto the brain's existing
input neurons and the readout takes the settled state. Both are factorised through a small
rank so the task-specific parameters stay the same order as the language arm's; the
reference ChessFly uses a full 780 x 10,855 encoder, which is larger than this whole model.
"""
import numpy as np
import torch
from torch import nn

from .chess_encoding import FEATURES

MOVES = 1968
VALUE_BINS = 64


class ChessFly(nn.Module):
    def __init__(self, brain, settle_steps=5, encoder_rank=64, readout_rank=64,
                 moves=MOVES, value_bins=VALUE_BINS, features=FEATURES):
        super().__init__()
        # A plain attribute, not a submodule. The brain is shared with the language model,
        # and registering it here would list its parameters twice and write a second copy
        # of all 9,050,172 edge values into every checkpoint. Autograd does not care about
        # registration, so gradients still reach the brain's own parameters.
        object.__setattr__(self, "brain", brain)
        self.settle_steps = settle_steps
        self.value_bins = value_bins
        inputs, outputs = brain.in_index.numel(), brain.out_index.numel()
        self.encoder = nn.Sequential(nn.Linear(features, encoder_rank, bias=False),
                                     nn.Linear(encoder_rank, inputs, bias=False))
        self.ln = nn.LayerNorm(outputs)
        self.trunk = nn.Linear(outputs, readout_rank, bias=False)
        self.policy = nn.Linear(readout_rank, moves, bias=False)
        self.value = nn.Linear(readout_rank, value_bins, bias=False)
        self.register_buffer("bin_centres", (torch.arange(value_bins) + .5) / value_bins)

    def settle(self, drive, steps=None, collect=False):
        """Run the brain's recurrence from rest under `drive` of shape (n_in, batch).

        A 3-D drive of shape (steps, n_in, batch) is applied one slice per step instead,
        which is what the language model does and what the parity test exercises.
        """
        brain = self.brain
        config = brain.config
        runtime, values = brain.sparse_runtime(), brain.effective_values()
        gain, bias, rec_gain = brain.gain[:, None], brain.bias[:, None], brain.rec_gain[:, None]
        sequenced = drive.dim() == 3
        steps = steps or (drive.shape[0] if sequenced else self.settle_steps)
        batch = drive.shape[-1]
        state = drive.new_zeros(config.n_neurons, batch)
        collected = []
        for step in range(steps):
            current = drive[step] if sequenced else drive
            recurrent = runtime.mm(state.t().contiguous(), values).t().contiguous()
            pre = (rec_gain * recurrent).index_add(0, brain.in_index, current)
            state = (1 - config.leak) * state + config.leak * torch.tanh(gain * pre + bias)
            if collect:
                collected.append(state[brain.out_index])
        if collect:
            return torch.stack(collected, dim=0).permute(2, 0, 1)
        return state[brain.out_index].t()

    def chess_parameters(self):
        """The task-specific parameters: everything here except the shared brain."""
        for module in (self.encoder, self.ln, self.trunk, self.policy, self.value):
            yield from module.parameters()

    def forward(self, features):
        drive = self.encoder(features).t().contiguous()
        settled = self.settle(drive)
        hidden = self.trunk(self.ln(settled))
        logits = self.policy(hidden)
        return logits, self.value(hidden)

    def value_from_logits(self, value_logits):
        """Expected value under the predicted distribution over bins."""
        return (value_logits.softmax(-1) * self.bin_centres).sum(-1)

    def value_targets(self, values):
        """Two-hot targets: a value between bin centres splits across its two neighbours."""
        scaled = (values.clamp(0, 1) * self.value_bins - .5).clamp(0, self.value_bins - 1)
        lower = scaled.floor().long()
        upper = (lower + 1).clamp(max=self.value_bins - 1)
        weight = (scaled - lower.to(scaled.dtype)).unsqueeze(-1)
        target = torch.zeros(values.shape[0], self.value_bins, device=values.device,
                             dtype=torch.float32)
        target.scatter_(1, lower.unsqueeze(-1), 1 - weight)
        target.scatter_add_(1, upper.unsqueeze(-1), weight)
        return target


def masked_move_loss(logits, targets, legal_offsets=None, legal_indices=None):
    """Cross-entropy over all moves; legality is a scoring-time mask, not a training one."""
    return nn.functional.cross_entropy(logits.float(), targets)


def top1_against_legal(logits, targets, offsets, indices):
    """Fraction of positions whose highest-scoring *legal* move is Stockfish's choice."""
    offsets = torch.as_tensor(offsets, dtype=torch.long)
    indices = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long)
    targets = torch.as_tensor(targets, dtype=torch.long)
    hits = 0
    for position in range(targets.numel()):
        legal = indices[offsets[position]:offsets[position + 1]]
        if legal.numel() == 0:
            continue
        best = legal[logits[position, legal].argmax()]
        hits += int(best.item() == targets[position].item())
    return hits / max(targets.numel(), 1)
