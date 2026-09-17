"""One encoder, one decoder, one output space, and a router — for both tasks at once.

The two-head arm gives each task its own encoder and its own readout. Stages 4 to 7 all
found the heads do most of the work, so that arm can come back showing no interference for
an uninteresting reason: the tasks never really meet. Two private pathways over a shared
reservoir is barely one brain doing two things.

This removes the escape route. Both tasks are injected through the *same* projection into
the brain, and both are read out of the *same* head over a single output space:

    outputs   [0, 1024)      the byte-level BPE tokens the language task predicts
              [1024, 2992)   the 1,968 chess moves

Only two small adapters stay task-specific, and they are what makes the inputs commensurate
at all: a token embedding for language and a board projection for chess, each producing the
same shaped current that `brain.in_proj` then injects. Everything downstream is shared.

That shared output space is the point. Reading a story, the model must put no mass on chess
moves; looking at a board, none on words. Nothing tells it which situation it is in except
the settled state of the brain, so **the routing decision is real and it is learned**. A
router head is trained alongside as a *probe*: its accuracy measures whether the brain's
state encodes task identity, and it is never used to gate the output. Leakage -- the
probability mass landing in the other task's range -- is the headline measurement, and it
is what a language cross-entropy over 2,992 classes costs above one over 1,024.

A routing module that merely *chose* which encoder to run would measure nothing: the caller
already knows whether it passed token ids or a board, and the shapes differ. The hard
decision is at the output, so that is where it is made.
"""
import torch
from torch import nn

from .chess_encoding import FEATURES
from .chess_model import settle

LANGUAGE, CHESS = 0, 1
TOKENS, MOVES = 1024, 1968
OUTPUTS = TOKENS + MOVES


class UnifiedFly(nn.Module):
    def __init__(self, brain, settle_steps=5, readout_rank=64, value_bins=64,
                 features=FEATURES, tokens=TOKENS, moves=MOVES):
        super().__init__()
        # A plain attribute: the brain is shared, and registering it here would list its
        # parameters twice and duplicate every edge value into the checkpoint.
        object.__setattr__(self, "brain", brain)
        self.settle_steps, self.tokens, self.moves = settle_steps, tokens, moves
        self.value_bins = value_bins
        outputs = brain.out_index.numel()
        slots, width = brain.in_proj.shape[0], brain.in_proj.shape[1]
        # The board is presented to the brain as a static scene: one view per delay slot,
        # each the same width a token embedding has, so both tasks go through brain.in_proj.
        self.board_adapter = nn.Linear(features, slots * width, bias=False)
        self.ln = nn.LayerNorm(outputs)
        self.trunk = nn.Linear(outputs, readout_rank, bias=False)
        self.head = nn.Linear(readout_rank, tokens + moves, bias=False)
        self.value = nn.Linear(readout_rank, value_bins, bias=False)
        self.router = nn.Linear(readout_rank, 2, bias=False)
        self.register_buffer("bin_centres", (torch.arange(value_bins) + .5) / value_bins)

    def read(self, hidden):
        trunk = self.trunk(self.ln(hidden))
        return self.head(trunk), self.value(trunk), self.router(trunk)

    def language(self, input_ids, attention_mask=None, cache_params=None):
        """Uses the pinned brain's own forward pass; only the readout is ours."""
        out = self.brain(input_ids, attention_mask=attention_mask, cache_params=cache_params,
                         use_cache=True, return_dict=True)
        logits, value, router = self.read(out.last_hidden_state)
        return logits, router, out.cache_params

    def board_drive(self, features):
        slots, width = self.brain.in_proj.shape[0], self.brain.in_proj.shape[1]
        views = self.board_adapter(features).view(-1, slots, width)
        return torch.cat([views[:, j] @ self.brain.in_proj[j] for j in range(slots)], dim=-1)

    def chess(self, features, steps=None):
        drive = self.board_drive(features).t().contiguous()
        settled = settle(self.brain, drive, steps or self.settle_steps)
        return self.read(settled)

    def value_from_logits(self, value_logits):
        return (value_logits.softmax(-1) * self.bin_centres).sum(-1)

    def value_targets(self, values):
        scaled = (values.clamp(0, 1) * self.value_bins - .5).clamp(0, self.value_bins - 1)
        lower = scaled.floor().long()
        upper = (lower + 1).clamp(max=self.value_bins - 1)
        weight = (scaled - lower.to(scaled.dtype)).unsqueeze(-1)
        target = torch.zeros(values.shape[0], self.value_bins, device=values.device,
                             dtype=torch.float32)
        target.scatter_(1, lower.unsqueeze(-1), 1 - weight)
        target.scatter_add_(1, upper.unsqueeze(-1), weight)
        return target

    def leakage(self, logits, task):
        """Probability mass this output puts in the *other* task's range."""
        probabilities = logits.float().softmax(-1)
        other = probabilities[..., self.tokens:] if task == LANGUAGE \
            else probabilities[..., :self.tokens]
        return float(other.sum(-1).mean())

    def own_parameters(self):
        """Everything trained here except the shared brain."""
        for module in (self.board_adapter, self.ln, self.trunk, self.head, self.value, self.router):
            yield from module.parameters()


def move_targets(moves, tokens=TOKENS):
    """Chess targets live above the language range in the shared output space."""
    return moves + tokens
