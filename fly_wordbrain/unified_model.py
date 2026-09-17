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

LANGUAGE, CHESS, SENTIMENT = 0, 1, 2
TASKS = 3
TOKENS, MOVES, CLASSES = 1024, 1968, 2
OUTPUTS = TOKENS + MOVES + CLASSES


class UnifiedFly(nn.Module):
    def __init__(self, brain, settle_steps=5, readout_rank=64, value_bins=64,
                 features=FEATURES, tokens=TOKENS, moves=MOVES, classes=CLASSES,
                 tasks=TASKS, task_cue=True):
        super().__init__()
        # A plain attribute: the brain is shared, and registering it here would list its
        # parameters twice and duplicate every edge value into the checkpoint.
        object.__setattr__(self, "brain", brain)
        self.settle_steps, self.tokens, self.moves = settle_steps, tokens, moves
        self.classes, self.value_bins = classes, value_bins
        # Sentiment arrives through the same tokenizer, embedding and injection as
        # language, so nothing in the input says which question is being asked. A learned
        # vector per task is added in embedding space, which is the smallest cue the brain
        # can condition on: d_embed numbers per task, no tokenizer change, and it reaches
        # the neurons through the pinned in_proj like any token.
        #
        # Optional, because the two-task arms were trained before it existed. Creating it
        # unconditionally would add a parameter their checkpoints do not have, and
        # restore_parameters demands an exact key match -- so scoring those arms would
        # fail after the fact. An arm's config records which form it used.
        self.task_cue = task_cue
        self.task_vector = nn.Embedding(tasks, brain.in_proj.shape[1]) if task_cue else None
        outputs = brain.out_index.numel()
        slots, width = brain.in_proj.shape[0], brain.in_proj.shape[1]
        # The board is presented to the brain as a static scene: one view per delay slot,
        # each the same width a token embedding has, so both tasks go through brain.in_proj.
        self.board_adapter = nn.Linear(features, slots * width, bias=False)
        self.ln = nn.LayerNorm(outputs)
        self.trunk = nn.Linear(outputs, readout_rank, bias=False)
        self.head = nn.Linear(readout_rank, tokens + moves + classes, bias=False)
        self.value = nn.Linear(readout_rank, value_bins, bias=False)
        # One output per task. A two-task arm's router is 2 wide and a three-task arm's is
        # 3, so an arm's task count is part of the form a scorer has to rebuild.
        self.router = nn.Linear(readout_rank, tasks, bias=False)
        self.register_buffer("bin_centres", (torch.arange(value_bins) + .5) / value_bins)

    def read(self, hidden):
        trunk = self.trunk(self.ln(hidden))
        return self.head(trunk), self.value(trunk), self.router(trunk)

    def cue(self, task, device):
        if self.task_vector is None:
            return 0.0
        return self.task_vector(torch.full((1,), task, dtype=torch.long, device=device))

    def embed(self, input_ids, task):
        """Token embeddings carrying the task cue, ready for the pinned forward pass."""
        return self.brain.wte(input_ids) + self.cue(task, input_ids.device)

    def language(self, input_ids, attention_mask=None, cache_params=None):
        """Uses the pinned brain's own forward pass; only the readout is ours."""
        out = self.brain(inputs_embeds=self.embed(input_ids, LANGUAGE),
                         attention_mask=attention_mask, cache_params=cache_params,
                         use_cache=True, return_dict=True)
        logits, value, router = self.read(out.last_hidden_state)
        return logits, router, out.cache_params

    def sentiment(self, input_ids, attention_mask):
        """One judgement per sequence, read at its last real token.

        The whole sequence is settled exactly as language is -- same embedding, same
        injection, same recurrence -- and differs only in the task cue and in reading once
        at the end instead of at every step. Padding is excluded by index rather than
        masked afterwards, so a short sequence is never read at a padded position.
        """
        out = self.brain(inputs_embeds=self.embed(input_ids, SENTIMENT),
                         attention_mask=attention_mask, use_cache=False, return_dict=True)
        last = attention_mask.to(torch.long).sum(dim=1) - 1
        final = out.last_hidden_state[torch.arange(input_ids.shape[0], device=last.device), last]
        logits, _, router = self.read(final)
        return logits, router

    def board_drive(self, features):
        slots, width = self.brain.in_proj.shape[0], self.brain.in_proj.shape[1]
        views = self.board_adapter(features).view(-1, slots, width) \
            + self.cue(CHESS, features.device)
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

    def ranges(self):
        """Where each task's outputs live in the shared space."""
        return {LANGUAGE: (0, self.tokens),
                CHESS: (self.tokens, self.tokens + self.moves),
                SENTIMENT: (self.tokens + self.moves, self.tokens + self.moves + self.classes)}

    def leakage(self, logits, task):
        """Probability mass this output puts outside its own task's range."""
        start, stop = self.ranges()[task]
        probabilities = logits.float().softmax(-1)
        return float((1.0 - probabilities[..., start:stop].sum(-1)).mean())

    def own_parameters(self):
        """Everything trained here except the shared brain."""
        modules = [self.board_adapter, self.ln, self.trunk, self.head, self.value, self.router]
        if self.task_vector is not None:
            modules.append(self.task_vector)
        for module in modules:
            yield from module.parameters()


def move_targets(moves, tokens=TOKENS):
    """Chess targets live above the language range in the shared output space."""
    return moves + tokens


def sentiment_targets(labels, tokens=TOKENS, moves=MOVES):
    """Sentiment targets live above the chess range."""
    return labels + tokens + moves
