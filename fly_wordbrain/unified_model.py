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

LANGUAGE, CHESS, SENTIMENT, TOXICITY = 0, 1, 2, 3
TASKS = 3
TOKENS, MOVES, CLASSES = 1024, 1968, 2
TOXICITY_CLASSES = 0
OUTPUTS = TOKENS + MOVES + CLASSES

# Which tasks are a pooled judgement over a token stream. Both arrive through the same
# tokenizer, embedding and injection as language, and differ only in the question asked
# and the range their answer is read from.
POOLED_TASKS = (SENTIMENT, TOXICITY)


class UnifiedFly(nn.Module):
    def __init__(self, brain, settle_steps=5, readout_rank=64, value_bins=64,
                 features=FEATURES, tokens=TOKENS, moves=MOVES, classes=CLASSES,
                 tasks=TASKS, task_cue=True, sentiment_pooling="mean",
                 toxicity_classes=TOXICITY_CLASSES):
        super().__init__()
        # A plain attribute: the brain is shared, and registering it here would list its
        # parameters twice and duplicate every edge value into the checkpoint.
        object.__setattr__(self, "brain", brain)
        self.settle_steps, self.tokens, self.moves = settle_steps, tokens, moves
        self.classes, self.value_bins = classes, value_bins
        # Zero by default, so a three-task arm's head stays exactly `tokens + moves +
        # classes` wide and its checkpoints keep restoring. A fourth task widens the head
        # only for arms whose config declares it.
        self.toxicity_classes = toxicity_classes
        if sentiment_pooling not in ("mean", "last"):
            raise ValueError("sentiment_pooling must be 'mean' or 'last'")
        self.sentiment_pooling = sentiment_pooling
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
        self.head = nn.Linear(readout_rank, tokens + moves + classes + toxicity_classes,
                              bias=False)
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
        """Uses the pinned brain's own forward pass; only the readout is ours.

        `input_ids` is passed alongside `inputs_embeds` even though the embeddings already
        carry the drive. The brain builds its next cache as
        `cat([previous, input_ids])[-delay_k:] if input_ids is not None else previous`, so
        handing it embeddings alone silently freezes the eight-token delay history at the
        padding value: every chunk after the first would be driven by tokens that were
        never read. The embeddings decide the drive, the ids decide the history, and both
        are needed.
        """
        out = self.brain(input_ids=input_ids, inputs_embeds=self.embed(input_ids, LANGUAGE),
                         attention_mask=attention_mask, cache_params=cache_params,
                         use_cache=True, return_dict=True)
        logits, value, router = self.read(out.last_hidden_state)
        return logits, router, out.cache_params

    def sentiment(self, input_ids, attention_mask):
        """One judgement per sequence, from the whole sequence rather than its last token.

        The sequence is settled exactly as language is -- same embedding, same injection,
        same recurrence -- and differs only in the task cue and in being read once instead
        of at every step.

        **Why the mean and not the last token.** The recurrence has leak 0.9, so each step
        replaces ninety per cent of the state: a state retains 1e-4 of itself after four
        steps and 1e-26 after twenty-six. Reading only the final position therefore judges
        a twenty-six-token sentence on roughly its last handful of tokens plus whatever the
        eight delay slots carry, while the bag-of-tokens floor it has to beat sees every
        word. Averaging the settled state over the real positions gives the classifier the
        whole sentence. Padding is excluded from both the sum and the divisor, so a short
        sequence is not diluted by the batch's longest row.

        `sentiment_pooling="last"` keeps the original behaviour so the arms trained that
        way stay reproducible.
        """
        return self.classify(input_ids, attention_mask, SENTIMENT)

    def classify(self, input_ids, attention_mask, task):
        """Any pooled judgement over a token stream: sentiment, toxicity, the next one.

        Sentiment and toxicity differ in exactly two things -- the cue added at the input
        and the range the answer is read from -- so they share this path rather than each
        carrying a near-copy of it. A fourth task that duplicated the forward pass would
        be a place for the two to drift apart silently.
        """
        if task not in POOLED_TASKS:
            raise ValueError(f"classify is for pooled token-stream tasks, not task {task}")
        start, stop = self.ranges()[task]
        if stop <= start:
            raise ValueError(f"task {task} has no output range on this arm; "
                             f"its config did not declare one")
        out = self.brain(input_ids=input_ids, inputs_embeds=self.embed(input_ids, task),
                         attention_mask=attention_mask, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state
        if self.sentiment_pooling == "last":
            last = attention_mask.to(torch.long).sum(dim=1) - 1
            summary = hidden[torch.arange(input_ids.shape[0], device=last.device), last]
        else:
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            summary = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        logits, _, router = self.read(summary)
        return logits, router

    def toxicity(self, input_ids, attention_mask):
        return self.classify(input_ids, attention_mask, TOXICITY)

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
        sentiment_end = self.tokens + self.moves + self.classes
        return {LANGUAGE: (0, self.tokens),
                CHESS: (self.tokens, self.tokens + self.moves),
                SENTIMENT: (self.tokens + self.moves, sentiment_end),
                TOXICITY: (sentiment_end, sentiment_end + self.toxicity_classes)}

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


def pooled_targets(labels, unified, task):
    """Shift class labels into whichever range this task owns.

    Each pooled task's answer is a class index, but the loss is taken over the whole
    shared head, so the label has to be offset to the task's own slice. Deriving the
    offset from `ranges()` rather than restating `tokens + moves` means a fourth task
    cannot quietly land on the third's outputs.
    """
    start, stop = unified.ranges()[task]
    if stop <= start:
        raise ValueError(f"task {task} has no output range on this arm")
    return labels + start
