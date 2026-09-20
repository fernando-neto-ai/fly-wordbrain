"""A task gate that reads the input, so the pipeline no longer has to be told the task.

The router inside `UnifiedFly` cannot do this job, and it is worth being precise about
why. The task cue is added in embedding space *before* the brain runs, so the settled
state already encodes which task the caller declared; the router then reads that back out
of the trunk. Its accuracy is therefore partly a measurement of whether the cue survived
the recurrence, not of whether the input is classifiable. Using it to choose a task would
be circular: you need the answer to apply the cue that produces the answer.

This module sits earlier. It sees the raw input, before any cue, and its output selects
the cue, the settle path and the output range. That makes routing a real decision.

Two honesty constraints are built in.

**Chess is dispatched by shape, not classified.** A position is 780 floats and a story is
a sequence of token ids; separating them is a type check, and reporting a three-way
accuracy that includes it would inflate the number with a free class. `classify` takes
token ids only, and `route` documents the shape dispatch as a dispatch.

**Length is a confound, and the pipeline's own chunking is the control.** Whole
TinyStories rows are 76-319 tokens and SST-2 sentences 3-116, so a gate given whole rows
scores in the nineties by reading length alone. The language task is trained and evaluated
in 32-token chunks, so that is what this sees: a window, not a row. `length_only_baseline`
fits the same classifier on nothing but the valid-token count, and the gate has to beat it
before any claim about content is warranted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LANGUAGE, CHESS, SENTIMENT, TOXICITY = 0, 1, 2, 3
# The tasks that arrive as token streams, in the class order the gate predicts. Chess is
# absent on purpose: it is 780 floats and is dispatched by shape, not classified.
TOKEN_TASKS = (LANGUAGE, SENTIMENT, TOXICITY)


class TaskGate(nn.Module):
    """Predicts which token-stream task a window of ids belongs to.

    Deliberately small: a mean over learned token embeddings, then a linear map. The point
    is to establish whether the input is separable at all, not to win a benchmark, and a
    large model here would make it impossible to tell which.
    """

    def __init__(self, tokens=1024, width=64, classes=len(TOKEN_TASKS), use_length=False,
                 task_ids=None):
        super().__init__()
        self.tokens, self.width, self.classes = tokens, width, classes
        self.use_length = use_length
        self.embedding = nn.Embedding(tokens, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width + (1 if use_length else 0), classes)
        # Which task each predicted class means. Kept as a buffer so a saved gate carries
        # its own mapping: a gate trained on two classes and read back as three would
        # silently relabel every prediction.
        ids = task_ids if task_ids is not None else TOKEN_TASKS[:classes]
        self.register_buffer("task_ids", torch.tensor(list(ids), dtype=torch.long))

    def summarise(self, input_ids, attention_mask):
        mask = attention_mask.to(self.embedding.weight.dtype).unsqueeze(-1)
        pooled = (self.embedding(input_ids) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.norm(pooled)

    def forward(self, input_ids, attention_mask):
        summary = self.summarise(input_ids, attention_mask)
        if self.use_length:
            length = attention_mask.to(summary.dtype).sum(1, keepdim=True) / input_ids.shape[1]
            summary = torch.cat([summary, length], dim=-1)
        return self.head(summary)

    def classify(self, input_ids, attention_mask):
        """Task id per row, over the token-stream tasks only."""
        choice = self.forward(input_ids, attention_mask).argmax(-1)
        return self.task_ids.to(choice.device)[choice]


class LengthOnlyGate(nn.Module):
    """The control: everything the gate could learn from how long the input is.

    If this scores as well as TaskGate, the gate is reading the chunking convention rather
    than the text, and the routing claim is empty.
    """

    def __init__(self, classes=len(TOKEN_TASKS)):
        super().__init__()
        self.head = nn.Linear(2, classes)

    def forward(self, input_ids, attention_mask):
        valid = attention_mask.to(self.head.weight.dtype).sum(1, keepdim=True)
        feature = torch.cat([valid, valid / input_ids.shape[1]], dim=-1)
        return self.head(feature)


def route(payload, gate, input_ids=None, attention_mask=None):
    """Decide a task for one payload, before any cue is applied.

    A board is 780 floats and a token stream is a sequence of ids, so chess is a **shape
    dispatch** and is reported as such -- it is not a classification result. Only the
    language/sentiment call is a prediction.
    """
    if payload is not None and torch.is_tensor(payload) and payload.dtype.is_floating_point:
        return torch.full((payload.shape[0],), CHESS, dtype=torch.long,
                          device=payload.device), "shape"
    if input_ids is None:
        raise ValueError("route needs either float board features or token ids")
    return gate.classify(input_ids, attention_mask), "classified"
