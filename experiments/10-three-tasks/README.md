# Stage 10 — How many tasks fit in one brain?

**Objective.** [Stage 9](../09-two-tasks/README.md) put two tasks through one fly and
measured the price: sharing cost **0.4388 nats [0.4263, 0.4514]** and 6.24 accuracy points
against the language-only reference, and letting all 9,050,172 synapses move within ±10%
bought back only 3.1% of it. That establishes a cost. It does not say what shape the cost
has. One extra task could be a fixed toll — a constant amount of the brain's 148,179
per-neuron dynamics spent on arbitration — or it could compound, each task crowding the
next until the reservoir saturates.

Two points cannot tell those apart. Three can, so this stage adds a third task and asks
whether the second-task penalty repeats, shrinks or grows.

## The third task

Sentiment classification, on [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2).
It was chosen because it is *unlike* the first two in the way that matters here:

| | language | chess | sentiment |
|---|---|---|---|
| input | a token stream | a static board | a token stream |
| output | 1 of 1,024, every step | 1 of 1,968, once | 1 of 2, once |
| supervision | the next token | an engine's move | a human label |
| what the brain must hold | the recent past | a whole position | the whole sentence |

The last row is the interesting one. Language is scored every step, so the brain is never
asked to remember much: with `leak = 0.9`, 90% of the state is replaced per step and
whatever entered 26 steps ago has decayed to 1e-26. Chess injects a position across 8 delay
slots and settles for 5 steps, so the whole input is present at once. Sentiment is the only
task where the brain must carry information *across* a sequence and then answer about all
of it — the hardest thing to ask of a deliberately leaky reservoir.

### The readout follows from that

A first arm read sentiment from the final token's state, which is the conventional choice
and the wrong one here: at leak 0.9 the final state has almost no memory of the sentence's
opening. The shipped arm pools over the sequence instead, masked by length:

```python
mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
summary = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
```

Both arms were run; the comparison is in the results below.

## What is shared, and what the third task adds

The design is [Stage 9](../09-two-tasks/README.md)'s, widened by two classes:

| outputs | |
|---|---|
| `[0, 1024)` | byte-level BPE tokens |
| `[1024, 2992)` | the 1,968 chess moves |
| `[2992, 2994)` | negative, positive |

One brain, one `brain.in_proj`, one rank-64 trunk, one head over all 2,994 outputs. The
frozen connectome is untouched: `plasticity: fixed`, so this arm trains only the 148,179
per-neuron gain, rec_gain and bias values plus the task-specific adapters, and no synapse
moves. Stage 9 already showed that letting them move is worth about 3%, so the capacity
question is asked of the brain as measured.

Two things are new. A **task cue** — a learned vector per task added in embedding space —
and a **three-way router** on the trunk. The cue exists because three tasks through one
injection need some way to be distinguished at the input; the router remains a probe on the
output side, not a gate, under the same corruption test as before.

### The bug that cue introduced

Adding the cue meant passing `inputs_embeds` to the brain. The brain builds its 8-token
delay cache as `cat([previous, input_ids])[-k:] if input_ids is not None else previous`, so
passing embeddings *alone* leaves the cache frozen at its initial value forever — measured:
with ids `[4, 2]` the cache advanced, with embeddings alone it stayed `[0, 0]`. Every
language chunk would have been scored against a brain with no delay history. The fix is to
pass both, and the failure is recorded as its own lesson because nothing about it is
visible in a loss curve.

## Floors

The three accuracies are not comparable to each other, so each carries its own, and every
one is **recomputed from the split being scored** rather than written down:

| task | floor | value |
|---|---|---:|
| language | uniform over the token range | 1/1,024 |
| chess | commonest legal move | 12.33% |
| chess | uniform legal draw, E[1/n] | 8.48% |
| sentiment | majority class, counted from the split's labels | **50.92%** |

That last number is derived rather than asserted for a reason: an earlier version of this
analysis carried a hardcoded sentiment floor of 72.48% that appears in no record, corpus or
script in this repository. Against a split whose labels are 444 positive and 428 negative,
it turned a large margin into an apparent narrow miss. Floors are now counted, and the
chess floors are read from the measured baselines record, which must declare it was
measured on the same split or the run aborts.

### What bounds the sentiment number

SST-2 trains on treebank phrases and evaluates on whole sentences, so a training phrase can
sit inside an audit sentence without sharing its digest. Checked exactly by n-gram lookup
over all 60,000 training rows: **24 of 872 audit rows (2.75%)** contain a training phrase of
four words or more. This is a property of the dataset, not of this build, and it bounds what
the accuracy means. The audit split is the official SST-2 validation set, used because the
official test labels are hidden.

## Result

**The cost does not compound. It collapses.** The second task cost 6.4 times what the third
did, and the third cost chess nothing measurable at all.

All 16,800 updates, frozen synapses, one 2,994-output head. Scored on the same 200-story
audit population as every stage since 6, 10,000 held-out chess positions, and the 872-row
official SST-2 validation set.

| | language CE | language top-1 | chess top-1 *(legal-masked)* | sentiment |
|---|---:|---:|---:|---:|
| I — language alone | **2.9493** | **37.43%** | — | — |
| U — + chess | 3.3881 | 31.19% | **17.50%** | — |
| **T — + chess + sentiment** | 3.4571 | 30.31% | 17.45% | **71.79%** |
| floors | — | — | 12.33% | 50.92% |

### The shape of the cost

| step | language CE | 95% CI | language top-1 |
|---|---:|---|---:|
| adding chess to language | **+0.4388** | [0.4263, 0.4514] | −6.24pp |
| adding sentiment to both | **+0.0690** | [0.0577, 0.0801] | −0.88pp |

Both intervals exclude zero, so both costs are real. But the second task took **6.4×** what
the third took. Whatever the first act of sharing buys — arbitration, some reorganisation of
the 148,179 per-neuron dynamics into a form that can serve more than one master — appears to
be paid once rather than per task.

On chess the third task cost **0.05 accuracy points**, from 17.50% to 17.45%. That is far
inside the ±0.37pp standard error on 10,000 positions. Sentiment was, to measurement, free
for chess. Value MAE moved 0.1135 → 0.1131.

The T-minus-U contrast is paired on the same stories with the same estimator and seed as
every other interval here, and reports `language_range_only` **only**: T predicts over 2,994
classes and U over 2,992, so differencing their full-output cross-entropies would measure
head width, not skill. Both renormalise over the same 1,024 token logits.

### Sentiment works, comfortably

**71.79%** on the official SST-2 validation set against a **50.92%** majority-class floor
counted from that split's own labels — **+20.87 points**. On the 2,000-row internal
validation split it reaches 74.70%. The contamination bound does not threaten it: even
discarding all 24 audit rows that contain a four-word-or-longer training phrase as free
wins, the figure is 69.1%, still eighteen points clear.

Pooling mattered. The last-token arm plateaued several points lower, which is what the leak
predicts — at `leak = 0.9` the final state retains almost nothing of a sentence's opening,
so reading sentiment there asks the brain for memory it structurally does not have. Pooling
over the sequence asks instead for an average of local impressions, which a leaky reservoir
can supply.

### Leakage stayed negligible throughout

| task | mass outside its own range |
|---|---:|
| language | 0.0041 nats |
| chess | 0.0075 |
| sentiment | 0.00012 |

The uniform null is 1970/2994 = 0.658 for language. Three tasks share one output space, and
none of them puts meaningful mass in another's range, with nothing telling the model which
task it is looking at except the settled state of the brain.

### The router is *more* depth-dependent with three tasks, not less

This is where the three-task arm differs qualitatively from the two-task one, and it is the
opposite of what the accuracy alone suggests:

| | settle depth 5 | settle depth 32 |
|---|---:|---:|
| U — two tasks | 1.000 | 1.000 |
| **T — three tasks** | **0.515** | 1.000 |

The two-task router is depth-invariant. The three-task router is not: at depth 5 it is at
chance. Accuracy cannot say *how* it fails, so the probe now records where the mass goes,
and the answer was not the one worth guessing — language and sentiment are both token
streams, so the natural hypothesis is that they get confused. They do not:

| true task, depth 5 | called language | called chess |
|---|---:|---:|
| language | 3 | **97** |
| chess | 0 | 100 |

At five settle steps the router calls almost everything chess. **It has learned that five
settle steps means chess** — which is true of its training distribution, where chess is the
only task settled that shallowly. With two tasks that shortcut was not available, or not
worth taking. With three it is.

The linear probe still separates the states perfectly at depth 5 against a 51% shuffled-label
floor, so the information is there; the trained router simply stops using it when a cheaper
cue is present. That is a finding about the router, not the brain — and it is exactly why
the router is kept as a probe with a corruption test rather than read as evidence the brain
knows what it is doing. Magnitude remains unruled-out as before: norm alone scores 100%.

### What is still not measured

Unchanged from [Stage 9](../09-two-tasks/README.md): there is no single-task chess arm, so
the *chess* cost of sharing has no baseline. What this stage adds is that the **marginal**
chess cost of a third task is zero, which is a different claim and a weaker one.

Nor does this say where saturation is. Two steps establish that the curve bends sharply
after the first; they cannot locate its knee. A fourth task is the cheap next measurement.

Receipts: [`T32threetaskspooled.json`](../configs/T32threetaskspooled.json) (pooled) and
[`T32threetasks.json`](../configs/T32threetasks.json) (last-token), the two-task baseline in
[Stage 9](../09-two-tasks/README.md), chess baselines in
[`chess-baselines-v1.json`](../records/chess-baselines-v1.json).
