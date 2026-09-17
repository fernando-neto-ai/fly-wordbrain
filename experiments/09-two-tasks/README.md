# Stage 9 — Can one brain hold two tasks?

**Objective.** Every stage so far gave the connectome one job. This asks whether a single
brain can hold two, and the question is sharper than it looks because of exactly *what* is
shared. The graph is frozen, so it cannot be fought over. The only shared trainable surface
is **148,179 per-neuron gain, rec_gain and bias values** — the brain's dynamics. Both tasks
must agree on one setting of them.

[Stage 7](../07-graph-control/README.md) found the fly's *wiring* contributes about 1%. If
sharing the dynamics is also nearly free, this brain is close to a generic reservoir and
both tasks are being solved in their heads. If sharing is expensive, the tuned dynamics are
task-specific — and the obvious next lever is the 9,050,172 synapses sitting frozen.

## Two ways to share, and why the second is the real test

The first design gives each task a private encoder and a private readout over the shared
brain. It is the honest starting point, but it has an escape route: Stages 4 to 7 all found
the heads do most of the work, so that arm can report no interference simply because the
tasks never meet. Two private pathways over a shared reservoir is barely one brain doing
two things.

The second removes the escape route. Both tasks inject through the **same** `brain.in_proj`
and read out of the **same** head over a single output space:

| outputs | |
|---|---|
| `[0, 1024)` | the byte-level BPE tokens the language task predicts |
| `[1024, 2992)` | the 1,968 chess moves |

Only two small adapters stay task-specific, and they exist purely to make the inputs
commensurate before the shared injection: a token embedding, and a board projection that
presents the position as a static scene, one view per delay slot at the width a token
embedding has. Unification is also a compression — one trunk and one head instead of two
brings the task-specific parameters to **4,286,325** against the two-head arm's ~8.3M.

That shared output space is the measurement. Reading a story the model must put no mass on
moves; on a board, none on words. Nothing tells it which it is looking at except the
settled state of the brain, so the routing decision is real and it is learned.

## Arms

All matched to [`I32rank64fixed10k`](../configs/I32rank64fixed10k.json) on the language
side — same corpus, same 16,800 updates, same seed, same two-phase schedule — so the
language scores stay comparable to [Stage 6](../06-corpus-size/README.md)'s 2.9493.

| arm | interfaces | brain | synapses | isolates |
|---|---|---|---|---|
| I *(Stage 6)* | private | language only | frozen | the reference |
| C | private | chess only | frozen | the chess reference |
| M | private | shared | frozen | cost of sharing the *dynamics* |
| N | private | shared | ±10% | whether adapting the connectome buys it back |
| **U** | **unified + router** | shared | frozen | cost of sharing the *interfaces* too |
| **V** | **unified + router** | shared | ±10% | the same, with the connectome free to move |

`N` and `V` use the existing `bounded10` plasticity: every synapse gets a bounded,
sign-preserving multiplier `1 + 0.1·tanh(θ)`. That is a bounded form of what the ChessFly
reference trains — it learns `W_ij = sign_ij · exp(θ_ij)`, sign frozen — and it is the
direct test of whether modifying the connectome relieves a two-task squeeze.

## What is measured, and what is not trusted

**Leakage** is the headline: the probability mass a task's output puts in the other task's
range. It starts at the uniform null, 1024/2992 = 0.342, and it is exactly what a language
cross-entropy over 2,992 classes costs above one over 1,024. A unified arm can only match
arm I's 2.9493 once leakage is gone, which makes the comparison honest rather than
flattering.

**The router is a probe, not a gate.** It reads the trunk and predicts which task the brain
is in; a test corrupts it and checks the logits do not move, so it cannot quietly become a
crutch. A router that merely *chose* which encoder to run would measure nothing — the
caller already knows what it passed, and the shapes differ.

And its accuracy is **not taken at face value**. It reaches 100% within a couple of hundred
updates, which is far too fast: language chunks settle for 32 steps and chess positions for
5, so the two states differ in how long the recurrence has been running before anything
about words or boards is considered. A probe reading depth looks exactly like a probe
reading content. [`scripts/probe_task_identity.py`](../../scripts/probe_task_identity.py)
re-measures at matched settle depth and carries its own floor — a fresh linear probe fitted
on one half and scored on the other, against the same probe with shuffled labels. The
floor is load-bearing: the settled state has 49,393 dimensions against a few thousand
samples, so the training split separates whatever the labels are.

## Mixing the tasks inside a batch

Every update's gradient already contains both tasks — the two losses are summed before
`backward()`, so the optimizer never sees a single-task step. Whether they also sit in one
literal tensor turns out to be plumbing rather than science *here*, and it was measured
rather than argued: rows in this architecture do not interact. One batch of six gives
gradients identical to 2+4 accumulated to **3.8e-06**, and a row's logits do not change
when the batch around it does. There is no batch statistic anywhere — `W·x` is applied per
row and LayerNorm is per row. In most architectures the distinction matters; in this one it
provably cannot.

## Result

*Arms running. This section is filled in from the audit, not before.*

Receipts: arm table [`multitask-v1-arms.json`](../configs/multitask-v1-arms.json), chess
baselines [`chess-baselines-v1.json`](../records/chess-baselines-v1.json), the task itself
in [Stage 8](../08-chess/README.md).
