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

| arm | interfaces | brain | synapses | isolates | status |
|---|---|---|---|---|---|
| I *(Stage 6)* | private | language only | frozen | the language reference | **complete** |
| C | private | chess only | frozen | the chess reference | **stopped at 1,967 updates** |
| M | private | shared | frozen | cost of sharing the *dynamics* | **not run** |
| N | private | shared | ±10% | whether adapting the connectome buys it back | **not run** |
| **U** | **unified + router** | shared | frozen | cost of sharing the *interfaces* too | **complete** |
| **V** | **unified + router** | shared | ±10% | the same, with the connectome free to move | **complete** |

Only the two unified arms and the language reference completed. `M` and `N` were dropped
because the unified design subsumes them — they test the weaker form of sharing, and `U`
answered the strong form directly. `C` was stopped early to give the machine to the
three-task arm. That abandonment has a **cost recorded in the results below**: there is no
single-task chess arm at a matched budget, so the language side of the sharing question is
measured and the chess side is not.

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

**One brain holds two tasks. Sharing is not free, and adapting the connectome barely helps.**

Both unified arms trained all 16,800 updates with one encoder, one head and one
2,992-output space. Scored on the same 200-story audit population as every other stage
(45,059 next-token targets), and on 10,000 held-out chess positions.

| | language CE | language top-1 | chess top-1 *(legal-masked)* | value MAE | router |
|---|---:|---:|---:|---:|---:|
| I — language only *(Stage 6)* | **2.9493** | **37.43%** | — | — | — |
| U — unified, frozen synapses | 3.3881 | 31.19% | **17.50%** | 0.1135 | 100% |
| V — unified, ±10% synapses | **3.3743** | 31.35% | 17.21% | 0.1140 | 100% |
| naive floors | — | — | 12.33% *(commonest legal move)* | 0.1891 | — |

Language CE is `language_range_only` — the same targets against a softmax restricted to the
1,024 token logits — which is what compares to arms that never had a chess head.

### Sharing costs 0.44 nats

Adding chess to the same brain, the same injection and the same head costs
**+0.4388 nats [0.4263, 0.4514]** and **−6.24 accuracy points [−6.62, −5.86]** against the
language-only reference. That is a wide, unambiguous interference penalty: the two tasks
are genuinely competing for the same 148,179 per-neuron dynamics and the same rank-64
trunk, and the brain cannot serve both as well as it serves one.

This is the first stage where the *brain* is clearly the binding constraint. Stages 4–7
kept finding the readout did the work; here the readout is shared by construction, and the
cost shows up immediately.

### Adapting the connectome recovers 3% of it

The paired contrast — same stories, same estimator, same 10,000 resamples, seed 1729,
[`unified-v-minus-u.json`](../records/unified-v-minus-u.json):

| V minus U | delta | 95% CI |
|---|---:|---|
| language cross-entropy | **−0.0138** | [−0.0176, −0.0100] |
| language top-1 | +0.16pp | [−0.05pp, +0.38pp] |
| chess top-1 *(legal-masked)* | −0.29pp | *not paired; see below* |

Letting all 9,050,172 synapses move — through 18,322 bounded, sign-preserving cell-type
gains, `W ← W·(1 + 0.1·tanh(θ))` — buys **0.0138 nats**. The interval excludes zero, so the
effect is real, and it is **3.1% of the 0.4388-nat penalty it was meant to relieve**. The
accuracy interval straddles zero. On chess it is 0.29pp *worse*, which is smaller than the
0.38pp marginal standard error on 10,000 positions and was not measured as a paired
contrast, so it is best read as no effect rather than as a loss.

Put beside [Stage 7](../07-graph-control/README.md), the scale is almost comic:

| operation on the connectome | effect on language CE |
|---|---:|
| rewire all 9.05M edges at random *(Stage 7)* | +0.0100 [+0.0026, +0.0175] |
| adapt all 9.05M edges within ±10% *(here)* | −0.0138 [−0.0176, −0.0100] |

**Destroying the wiring and improving it are the same size, and both are tiny.** The
hypothesis that "modifying the connectome becomes interesting" when capacity is tight is
measurable, and the measurement says it is worth about one part in thirty of the squeeze.
The caveat that keeps this from being final is granularity: `bounded10` has one parameter
per cell type per side, not one per synapse, and the ChessFly reference trains a free
per-edge `W = sign·exp(θ)`. A per-edge arm could answer differently; this one cannot.

### Chess works, and is well short of the reference

17.50% legal-masked top-1 against a 12.33% commonest-legal-move prior and an 8.48% uniform
legal draw — **2.1× random**, and 5.2 points over the strongest naive baseline. Value MAE
0.1135 against a 0.1891 constant predictor. The model is really reading boards.

It is also well short of ChessFly's 30.4%, and three differences each plausibly account for
part of that: the reference trains a **free per-edge weight matrix** where this is frozen;
it uses a 138,639-neuron FlyWire connectome against this 49,393-neuron MaleCNS; and it saw
roughly 4.4M positions against this arm's 537,600 — **0.896 passes over the corpus, not
even one epoch**. None of those is separated here, so no claim is made about which matters.

### What is not measured

**The chess cost of sharing is unknown.** Arm C was stopped at 1,967 updates, so there is
no single-task chess arm at 16,800 updates to compare 17.50% against. The 0.4388-nat figure
above is the *language* cost of sharing; the symmetric number for chess does not exist. It
is the first thing to run once the machine is free.

### The router reads content, not depth

The router hits 100% on both arms, and that number is worth nothing on its own — it is
reached within a couple of hundred updates. The
[matched-settle-depth control](../records/unified-U32unifiedfixed-task-identity-probe.json)
re-measures with both tasks having run the recurrence the same number of steps, against a
shuffled-label floor fitted the same way:

| | probe held-out | shuffled-label floor | norm alone | direction alone |
|---|---:|---:|---:|---:|
| U, 5 steps | 100% | 51% | 100% | 100% |
| U, 32 steps | 100% | 51% | 100% | 100% |
| V, 5 steps | 100% | 57% | 94% | 100% |
| V, 32 steps | 100% | 49% | 99% | 100% |

Depth is ruled out — separability survives matched settle steps, and the floor confirms the
probe is not just memorising 49,393 dimensions from a few hundred samples. **Magnitude is
not ruled out.** Mean state norms differ by task (U: 78.9 language vs 86.9 chess) and a
probe given *only* the norm still scores 100% on U. Direction alone also scores 100%, so
the task identity is present in both, and this cannot say the router uses the content
rather than the gain. That is why the router is kept as a probe with a corruption test
rather than being read as evidence the brain "knows what it is doing".

Receipts: arm table [`multitask-v1-arms.json`](../configs/multitask-v1-arms.json), chess
baselines [`chess-baselines-v1.json`](../records/chess-baselines-v1.json), the task itself
in [Stage 8](../08-chess/README.md).
