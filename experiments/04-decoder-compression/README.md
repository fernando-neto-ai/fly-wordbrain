# Stage 4 — The readout was the whole model

**Objective.** Stage 2 established that **95.9% of everything this architecture learns is
the 50,578,432-parameter output matrix** sitting on top of 49,393 neurons. Stage 3 showed
the input interface can shrink 75% for free. So: replace the full readout with a
factorized low-rank one and find where quality breaks.

`W_out` (1024 × 49,393) becomes two bias-free matrices with no activation between them:
49,393 → `r` → 1,024, costing `r·(49393 + 1024)` parameters. **Every neuron still reaches
the readout.** Nothing about the brain changes.

## The arms

| Arm | Readout | Readout parameters | Total trainable | Status |
|---|---|---:|---:|---|
| [E32rank128fixed](../configs/E32rank128fixed.json) | rank 128 | 6,453,376 | 7,183,157 | accepted early stop |
| [F32rank128bounded](../configs/F32rank128bounded.json) | rank 128 + 18,322 edge gains | 6,453,376 | 7,201,479 | [stopped to correct ordering](F32rank128bounded-order-correction.md) |
| [G32rank64fixed](../configs/G32rank64fixed.json) | rank 64 | 3,226,688 | **3,956,469** | accepted early stop — **recommended** |
| [H32rank32fixed](../configs/H32rank32fixed.json) | rank 32 | 1,613,344 | **2,343,125** | accepted early stop — smaller option |

All keep width 32, eight explicit delays, 148,179 neuron parameters, 98,786 LayerNorm
parameters and the canonical 9,050,172-edge graph. **G and H share byte-identical frozen
graph buffers** — `w_values` SHA256 `e6408887…`, identical in both — so no edge was
modified in either.

## Result

Shrinking the readout did not cost quality. It **recovered** it.

Holding width 32 fixed so only the readout parameterization changes, B → G:

| | B32fixed, full readout | G32rank64fixed, rank 64 | change |
|---|---:|---:|---|
| trainable parameters | 51,308,213 | 3,956,469 | **12.97× fewer** |
| audit cross-entropy | 4.912468 | **3.280159** | **−1.632309 nats** |
| audit perplexity | 135.97 | **26.58** | **5.12× lower** |
| audit top-1 accuracy | 27.7237% | **33.3740%** | **+5.6504 points** |

The 50.6M-parameter readout was memorising 1,000 stories. Constraining its rank acts as a
regularizer, and the effect is large enough to move G *past the released reference model*
on a population that selected neither. See [Stage 5](../05-shared-population/README.md)
for that comparison with paired confidence intervals.

Rank 64 → 32 costs a little: H gives up **0.028935 nats** and **0.7945 accuracy points** on
the audit population for **1,613,344 fewer parameters**. On our selection-validation split
that gap reads as 1.4172 points. G remains the recommended baseline; H is the small option.

## Reports and receipts

Stop reports: [E](E32rank128fixed-accepted-early-stop.md) ·
[G](G32rank64fixed-accepted-early-stop.md) · [H](H32rank32fixed-accepted-early-stop.md).
Launches: [G](G32rank64fixed-launch.md) · [H](H32rank32fixed-launch.md).
Assessments: [post-G](rank64-post-G-assessment.md) · [post-H](rank32-post-H-assessment.md).
Unedited generations: [rank64](rank64-generated-texts.md) · [rank32](rank32-generated-texts.md),
reviewed in [rank64](rank64-text-review.md) and [rank32](rank32-text-review.md).
[Validation curves](figures/rank32-vs-rank64-validation.png). Full comparison in
[DECODER_REDUCTION.md](DECODER_REDUCTION.md).

## What this does not show

Text quality did not improve the way the numbers did. Across 24 G/H generation records
there are only **18 distinct texts**; neither model reliably retains a story premise, and
contiguous training-phrase overlap reached 12 words for H and 20 for G's accuracy winner.
Six greedy prompts do not establish generalization.

Lower cross-entropy with a smaller head is a statement about **this architecture's
readout**, not about the fly. Establishing that the connectome contributes anything needs
trained randomized-graph and zero-edge controls under this same recipe — they have not been
run. F was stopped at 2,600 updates to correct experiment ordering and **must not be
resumed automatically**; C and D remain deferred.
