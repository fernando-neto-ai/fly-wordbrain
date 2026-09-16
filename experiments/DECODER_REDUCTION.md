# Decoder reduction after the width-32 encoder assessment

**Order corrected 2026-09-15:** the user intended rank64 to train after E.
The automatic F continuation was an ordering error and is superseded. G
launched after F cessation and its own numerical preflight. F
completion or quality assessment was not a prerequisite. Earlier E/F sequencing
below describes the historical plan only.

**Current state:** G is accepted as an early stop after 14 completed epochs,
16,808 observed / 16,800 durable updates. Its separate winners are validation
CE 3.093705 at 16,600 and accuracy 36.8885% at 15,552. The
[post-G assessment](reports/rank64-post-G-assessment.md),
[matched texts](reports/rank64-generated-texts.md) and
[text review](reports/rank64-text-review.md) report the completed E/G evaluation.
The user selected H32rank32fixed next. Its full macm3 run launched at
00:57:20 UTC on 2026-09-16 after separate rank32 parity and optimizer-smoke gates
passed. G is the preserved primary baseline. H's first full validation at 100
updates measured CE 5.827075 and accuracy 8.4484%; quality assessment remains pending.
See the [H launch report](reports/H32rank32fixed-launch.md).

The user authorized this pair on 2026-09-15: reduce the decoder as proposed,
retain the reduced encoder, and test whether modest brain adaptation recovers
any lost quality. The completed B assessment supports keeping encoder width32.
The former C/D full-head queue is superseded and must not resume automatically.

| Arm | Encoder | Decoder | Additional edge gains | Total trainable |
|---|---:|---:|---:|---:|
| B32fixed preserved baseline | 482,816 | 50,578,432 | 0 | 51,308,213 |
| E32rank128fixed | 482,816 | 6,453,376 | 0 | 7,183,157 |
| F32rank128bounded | 482,816 | 6,453,376 | 18,322 | 7,201,479 |

Both new readouts are bias-free linear factors, 49,393→128→1,024, with no
activation between them. They read every neuron. This removes44,125,056 decoder
parameters, approximately87.24% of the head. Encoder width32 and eight explicit
input delays remain unchanged. This tests decoder compression; it does not yet
test whether recurrence can replace the explicit input history.

All learned parameters initialize from scratch, seed42, using the existing
trainer. Shared E/F parameters must initialize identically; the only additional
F values are two zero-initialized cell-type gain vectors. Factorization also
changes initial logit scale relative to B, so B-versus-E is not an initialization-
matched isolated rank intervention. No SVD initialization or B warm start is used.

The49,393 neurons and9,050,172 canonical directed edge endpoints and stored
base values are retained. Both arms train the existing148,179 neuron parameters.
F adds18,322 source/destination cell-type gains with effective base-edge factors
within0.9–1.1; this does not constrain the original neuron gain/rec_gain parameters.
The count of shared gains is not the number or fraction of synapses affected.

Use the pinned reference, ConnecTorch revision, dataset and type groups declared
in each configuration. Train only on macm3's M3 GPU, serially. First verify
rank128 CPU/MPS output and gradient parity, exact graph identity and shared
initialization, then run eight optimizer-update smokes for E and F. Require
actual updates to both decoder factors and, for F, the bounded gain path.
Full training follows E then F with the unchanged30+14 epoch recipe,
batch8/TBPTT32, full100-story validation every100 updates and no final test.
Source files used by the earlier campaign remain unchanged. New controllers,
preflight and progress helpers have separate provenance and output directories.

Retain both minimum-validation-CE and maximum-validation-accuracy checkpoints,
with actual weights and selection receipts. The preserved B winners are CE4.512509
at update9,700 and accuracy33.313523% at update15,200. B stopped after16,262
observed/16,200 durable updates and13 completed epochs; it did not complete the
declared schedule. Compare matched updates and token budgets as well as independent
winners. Do not claim full-schedule parity from unequal early-stopped runs.

Review after at least eight completed epochs for a practical plateau: less than
0.10 improvement in global best CE and less than0.5 percentage points in global
best accuracy over2,000 updates, alongside training loss and equal validation
windows. Recheck immediately before any authorized stop. Preserve both winners,
latest state, exact process identities and the actual stop reason. Never restart
a failure silently or overlap GPU workers.

Assess E against B for compression cost and F against E for the effect of the
bounded gains. A practical quality-loss flag is more than0.10 CE or one accuracy
point at comparable budgets, or consistent text deterioration. These are review
thresholds, not statistical significance claims. Generate matching texts from
both retained winners, inspect training overlap, and retain the reserved test.
Only consider finer or less restricted brain adaptation after verifying the
existing gain path changes predictions and measuring its remaining deficit.
One seed and this small dataset cannot establish anatomical superiority.

## Accepted rank64 comparison

The user clarified that G32rank64fixed should train after E. It followed the
preserved E32rank128fixed baseline directly; the incorrectly launched F run
is stopped under that explicit correction, with its partial history retained.
It retains encoder width32, eight explicit delays, fixed canonical edges and the
same148,179 trained neuron values, data, seed42 and planned optimizer schedule.
The bias-free decoder becomes49,393→64→1,024:3,226,688 decoder parameters and
3,956,469 total. This is half the rank128 decoder size; neuron coverage is unchanged.

Compare G primarily against E at matched update/token budgets, retaining both
validation selectors and matching generation prompts. The existing initialization
policy changes initial logit variance with rank; disclose this when interpreting
early learning curves. A separate rank64 M3 CPU/MPS parity check and an
eight-update optimizer smoke passed before full training. Existing rank128 evidence
was not reused as a rank64 preflight. Model/trainer sources remain unchanged; the
obsolete F queue must not resume.

The configuration is experiments/configs/G32rank64fixed.json and its branch is
exp/encoder32-readout64-fixed. All actual training remains serial on macm3.
Additional edge adaptation remains unselected. Rank32 was subsequently selected
as the next decoder comparison, described below.

The launch-specific `scripts/stop_connectorch_rank64_plateau.py` performed the
accepted stop after fresh plateau, process-identity and coherent-checkpoint checks;
it never starts a successor and must not be rerun against the stopped arm.
Renewed accuracy gains had superseded the first plateau observation at
13,000 updates; that [partial report](reports/G32rank64fixed-progress.md) is
preserved as historical evidence. At 16,800, trailing 2,000-update global-best
gains were 0.014504 CE and 0.233154 accuracy percentage points, meeting both
thresholds. The [accepted-stop report](reports/G32rank64fixed-accepted-early-stop.md)
records preservation and the incomplete schedule.

After G's accepted stop and confirmed process cessation, the rank-aware
evaluator completed on idle macm3 with the following command. The output now
exists; any separately authorized repeat must use a fresh output directory.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-connectorch/bin/python \
  scripts/evaluate_connectorch_decoder_quality.py \
  --arm-e results/connectorch-decoder-v1/arms/E32rank128fixed \
  --arm-g results/connectorch-rank64-v1/arms/G32rank64fixed \
  --output results/connectorch-post-g-quality-v1 --threads 4
```

The completed MPS evaluation compares both retained selectors, identical text prompts, training
overlap, exact training-target budgets, and neuron changes from initialization
and between E/G. It preserves the reserved test and verifies graph/parameter
immutability during inference. The earlier encoder-only A/B evaluator remains
unchanged and is not the E/G entry point.

## Selected rank32 comparison

H32rank32fixed compares a rank32 readout against the accepted, stopped
G32rank64fixed baseline. It starts from scratch with the same seed42 recipe,
width32 encoder, eight explicit delays, tokenizer, dataset and optimizer schedule.
The bias-free linear factors are 49,393→32→1,024, with no intermediate activation.
Every neuron still contributes to the decoder.

| Component | G rank64 | H rank32 |
|---|---:|---:|
| Encoder | 482,816 | 482,816 |
| Decoder | 3,226,688 | 1,613,344 |
| Neuron gains and biases | 148,179 | 148,179 |
| Layer normalization | 98,786 | 98,786 |
| Total trainable | 3,956,469 | 2,343,125 |

This halves the decoder and removes 1,613,344 total parameters. H retains all
49,393 neuron identities, 9,050,172 canonical edge endpoints and stored base
weights. Existing neuron gain, recurrent-gain and bias values remain trainable;
no additional edge gains, unfreezing or rewiring are selected. Changing rank
also changes random head shapes and initial logit variance, so this is not a
strictly initialization-matched intervention or a warm start from G.

Rank32 native CPU/MPS parity passed on macm3: zero CE difference and maximum
gradient relative-L2 difference 3.052×10⁻⁵, with exact graph identity. The separate
eight-update optimizer smoke passed, verifying nonzero gradients and changes to
both decoder factors. Full training then launched at 00:57:20 UTC on 2026-09-16,
with MPS fallback disabled. These gates verify the execution path, not text quality.

The [configuration](configs/H32rank32fixed.json) is tracked on
`exp/encoder32-readout32-fixed`; outputs are under
`results/connectorch-rank32-v1`. The [launch report](reports/H32rank32fixed-launch.md)
and [launch receipt](runs/rank32-v1-launch.json) identify the actual run.
Refresh H against preserved G with `scripts/refresh_connectorch_rank32_progress.py`.

Retain separate minimum-CE and maximum-accuracy weights. Compare both independent
winners and common update/non-padding-target budgets on the unchanged 100-story,
21,874-target validation population. Apply the same plateau review and quality-loss
flags above, then inspect identical text prompts and training overlap after stopping.
The reserved test remains unused. No further experiment is queued by this rank32
selection, and a single seed cannot establish an anatomical language advantage.
