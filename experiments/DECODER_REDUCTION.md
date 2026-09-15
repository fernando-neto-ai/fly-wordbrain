# Decoder reduction after the width-32 encoder assessment

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
