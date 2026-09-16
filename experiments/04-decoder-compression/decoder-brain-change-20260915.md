# Brain changes with the rank128 decoder

Audit performed on macm3 using CPU tensor reads only. Training continued; no model forward, optimizer step, GPU evaluation, or reserved-test evaluation was added. The current E minimum-CE and maximum-accuracy selectors both pointed to update8,300 at capture; latest was8,371. All comparisons below bind exact checkpoint hashes in the accompanying record.

## Preserved anatomy and learned dynamics

All49,393 neurons and9,050,172 canonical directed edges are preserved. Stored connection values, endpoints, zeros and signs match the fixed reference. Both E and earlier B train148,179 neuron parameters:49,393 each of gain, recurrent gain and bias. All148,179 differ numerically from initialization in the captured checkpoints; this count includes arbitrarily small optimizer changes. Neither run has trainable base-edge multipliers.

The common starting values are gain=1, bias=0 and recurrent gain equal to the reference normalization target divided by incoming absolute connection strength (or1 for rows without incoming strength). These reconstructed initial vectors match the recorded initialization hashes exactly. The published checkpoint already has learned neuron values, so its values are not the from-scratch baseline.

## Drift from the shared initialization

| Quantity | E reduced decoder, CE winner8,300 | B full decoder, CE winner9,700 | B full decoder, accuracy winner15,200 |
|---|---:|---:|---:|
| Neuron gain relative L2 | 44.50% | 60.47% | 73.67% |
| Recurrent gain relative L2 | 7.64% | 8.73% | 12.32% |
| Bias displacement RMS | 0.07421 | 0.06564 | 0.06999 |
| Effective recurrent matrix relative Frobenius | 43.61% | 59.21% | 72.01% |

These retained selectors have unequal training budgets. Parameter movement includes optimizer weight decay as well as gradient updates. Relative L2 means norm(current−initial)/norm(initial), not the percentage of neurons that changed. Bias started at zero, so its relative percentage is undefined.

At the matched7,700-update checkpoint in the validation logs, E versus B had gain displacement RMS0.424910 versus0.527244, recurrent-gain displacement RMS47.344208 versus47.335350, and gain×recurrent-gain displacement RMS173.732513 versus283.920593. E gain movement was19.41% smaller and product movement38.81% smaller. These aggregate logged distances do not supply a retained B7,700 tensor checkpoint for a direct vector difference.

## Effective recurrent matrix

Before tanh, recurrence uses M=diag(gain×rec_gain)W. Its relative Frobenius displacement is computed from incoming row sums of W², avoiding materializing a dense matrix. This weights each row by its actual connection strength. Leak and state-dependent tanh saturation are not included, so this is not a measured activity-manifold distance or the full state Jacobian.

At E update8,300,88.83% of the49,252 neurons with nonzero incoming connections changed their recurrent row multiplier by more than10%; the median absolute relative change was35.02%. The median signed change was−30.88%. Neuron gains are unconstrained:21 active destination rows reversed the sign of their effective recurrent multiplier, affecting7,555 incoming nonzero edges (about0.0835% of stored edges;0.2015% of absolute base-weight mass). The stored edge signs remain unchanged.

The ±10% bound belongs to the additional edge multipliers in queued F, not the existing gain/rec_gain values. Reducing the decoder improved validation without adding brain parameters or rewiring the graph; these drift measurements alone cannot establish that useful computational burden shifted to the brain.

The direct E-versus-B effective-matrix difference is29.37% for their CE winners and36.55% for their accuracy winners, relative to B, at unequal budgets. Raw per-row E/B ratios can have very large values because learned B multipliers approach zero; do not interpret those ratios as typical neuronal change. The global weighted matrix norm is more stable.

## Provenance

- Parameter/operator audit: [structured record](../records/decoder-brain-change-20260915.json).
- Full raw statistics and read-only analysis script: local results/connectorch-brain-change-v1 and macm3:/Users/fernando/fly_wordbrain_connectorch/results/connectorch-brain-change-v1.
- Shared source/data bindings were verified before and after analysis; checkpoint reads hash and load the same open file, handling atomic selector replacements.
- For a future functional comparison, evaluate controlled activity/state responses and normalization-preserving interventions on macm3 after the active trainer releases the GPU.
