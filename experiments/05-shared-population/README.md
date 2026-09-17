# Stage 5 — One population, every model

**Objective.** Make the numbers comparable. Until this stage they were not.

Every arm report replayed the **100-story selection-validation split** (21,874 targets).
The released-reference audit used a **separate 200-story population** (45,059 targets).
Reading 3.09 against 3.99 across those two tables would have been meaningless. This stage
scores the released reference, our full-size reconstruction and five preserved arms on
**both** populations in one inference-only run.

Run it with [`scripts/evaluate_compression_curve.py`](../../scripts/evaluate_compression_curve.py);
arms are declared in [`configs/compression-curve-arms.json`](../configs/compression-curve-arms.json),
generated from the accepted-stop receipts so no path or hash is typed by hand.

## Correctness gate

The evaluator reproduces the previously published audit numbers exactly — released
**3.988231 / 31.3833%**, reconstruction **5.036168 / 26.5807%** — and every arm's
validation score replays its accepted-stop receipt to within **4.2e-08**. No backward
pass, no optimizer, no parameter or graph mutation; the reserved 100-story test stays
sealed. Gates are in [`records/compression-curve-v1.json`](../records/compression-curve-v1.json).

![Held-out quality against trainable parameter count](figures/compression-curve.png)

## Result

| Arm | Trainable | Audit CE | Audit PPL | Audit acc | ΔCE vs released (95% CI) | Δacc (95% CI) |
|---|---:|---:|---:|---:|---|---|
| released reference | 52,756,661 | 3.988231 | 53.96 | 31.3833% | — | — |
| our reconstruction | 52,756,661 | 5.036168 | 153.88 | 26.5807% | +1.0479 [+1.015, +1.081] | −4.80 pp |
| A128fixed | 52,756,661 | 5.050346 | 156.08 | 27.4795% | +1.0621 [+1.027, +1.098] | −3.90 pp |
| B32fixed | 51,308,213 | 4.912468 | 135.97 | 27.7237% | +0.9242 [+0.890, +0.959] | −3.66 pp |
| E32rank128fixed | 7,183,157 | 3.373062 | 29.17 | 32.6350% | **−0.6152** [−0.643, −0.588] | **+1.25 pp** [+0.86, +1.66] |
| **G32rank64fixed** | **3,956,469** | **3.280159** | **26.58** | **33.3740%** | **−0.7081** [−0.738, −0.679] | **+1.99 pp** [+1.61, +2.37] |
| G32rank64fixed (acc selector) | 3,956,469 | 3.272451 | 26.38 | 33.4961% | **−0.7158** [−0.746, −0.686] | **+2.11 pp** [+1.70, +2.53] |
| H32rank32fixed | 2,343,125 | 3.309095 | 27.36 | 32.5795% | **−0.6791** [−0.711, −0.647] | **+1.20 pp** [+0.78, +1.62] |

Paired whole-story bootstrap, 10,000 resamples, seed 1729. Full table with both
populations: [shared-population-comparison.md](shared-population-comparison.md).

**The split is by readout, not by size.** Every full-readout arm lands *above* the released
reference on cross-entropy; every low-rank arm lands *below* it, with confidence intervals
that exclude zero on both metrics. G reaches −0.708 nats and +1.99 accuracy points against
a model **13.33× its size**; H does it at **22.52×** smaller.

## How to read this honestly

- **The clean internal contrast is B → G** (Stage 4): same data, seed, recipe, graph and
  encoder width, only the readout parameterization differs. That one is fully controlled.
- **The comparison against the released model is not controlled.** Its training stories,
  trainer and tokenizer-fitting population are unknown. Our arms were trained on our own
  1,000 TinyStories under our reconstructed recipe.
- If the released model's training set overlaps either population, its score here is
  **flattered**, which makes our margin conservative rather than inflated. We cannot verify
  this either way.
- The validation column selected our checkpoints and is not a held-out estimate for our
  arms. The audit column selected none of them, but it was revealed by an earlier audit and
  is no longer untouched.
- **None of this is evidence of an anatomical prior.** Every arm here uses the same frozen
  connectome, so nothing in this table separates "the fly's wiring helps" from "this
  recurrent shape plus a rank-constrained readout suits 1,000 short stories".
  [Stage 7](../07-graph-control/README.md) settled it separately with a randomised-graph
  arm: the wiring is worth **0.0100 nats [+0.0026, +0.0175]**, about 1% of the margin in
  this table. A zero-edge control, multiple seeds and a declared final-test protocol remain
  outstanding.
