# Delayed-memory training: no improvement in the first controlled run

The final-only memory objective did not improve earlier-word identity or order
in this 256-update experiment. Both the trainable-fast-rule treatment and the
matched decoder-only control selected **retention step 0**: every later
validation checkpoint had worse mean cross entropy. This is a negative result
for this training recipe and budget, not a capacity bound on the connectome.

Run: `retention-step3072-v1`, September 15, 2026. Starting language checkpoint:
step 3,072, SHA-256
`6527999333896856d0b8c0265ed02424385340f1c829ab12d20236d7bdb67797`.
The [protocol](RETENTION_PROTOCOL.md) was fixed before training.

## Training and validation

The treatment trained 8,838 existing plasticity parameters and 3,084 new linear
head parameters. The control trained identical heads from cached frozen-circuit
responses. Both trained with fast weights enabled, using the same examples,
head initialization, normalization, and head optimizer. The 166,700 neurons and
25,582,938 base edges, existing rate dynamics, and input/output interface were
preserved. The original language-training run continued separately.

There were 768 training episodes, 192 validation episodes, and 384 test episodes,
constructed from 224 fresh source-story groups with no source or full-context
overlap with the earlier frozen probe. The 256 batches of 16 supplied 4,096
training presentations. Each final input is held identical across all labels
within its context group. Labels enter only the memory loss, not neural inputs.

| Arm / update | Validation identity | Validation order | Mean validation CE |
|---|---:|---:|---:|
| Both / 0 | 10.0% | 53.125% | 1.5101 |
| Fast rules + heads / 256 | 10.0% | 50.0% | 1.7387 |
| Decoder only / 256 | 10.0% | 50.0% | 1.7356 |

Updates 64, 128, and 192 also scored 10% identity and 50% order in both arms.
Their validation mean CE exceeded step 0. The fixed training-monitor subset
also remained at chance after training. Probabilities and gradients differed
between treatment and control despite equal final argmax accuracy.

## Held-out tests of validation-selected checkpoints

Each task has 32 held-out context groups: 320 identity episodes and 64 order
episodes. Both arms restored step 0 and gave identical scores in each condition.
These tests evaluate the initialization before retention training; the original
language checkpoint was already trained.

| Condition, both selected arms | Identity | Order | Identity CE | Order CE |
|---|---:|---:|---:|---:|
| Fast on | 10.0% | 50.0% | 2.332114 | 0.722116 |
| Fast off throughout | 10.0% | 50.0% | 2.332784 | 0.722461 |
| Clear H immediately before prediction 8 | 10.0% | 50.0% | 2.332758 | 0.722446 |
| Chance accuracy / uniform CE | 10.0% | 50.0% | 2.302585 | 0.693147 |

The ablations cannot establish how the retention-trained step-256 rules use
memory, because validation rejected that checkpoint. Clearing only H also
leaves earlier fast-weight effects already carried in neural activity intact.

## Verification and scope

All 8,838 plasticity values and 3,084 new head values changed and remained finite.
Every trainable tensor received nonzero gradients. The base graph, interface,
original 2,570-value language head, and fixed calibration remained unchanged.
Independent wiring and checkpoint audits passed; 25 implementation/reporting
tests passed. Fourteen training/inference source files are archived with hashes.

Native MPS execution on macm3 took **1,991.8 seconds (33 minutes 12 seconds)**,
including cache extraction, preflight, training, validation, and six test passes.
The complete artifacts, checkpoints, cached features, audits, generated report,
and curves are in `results/retention-step3072-v1/` on both the local project and
macm3. Generate the dashboard with `scripts/report_retention.py`.

This is one seed, starting checkpoint, and optimization budget on a synthetic
memory task. It does not measure next-word accuracy or a benefit of biological
geometry over a randomized graph. The next discriminating test would first
establish learning at a shorter delay where the target is absent from the
two-word input, then increase that delay under a separate protocol.
